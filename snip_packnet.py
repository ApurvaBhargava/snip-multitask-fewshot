import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from utils import set_seed
from helpers.snip_helper import compute_snip_scores
from models.resnet18 import ResNet18_CIFAR
from models.resnet20 import ResNet20_CIFAR
from models.simpleconvnet import SimpleConvNet
from models.vgg11 import VGG11_CIFAR
from models.wideresnet import WideResNet_CIFAR


def get_model_factory(model_name, num_classes):
    name = model_name.lower()
    factory = {
        "resnet18": lambda: ResNet18_CIFAR(num_classes=num_classes),
        "resnet20": lambda: ResNet20_CIFAR(num_classes=num_classes),
        "simpleconv": lambda: SimpleConvNet(num_classes=num_classes),
        "vgg11": lambda: VGG11_CIFAR(num_classes=num_classes),
        "wideresnet": lambda: WideResNet_CIFAR(depth=16, widen_factor=2, num_classes=num_classes),
    }
    if name not in factory:
        raise ValueError(f"Unknown model: {model_name}. Choices: {list(factory.keys())}")
    return factory[name]


def split_cifar100_into_tasks(dataset, num_tasks=5):
    classes_per_task = 100 // num_tasks
    subsets = []
    targets = np.array(dataset.targets)
    for task_id in range(num_tasks):
        cls_start = task_id * classes_per_task
        cls_end = cls_start + classes_per_task
        mask = np.isin(targets, list(range(cls_start, cls_end)))
        indices = np.where(mask)[0]
        subsets.append(Subset(dataset, indices.tolist()))
    return subsets


def make_dataloader(dataset, batch_size, shuffle, num_workers, device, drop_last=False):
    kwargs = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    if device.type in {"cuda", "mps"}:
        kwargs["pin_memory"] = True
        kwargs["pin_memory_device"] = device.type
    return DataLoader(dataset, **kwargs)


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
    return 100.0 * correct / max(total, 1)


def init_mask_like(model, fill_value):
    return {
        name: torch.full_like(param, fill_value, device=param.device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def allocate_task_mask(model, loader, device, free_masks, keep_ratio, loss_fn):
    scores = compute_snip_scores(model, loader, loss_fn, device, mode="labeled")
    masked_scores = {}
    total_free = 0
    for name, score in scores.items():
        available = free_masks[name]
        masked = score.to(device) * available
        masked_scores[name] = masked
        total_free += int(available.sum().item())

    if total_free == 0:
        raise RuntimeError("No free weights left to allocate for the next task.")

    k = max(1, int(total_free * keep_ratio))
    flat_scores = torch.cat([masked_scores[name].view(-1) for name in masked_scores])
    topk_vals, _ = torch.topk(flat_scores, k)
    threshold = topk_vals[-1]

    task_mask = {}
    for name, masked in masked_scores.items():
        mask = (masked >= threshold).float() * free_masks[name]
        task_mask[name] = mask
    return task_mask


def apply_packnet_constraints(model, train_mask, frozen_mask, frozen_weights):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in train_mask:
                continue
            mask_train = train_mask[name]
            mask_frozen = frozen_mask[name]
            mask_other = 1.0 - mask_train - mask_frozen

            # Keep frozen weights fixed
            if mask_frozen.sum() > 0:
                param.data = param.data * (1 - mask_frozen) + frozen_weights[name] * mask_frozen

            # Ensure unallocated weights stay zero
            if mask_other.sum() > 0:
                param.data = param.data * (1 - mask_other)


def train_task_with_mask(model, train_loader, device, epochs, lr, train_mask, frozen_mask, frozen_weights):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    autocast_enabled = device.type in {"cuda", "mps"}
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.bfloat16

    apply_packnet_constraints(model, train_mask, frozen_mask, frozen_weights)

    for _ in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is None or name not in train_mask:
                        continue
                    grad_mask = train_mask[name]
                    param.grad *= grad_mask

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            apply_packnet_constraints(model, train_mask, frozen_mask, frozen_weights)

        scheduler.step()

    with torch.no_grad():
        for name, param in model.named_parameters():
            frozen_mask[name] = frozen_mask[name] + train_mask[name]
            frozen_weights[name] = param.detach().clone() * frozen_mask[name]
            # update train mask as those weights become frozen
            train_mask[name] = torch.zeros_like(train_mask[name], device=device)


def sequential_baseline(model_factory, train_loaders, test_loaders, device, epochs, lr):
    model = model_factory().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loaders))
    autocast_enabled = device.type in {"cuda", "mps"}
    dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_acc = np.zeros(len(train_loaders))

    for task_id, train_loader in enumerate(train_loaders):
        for _ in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                scheduler.step()

        best_acc[task_id] = evaluate_model(model, test_loaders[task_id], device)

    final_acc = []
    forgetting = []
    for idx, test_loader in enumerate(test_loaders):
        acc = evaluate_model(model, test_loader, device)
        final_acc.append(acc)
        forgetting.append(max(0.0, best_acc[idx] - acc))

    return {
        "model": model,
        "final_acc": final_acc,
        "avg_acc": float(np.mean(final_acc)),
        "avg_forgetting": float(np.mean(forgetting)),
        "per_task_forgetting": forgetting,
    }


def packnet_with_snip(model_factory, train_loaders, test_loaders, device, args):
    model = model_factory().to(device)
    loss_fn = nn.CrossEntropyLoss()
    free_masks = init_mask_like(model, 1.0)
    frozen_mask = init_mask_like(model, 0.0)
    frozen_weights = init_mask_like(model, 0.0)
    keep_ratio = 1.0 - args.snip_sparsity

    best_acc = np.zeros(len(train_loaders))

    for task_id, train_loader in enumerate(train_loaders):
        print(f"\n[PackNet] Allocating mask for task {task_id + 1}")
        task_mask = allocate_task_mask(model, train_loader, device, free_masks, keep_ratio, loss_fn)
        for name in free_masks:
            free_masks[name] = torch.clamp(free_masks[name] - task_mask[name], min=0.0)

        train_task_with_mask(
            model,
            train_loader,
            device,
            epochs=args.epochs,
            lr=args.lr,
            train_mask=task_mask,
            frozen_mask=frozen_mask,
            frozen_weights=frozen_weights
        )

        best_acc[task_id] = evaluate_model(model, test_loaders[task_id], device)
        print(f"[PackNet] Task {task_id + 1} accuracy: {best_acc[task_id]:.2f}%")

    final_acc = []
    forgetting = []
    for idx, test_loader in enumerate(test_loaders):
        acc = evaluate_model(model, test_loader, device)
        final_acc.append(acc)
        forgetting.append(max(0.0, best_acc[idx] - acc))

    return {
        "model": model,
        "final_acc": final_acc,
        "avg_acc": float(np.mean(final_acc)),
        "avg_forgetting": float(np.mean(forgetting)),
        "per_task_forgetting": forgetting
    }


def main():
    parser = argparse.ArgumentParser(description="SNIP-based PackNet continual learning on CIFAR-100")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--snip_sparsity", type=float, default=0.8)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="expt2_results.json")
    args = parser.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Running on device: {device}")

    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)

    train_tasks = split_cifar100_into_tasks(train_dataset)
    test_tasks = split_cifar100_into_tasks(test_dataset)

    train_loaders = [
        make_dataloader(task, args.batch_size, shuffle=True, num_workers=args.workers, device=device, drop_last=True)
        for task in train_tasks
    ]
    test_loaders = [
        make_dataloader(task, args.batch_size, shuffle=False, num_workers=args.workers, device=device, drop_last=False)
        for task in test_tasks
    ]

    model_factory = get_model_factory(args.model, num_classes=100)

    results = {
        "baseline": [],
        "packnet": []
    }

    for run_id in range(args.runs):
        print(f"\n================ Run {run_id + 1}/{args.runs} ================")
        set_seed(args.seed + run_id)

        baseline_metrics = sequential_baseline(
            model_factory,
            train_loaders,
            test_loaders,
            device,
            epochs=args.epochs,
            lr=args.lr
        )
        results["baseline"].append(baseline_metrics)
        print(f"Baseline avg acc: {baseline_metrics['avg_acc']:.2f}%, "
              f"avg forgetting: {baseline_metrics['avg_forgetting']:.2f}%")

        set_seed(args.seed + run_id)  # reset before packnet run
        packnet_metrics = packnet_with_snip(
            model_factory,
            train_loaders,
            test_loaders,
            device,
            args
        )
        results["packnet"].append(packnet_metrics)
        print(f"PackNet avg acc: {packnet_metrics['avg_acc']:.2f}%, "
              f"avg forgetting: {packnet_metrics['avg_forgetting']:.2f}%")

    summary = {}
    for key, runs in results.items():
        summary[key] = {
            "avg_acc": float(np.mean([r["avg_acc"] for r in runs])),
            "avg_forgetting": float(np.mean([r["avg_forgetting"] for r in runs])),
            "runs": runs
        }

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()

