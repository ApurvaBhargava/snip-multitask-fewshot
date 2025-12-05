import argparse
import json
import os
from collections import defaultdict

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


def make_dataloader(dataset, batch_size, shuffle, num_workers, device):
    kwargs = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False)
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    if device.type in {"cuda", "mps"}:
        kwargs["pin_memory"] = True
        kwargs["pin_memory_device"] = device.type
    return DataLoader(dataset, **kwargs)


def split_cifar100_into_tasks(dataset, num_tasks=5):
    classes_per_task = 100 // num_tasks
    subsets = []
    targets = np.array(dataset.targets)
    for task_id in range(num_tasks):
        cls_start = task_id * classes_per_task
        cls_end = cls_start + classes_per_task
        task_classes = list(range(cls_start, cls_end))
        indices = np.where(np.isin(targets, task_classes))[0]
        subsets.append(Subset(dataset, indices.tolist()))
    return subsets


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


def compute_fisher_information(model, dataloader, device, loss_fn, max_samples):
    fisher = {}
    model.eval()
    total_seen = 0
    for name, param in model.named_parameters():
        fisher[name] = torch.zeros_like(param, device=device)

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        model.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += (param.grad.detach() ** 2)

        total_seen += inputs.size(0)
        if total_seen >= max_samples:
            break

    for name in fisher:
        fisher[name] /= max(total_seen, 1)
    return fisher


def snapshot_params(model):
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def compute_penalty(model, memory, lam):
    if lam <= 0 or not memory:
        return torch.zeros((), device=next(model.parameters()).device)

    penalty = torch.zeros((), device=next(model.parameters()).device)
    for record in memory:
        importance = record["importance"]
        params_ref = record["params"]
        for name, param in model.named_parameters():
            if name not in importance:
                continue
            penalty = penalty + (importance[name] * (param - params_ref[name]) ** 2).sum()
    return 0.5 * lam * penalty


def train_task(model, train_loader, device, epochs, lr, penalty_records=None, lam=0.0):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    autocast_enabled = device.type in {"cuda", "mps"}
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.bfloat16

    for _ in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                if penalty_records:
                    loss = loss + compute_penalty(model, penalty_records, lam)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            scheduler.step()


def record_importance(model, loader, device, loss_fn, mode, max_fisher_samples):
    if mode == "snip":
        importance = compute_snip_scores(model, loader, loss_fn, device, mode="labeled")
        importance = {k: v.to(device) for k, v in importance.items()}
    elif mode == "ewc":
        importance = compute_fisher_information(model, loader, device, loss_fn, max_samples=max_fisher_samples)
    else:
        raise ValueError(f"Unsupported importance mode: {mode}")

    return {
        "importance": importance,
        "params": snapshot_params(model)
    }


def run_variant(variant, model_factory, task_train_loaders, task_test_loaders, device, args):
    model = model_factory().to(device)
    loss_fn = nn.CrossEntropyLoss()
    records = []
    lam = 0.0
    if variant == "snip":
        lam = args.snip_lambda
        importance_mode = "snip"
    elif variant == "ewc":
        lam = args.ewc_lambda
        importance_mode = "ewc"
    else:
        importance_mode = None

    best_acc_per_task = np.zeros(len(task_test_loaders))

    for task_id, train_loader in enumerate(task_train_loaders):
        train_task(
            model,
            train_loader,
            device,
            epochs=args.epochs,
            lr=args.lr,
            penalty_records=records if importance_mode else None,
            lam=lam
        )

        task_acc = evaluate_model(model, task_test_loaders[task_id], device)
        best_acc_per_task[task_id] = task_acc

        if importance_mode:
            record = record_importance(
                model,
                train_loader,
                device,
                loss_fn,
                mode=importance_mode,
                max_fisher_samples=args.fisher_samples
            )
            records.append(record)

    final_accs = []
    forgetting = []
    for task_id, test_loader in enumerate(task_test_loaders):
        acc = evaluate_model(model, test_loader, device)
        final_accs.append(acc)
        forgetting.append(max(0.0, best_acc_per_task[task_id] - acc))

    return {
        "final_acc": final_accs,
        "avg_acc": float(np.mean(final_accs)),
        "avg_forgetting": float(np.mean(forgetting)),
        "per_task_forgetting": forgetting
    }


def main():
    parser = argparse.ArgumentParser(description="SNIP vs EWC continual learning on CIFAR-100")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--snip_lambda", type=float, default=50.0)
    parser.add_argument("--ewc_lambda", type=float, default=50.0)
    parser.add_argument("--fisher_samples", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="expt1_results.json")
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
        make_dataloader(task, args.batch_size, shuffle=True, num_workers=args.workers, device=device)
        for task in train_tasks
    ]
    test_loaders = [
        make_dataloader(task, args.batch_size, shuffle=False, num_workers=args.workers, device=device)
        for task in test_tasks
    ]

    model_factory = get_model_factory(args.model, num_classes=100)

    aggregated = defaultdict(list)

    for run_id in range(args.runs):
        print(f"\n================ Run {run_id + 1}/{args.runs} ================")
        set_seed(args.seed + run_id)
        for variant in ["baseline", "snip", "ewc"]:
            print(f"\n--- Variant: {variant} ---")
            metrics = run_variant(
                variant,
                model_factory,
                train_loaders,
                test_loaders,
                device,
                args
            )
            aggregated[variant].append(metrics)
            print(f"{variant} avg acc: {metrics['avg_acc']:.2f}%, avg forgetting: {metrics['avg_forgetting']:.2f}%")

    summary = {}
    for variant, runs in aggregated.items():
        summary[variant] = {
            "avg_acc": float(np.mean([r["avg_acc"] for r in runs])),
            "avg_forgetting": float(np.mean([r["avg_forgetting"] for r in runs])),
            "runs": runs,
        }

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
