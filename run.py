import os
import torch
import torch.nn as nn
import json
import argparse
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils import set_seed
from helpers.snip_helper import *
from helpers.train_helper import *

# Models
from models.resnet18 import ResNet18_CIFAR
from models.resnet20 import ResNet20_CIFAR
from models.simpleconvnet import SimpleConvNet
from models.vgg11 import VGG11_CIFAR
from models.wideresnet import WideResNet_CIFAR


# --------------------------------------------------------------------
# Model Factory
# --------------------------------------------------------------------
def get_model(model_name, num_classes):
    """
    Returns a CIFAR-compatible model instance based on the --model argument.
    """
    model_name = model_name.lower()

    model_dict = {
        "resnet18": lambda: ResNet18_CIFAR(num_classes=num_classes),
        "resnet20": lambda: ResNet20_CIFAR(num_classes=num_classes),
        "simpleconv": lambda: SimpleConvNet(num_classes=num_classes),
        "vgg11": lambda: VGG11_CIFAR(num_classes=num_classes),
        "wideresnet": lambda: WideResNet_CIFAR(depth=16, widen_factor=2, num_classes=num_classes),
    }

    if model_name not in model_dict:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {', '.join(model_dict.keys())}"
        )

    return model_dict[model_name]()


def maybe_compile(model, enable_compile):
    if enable_compile and hasattr(torch, "compile"):
        try:
            return torch.compile(model)
        except Exception as exc:
            print(f"[WARN] torch.compile failed ({exc}). Falling back to eager mode.")
    return model


def make_dataloader(dataset, batch_size, shuffle, device, num_workers, drop_last=False):
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "drop_last": drop_last
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    if device.type in {"cuda", "mps"}:
        loader_kwargs["pin_memory"] = True
        loader_kwargs["pin_memory_device"] = device.type

    return DataLoader(dataset, **loader_kwargs)


def parse_list_arg(raw_value, cast_fn):
    if isinstance(raw_value, (list, tuple)):
        return [cast_fn(v) for v in raw_value]
    if isinstance(raw_value, str):
        parts = [p.strip() for p in raw_value.split(",") if p.strip()]
        return [cast_fn(p) for p in parts]
    raise ValueError(f"Unable to parse list argument from value: {raw_value}")


# --------------------------------------------------------------------
# Build Few-Shot Subset
# --------------------------------------------------------------------
def build_few_shot_subset(dataset, k_per_class, num_classes, seed):
    """
    Build few-shot subset using a FIXED seed for the entire RUN.
    NOT per k-shot, NOT per mode, NOT per sparsity.
    """

    # Seed is specific to the RUN (NOT changing inside run)
    set_seed(seed)

    targets = np.array(dataset.targets)
    indices = []

    for class_id in range(num_classes):
        class_indices = np.where(targets == class_id)[0]
        chosen = np.random.choice(class_indices, size=k_per_class, replace=False)
        indices.extend(chosen)

    subset = torch.utils.data.Subset(dataset, indices)
    return subset


# --------------------------------------------------------------------
# Execute one COMPLETE run (all k × modes × sparsity)
# --------------------------------------------------------------------
def run_single_experiment(args, run_id, train_dataset, test_loader,
                          snip_labeled_loader, snip_unlabeled_loader, snip_cross_loader, device):

    print(f"\n========== RUN {run_id+1}/{args.runs} ==========")

    # ---- FIXED seed for entire RUN ----
    experiment_seed = 1000 + run_id
    set_seed(experiment_seed)

    # Experiment hyperparameters
    EPOCHS = args.epochs
    LR = 0.1
    TRAIN_BATCH_SIZE = args.train_batch
    NUM_CLASSES = 100
    loss_fn = nn.CrossEntropyLoss()

    K_SHOTS_LIST = args.k_list
    SNIP_SPARSITY_LIST = args.snip_sparsities
    SNIP_MODES = args.snip_modes

    results = {}
    snip_cache = {}
    snip_loader_map = {
        "labeled": snip_labeled_loader,
        "unlabeled": snip_unlabeled_loader,
        "crossdomain": snip_cross_loader
    }
    snip_iter_map = {mode: iter(loader) for mode, loader in snip_loader_map.items()}

    def next_snip_batch(mode):
        iterator = snip_iter_map[mode]
        try:
            batch = next(iterator)
        except StopIteration:
            snip_iter_map[mode] = iter(snip_loader_map[mode])
            iterator = snip_iter_map[mode]
            batch = next(iterator)
        return batch

    # -------------------------------------------------------------
    # Main Experiment Loop
    # -------------------------------------------------------------
    for k in K_SHOTS_LIST:

        print("\n" + "=" * 100)
        print(f"FEW-SHOT k = {k}")
        print("=" * 100)

        # FIXED sampling for this run
        few_shot_train = build_few_shot_subset(
            dataset=train_dataset,
            k_per_class=k,
            num_classes=NUM_CLASSES,
            seed=experiment_seed
        )

        effective_batch = max(1, min(TRAIN_BATCH_SIZE, len(few_shot_train)))
        train_loader = make_dataloader(
            few_shot_train,
            batch_size=effective_batch,
            shuffle=True,
            device=device,
            num_workers=args.workers,
            drop_last=False
        )

        results[k] = {
            "dense": None,
            "snip": {mode: {} for mode in SNIP_MODES}
        }

        # ---------------------------------------------------------
        # 1. Dense baseline
        # ---------------------------------------------------------
        dense_model = get_model(args.model, NUM_CLASSES).to(device)
        dense_model = maybe_compile(dense_model, args.torch_compile)

        dense_acc = train_model(
            dense_model,
            train_loader,
            test_loader,
            epochs=EPOCHS,
            lr=LR,
            device=device,
            use_snip=False,
            verbose=False
        )

        results[k]["dense"] = dense_acc
        print(f"[DENSE] k={k} → {dense_acc:.2f}%")

        # ---------------------------------------------------------
        # 2. SNIP settings
        # ---------------------------------------------------------
        for mode in SNIP_MODES:

            print(f"\n--- SNIP MODE: {mode} ---")

            snip_loader = snip_loader_map[mode]
            cache_key = (k, mode)

            if cache_key not in snip_cache:
                base_model = get_model(args.model, NUM_CLASSES).to(device)
                base_state = OrderedDict(
                    (name, tensor.detach().cpu().clone())
                    for name, tensor in base_model.state_dict().items()
                )
                batch = next_snip_batch(mode)
                snip_scores = compute_snip_scores(
                    base_model,
                    snip_loader,
                    loss_fn,
                    device=device,
                    mode=mode,
                    batch=batch
                )
                snip_cache[cache_key] = {
                    "state": base_state,
                    "scores": snip_scores
                }
                del base_model
            cache_entry = snip_cache[cache_key]

            for sp in SNIP_SPARSITY_LIST:

                print(f"[SNIP] k={k}, mode={mode}, sparsity={sp}")

                snip_model = get_model(args.model, NUM_CLASSES).to(device)
                snip_model.load_state_dict(cache_entry["state"])
                snip_model = maybe_compile(snip_model, args.torch_compile)
                masks = build_masks_from_scores(
                    cache_entry["scores"],
                    sparsity=sp,
                    device=device
                )

                snip_acc = train_model(
                    snip_model,
                    train_loader,
                    test_loader,
                    epochs=EPOCHS,
                    lr=LR,
                    device=device,
                    use_snip=True,
                    snip_sparsity=sp,
                    snip_mode=mode,
                    snip_data_loader=snip_loader,
                    verbose=False,
                    precomputed_masks=masks
                )

                results[k]["snip"][mode][sp] = snip_acc
                print(f"   → Result: {snip_acc:.2f}%")

    return results


# --------------------------------------------------------------------
# Average results over runs
# --------------------------------------------------------------------
def average_results(list_of_runs):

    final = {}
    K_list = sorted(list_of_runs[0].keys())

    for k in K_list:

        final[k] = {
            "dense": {"mean": None, "std": None},
            "snip": {}
        }

        dense_vals = [run[k]["dense"] for run in list_of_runs]
        final[k]["dense"]["mean"] = float(np.mean(dense_vals))
        final[k]["dense"]["std"] = float(np.std(dense_vals))

        modes = list(list_of_runs[0][k]["snip"].keys())
        final[k]["snip"] = {mode: {} for mode in modes}

        for mode in modes:
            sparsities = list(list_of_runs[0][k]["snip"][mode].keys())

            for sp in sparsities:
                vals = [run[k]["snip"][mode][sp] for run in list_of_runs]
                final[k]["snip"][mode][sp] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals))
                }

    return final


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
def main():
    """
    CLI Arguments:
        --model : resnet18, resnet20, simpleconv, vgg11, wideresnet
        --runs  : number of full experiments to average
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batch", type=int, default=128)
    parser.add_argument("--eval_batch", type=int, default=256)
    parser.add_argument("--snip_batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--k_list", type=str, default="1,2,5,10,20")
    parser.add_argument("--snip_sparsities", type=str, default="0.5,0.7,0.9,0.95")
    parser.add_argument("--snip_modes", type=str, default="labeled,unlabeled,crossdomain")
    parser.add_argument("--torch_compile", action="store_true",
                        help="Enable torch.compile for each model instantiation.")
    args = parser.parse_args()

    args.k_list = parse_list_arg(args.k_list, int)
    args.snip_sparsities = parse_list_arg(args.snip_sparsities, float)
    args.snip_modes = parse_list_arg(args.snip_modes, str)
    args.workers = max(0, args.workers)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # -----------------------------
    # Load datasets once
    # -----------------------------
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

    train_dataset = datasets.CIFAR100(root="./data", train=True, download=True,
                                      transform=transform_train)
    test_dataset = datasets.CIFAR100(root="./data", train=False, download=True,
                                     transform=transform_test)

    test_loader = make_dataloader(
        test_dataset,
        batch_size=args.eval_batch,
        shuffle=False,
        device=device,
        num_workers=args.workers,
        drop_last=False
    )

    snip_labeled_loader = make_dataloader(
        train_dataset,
        batch_size=args.snip_batch,
        shuffle=True,
        device=device,
        num_workers=args.workers,
        drop_last=True
    )

    snip_unlabeled_loader = make_dataloader(
        datasets.CIFAR100(root="./data", train=True, download=False,
                          transform=transform_train),
        batch_size=args.snip_batch,
        shuffle=True,
        device=device,
        num_workers=args.workers,
        drop_last=True
    )

    snip_cross_loader = make_dataloader(
        datasets.CIFAR10(root="./data", train=False, download=True,
                         transform=transform_test),
        batch_size=args.snip_batch,
        shuffle=True,
        device=device,
        num_workers=args.workers,
        drop_last=True
    )

    # -----------------------------
    # Multi-run
    # -----------------------------
    all_runs = []

    for run_id in range(args.runs):

        # NEW SEED FOR EACH RUN
        set_seed(42 + run_id)

        run_result = run_single_experiment(
            args,
            run_id,
            train_dataset,
            test_loader,
            snip_labeled_loader,
            snip_unlabeled_loader,
            snip_cross_loader,
            device
        )

        # Save each run individually
        with open(f"results_run{run_id+1}.json", "w") as f:
            json.dump(run_result, f, indent=4)

        all_runs.append(run_result)

    # -----------------------------
    # Average across runs
    # -----------------------------
    final_results = average_results(all_runs)

    with open("results.json", "w") as f:
        json.dump(final_results, f, indent=4)

    print("\nSaved averaged results to results.json")


if __name__ == "__main__":
    main()