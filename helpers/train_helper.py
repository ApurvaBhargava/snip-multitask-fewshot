import torch
import torch.nn as nn
import torch.optim as optim
from helpers.snip_helper import *

def train_model(model, train_loader, test_loader, epochs, lr, device,
                use_snip=False, snip_sparsity=0.9, snip_mode="labeled",
                snip_data_loader=None, verbose=True, precomputed_masks=None):

    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    autocast_enabled = device.type in {"cuda", "mps"}
    amp_dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    non_blocking = device.type in {"cuda", "mps"}

    masks = None
    best_test_acc = 0.0

    # ----- SNIP Initialization -----
    if use_snip:
        if precomputed_masks is not None:
            masks = {k: v.to(device) for k, v in precomputed_masks.items()}
        else:
            if verbose:
                print(f"Computing SNIP masks with sparsity={snip_sparsity} ...")

            masks = compute_snip_masks(
                model,
                snip_data_loader if snip_data_loader is not None else train_loader,
                loss_fn,
                sparsity=snip_sparsity,
                device=device,
                mode=snip_mode
            )
        apply_masks(model, masks)

    # ----- Training Loop -----
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,
                                dtype=amp_dtype,
                                enabled=autocast_enabled):
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if masks is not None:
                mask_gradients(model, masks)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()

            if masks is not None:
                apply_masks(model, masks)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # compute metrics this epoch
        train_loss = running_loss / max(total, 1)
        train_acc = 100.0 * correct / max(total, 1)

        test_acc = evaluate_model(model, test_loader, device, autocast_enabled, amp_dtype)
        best_test_acc = max(best_test_acc, test_acc)

        if verbose:
            print(
                f"Epoch [{epoch}/{epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Train Acc: {train_acc:.2f}% "
                f"Test Acc: {test_acc:.2f}% "
                f"(Best: {best_test_acc:.2f}%)"
            )

    return best_test_acc


def evaluate_model(model, dataloader, device, autocast_enabled=None, amp_dtype=torch.float16):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            with torch.autocast(device_type=device.type,
                                dtype=amp_dtype,
                                enabled=bool(autocast_enabled)):
                outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total
