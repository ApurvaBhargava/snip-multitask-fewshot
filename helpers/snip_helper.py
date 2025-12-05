import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_snip_scores(model, dataloader, loss_fn, device, mode="labeled", batch=None):
    """
    Compute SNIP saliency scores once so they can be thresholded multiple times.
    """
    model.to(device)
    model.train()
    model.zero_grad()

    if batch is None:
        inputs, targets = next(iter(dataloader))
    else:
        inputs, targets = batch
    inputs = inputs.to(device)

    if mode == "labeled":
        targets = targets.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
    elif mode == "unlabeled":
        bs = inputs.size(0)
        rotations = [
            inputs,
            torch.rot90(inputs, 1, [2, 3]),
            torch.rot90(inputs, 2, [2, 3]),
            torch.rot90(inputs, 3, [2, 3]),
        ]
        aug = torch.cat(rotations, dim=0)
        ssl_targets = torch.cat([
            torch.zeros(bs, dtype=torch.long),
            torch.ones(bs, dtype=torch.long),
            2 * torch.ones(bs, dtype=torch.long),
            3 * torch.ones(bs, dtype=torch.long),
        ]).to(device)
        outputs = model(aug)
        loss = F.cross_entropy(outputs, ssl_targets)
    elif mode == "crossdomain":
        targets = torch.zeros(inputs.size(0), dtype=torch.long).to(device)
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
    else:
        raise ValueError(f"Unsupported SNIP mode: {mode}")

    loss.backward()

    scores = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            g = param.grad
            s = torch.abs(param * g).detach().cpu()
            scores[name] = s

    return scores

def build_masks_from_scores(scores, sparsity, device):
    """
    Threshold cached SNIP scores to reach the requested sparsity.
    """
    flat_scores = torch.cat([s.view(-1) for s in scores.values()])
    num_params = flat_scores.numel()
    keep_ratio = 1.0 - sparsity
    k = max(int(num_params * keep_ratio), 1)

    topk_vals, _ = torch.topk(flat_scores, k)
    threshold = topk_vals[-1]

    masks = {}
    for name, score in scores.items():
        mask = (score >= threshold).float().to(device)
        masks[name] = mask
    return masks

def compute_snip_masks(model, dataloader, loss_fn, sparsity, device, mode="labeled"):
    """
    Backwards-compatible wrapper that computes and thresholds scores in one go.
    """
    scores = compute_snip_scores(model, dataloader, loss_fn, device, mode)
    return build_masks_from_scores(scores, sparsity, device)

def apply_masks(model, masks):
    with torch.no_grad():
        for (name, param) in model.named_parameters():
            if name in masks:
                param.data *= masks[name]

def mask_gradients(model, masks):
    for (name, param) in model.named_parameters():
        if name in masks and param.grad is not None:
            param.grad *= masks[name]