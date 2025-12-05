import torch
import random
import numpy as np
from torch.utils.data import Subset

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_few_shot_subset(dataset, k_per_class, num_classes=100, seed=42):
    """
    Returns: Subset(dataset, indices)
    """
    set_seed(seed)
    targets = np.array(dataset.targets)
    indices = []

    for c in range(num_classes):
        class_indices = np.where(targets == c)[0]
        np.random.shuffle(class_indices)
        if k_per_class > 0:
            chosen = class_indices[:k_per_class]
        else:
            chosen = []
        indices.extend(chosen.tolist())

    print(f"Few-shot subset with k={k_per_class}: {len(indices)} samples")
    return Subset(dataset, indices)
