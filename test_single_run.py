import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import json
import random
import numpy as np
from collections import defaultdict
from utils import *
from helpers.snip_helper import *
from helpers.train_helper import *
from models.resnet18 import ResNet18_CIFAR
from models.resnet20 import ResNet20_CIFAR
from models.simpleconvnet import SimpleConvNet
from models.vgg11 import VGG11_CIFAR
from models.wideresnet import WideResNet_CIFAR

def main():
  set_seed(42)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Using device:", device)

  # Standard CIFAR-100 normalization
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

  num_classes = 100

  EPOCHS = 30
  BATCH_SIZE = 128
  LR = 0.1

  K_SHOTS_LIST = [1, 2, 5, 10, 20]
  SNIP_SPARSITY_LIST = [0.5, 0.7, 0.9, 0.95]
  SNIP_MODES = ["labeled", "unlabeled", "crossdomain"]

  results = {}

  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

  snip_unlabeled_loader = DataLoader(
      datasets.CIFAR100(root="./data", train=True, download=False,
                        transform=transform_train),
      batch_size=128,
      shuffle=True,
      num_workers=2
  )

  snip_cross_loader = DataLoader(
      datasets.CIFAR10(root="./data", train=False, download=True,
                      transform=transform_test),
      batch_size=128,
      shuffle=True,
      num_workers=2
  )

  for k in K_SHOTS_LIST:
      print("\n" + "="*100)
      print(f"FEW-SHOT k = {k}")
      print("="*100)

      few_shot_train = build_few_shot_subset(
          train_dataset, k_per_class=k, num_classes=num_classes, seed=42
      )

      train_loader = DataLoader(
          few_shot_train,
          batch_size=BATCH_SIZE,
          shuffle=True,
          num_workers=2
      )

      results[k] = {
          "dense": None,
          "snip": {mode:{} for mode in SNIP_MODES}
      }

      # ----- DENSE -----
      dense_model = ResNet18_CIFAR(num_classes=num_classes)
      acc_dense = train_model(
          dense_model,
          train_loader,
          test_loader,
          epochs=EPOCHS,
          lr=LR,
          device=device,
          use_snip=False,
          verbose=False
      )
      results[k]["dense"] = acc_dense
      print(f"\n[DENSE] k={k} → {acc_dense:.2f}%")

      # ----- SNIP LOOPS -----
      for mode in SNIP_MODES:
          print(f"\n----- SNIP MODE: {mode} -----")

          # pick dataloader
          if mode == "labeled":
              snip_loader = train_loader
          elif mode == "unlabeled":
              snip_loader = snip_unlabeled_loader
          elif mode == "crossdomain":
              snip_loader = snip_cross_loader

          for sp in SNIP_SPARSITY_LIST:
              print(f"\n[SNIP] k={k}, mode={mode}, sparsity={sp}")

              snip_model = ResNet18_CIFAR(num_classes=num_classes)
              acc_snip = train_model(
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
                  verbose=False
              )

              results[k]["snip"][mode][sp] = acc_snip
              print(f"[RESULT] SNIP k={k}, mode={mode}, sp={sp} → {acc_snip:.2f}%")

  # ----- SAVE JSON -----
  with open("results.json", "w") as f:
      json.dump(results, f, indent=4)
  print("\nSaved results to results.json")


if __name__ == "__main__":
    main()