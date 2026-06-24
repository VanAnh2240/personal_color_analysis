"""
src/dataset.py
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold

from config import (IMG_SIZE, MEAN, STD, LAPA_NUM_CLASSES,
                    RAW_DIR, K_FOLDS, BATCH_SIZE, NUM_WORKERS)


def get_train_transforms(size=IMG_SIZE):
    return A.Compose([
        A.Resize(size[0], size[1]),
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=0.05, scale=(0.9, 1.1),
        rotate=(-15, 15), mode=cv2.BORDER_REFLECT, p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3,
                      saturation=0.2, hue=0.05, p=0.5),
        A.GaussNoise(std_range=(0.02, 0.1), p=0.2),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_val_transforms(size=IMG_SIZE):
    return A.Compose([
        A.Resize(size[0], size[1]),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def parse_landmarks(txt_path: str) -> np.ndarray | None:
    if not os.path.exists(txt_path):
        return None
    try:
        pts = []
        with open(txt_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    pts.append((float(parts[0]), float(parts[1])))
        if len(pts) == 106:
            return np.array(pts, dtype=np.float32)
    except Exception:
        pass
    return None

class LapaSegDataset(Dataset):
    def __init__(self, img_paths: list, lbl_paths: list, transform=None):
        assert len(img_paths) == len(lbl_paths), (
            f"img/lbl count mismatch: {len(img_paths)} vs {len(lbl_paths)}"
        )
        self.img_paths = img_paths
        self.lbl_paths = lbl_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        if img is None:
            raise FileNotFoundError(f"Cannot read: {self.img_paths[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

        lbl = cv2.imread(self.lbl_paths[idx], cv2.IMREAD_GRAYSCALE) 
        if lbl is None:
            raise FileNotFoundError(f"Cannot read: {self.lbl_paths[idx]}")

        if self.transform:
            aug = self.transform(image=img, mask=lbl)
            img = aug["image"] 
            lbl = aug["mask"]

        lbl = lbl.long().clamp(0, LAPA_NUM_CLASSES - 1)
        return img, lbl

class LapaInferenceDataset(Dataset):
    def __init__(self, img_paths: list, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)["image"] 

        return img, img_path

def _collect_labeled(split: str):
    img_dir = os.path.join(RAW_DIR, split, "images")
    lbl_dir = os.path.join(RAW_DIR, split, "labels")

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Images dir not found: {img_dir}")
    if not os.path.isdir(lbl_dir):
        raise FileNotFoundError(
            f"Labels dir not found for split='{split}': {lbl_dir}\n"
            f"Note: LaPa 'val' has no labels — use _collect_images() instead."
        )

    imgs, lbls = [], []
    missing = 0
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        base     = os.path.splitext(fname)[0]
        lbl_path = os.path.join(lbl_dir, base + ".png")
        if not os.path.exists(lbl_path):
            missing += 1
            continue
        imgs.append(os.path.join(img_dir, fname))
        lbls.append(lbl_path)

    if missing:
        print(f"  [dataset] {split}: {missing} images skipped (no matching label)")
    print(f"  [dataset] {split}: {len(imgs)} labeled pairs loaded")
    return imgs, lbls


def _collect_images(split: str):
    img_dir = os.path.join(RAW_DIR, split, "images")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Images dir not found: {img_dir}")

    imgs = sorted(
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    print(f"  [dataset] {split}: {len(imgs)} images (no labels)")
    return imgs


def get_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    # train — labeled
    tr_imgs, tr_lbls = _collect_labeled("train")
    train_ds = LapaSegDataset(tr_imgs, tr_lbls, get_train_transforms())
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True,
                          drop_last=True)

    # val — images only 
    va_imgs = _collect_images("val")
    val_ds  = LapaInferenceDataset(va_imgs, get_val_transforms())
    val_dl  = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)

    # test — labeled
    te_imgs, te_lbls = _collect_labeled("test")
    test_ds = LapaSegDataset(te_imgs, te_lbls, get_val_transforms())
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)

    return train_dl, val_dl, test_dl

def get_kfold_dataloaders(k: int = K_FOLDS,
                          batch_size: int = BATCH_SIZE,
                          num_workers: int = NUM_WORKERS):
    
    all_imgs, all_lbls = _collect_labeled("train")
    all_imgs = np.array(all_imgs)
    all_lbls = np.array(all_lbls)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(all_imgs)):
        tr_ds = LapaSegDataset(all_imgs[tr_idx].tolist(),
                               all_lbls[tr_idx].tolist(),
                               get_train_transforms())
        va_ds = LapaSegDataset(all_imgs[va_idx].tolist(),
                               all_lbls[va_idx].tolist(),
                               get_val_transforms())
        tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=True,
                           drop_last=True)
        va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
        print(f"  [kfold] Fold {fold+1}: "
              f"train={len(tr_ds)}  val={len(va_ds)}")
        yield fold, tr_dl, va_dl


def load_single_image(img_path: str, size=IMG_SIZE):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tf  = get_val_transforms(size)
    return tf(image=img)["image"].unsqueeze(0)


def dataset_summary():
    print("\nLaPa Dataset Summary")
    print("=" * 48)
    for split in ("train", "val", "test"):
        img_dir = os.path.join(RAW_DIR, split, "images")
        lbl_dir = os.path.join(RAW_DIR, split, "labels")
        lmk_dir = os.path.join(RAW_DIR, split, "landmarks")

        n_imgs = len(os.listdir(img_dir)) if os.path.isdir(img_dir) else 0
        n_lbls = len(os.listdir(lbl_dir)) if os.path.isdir(lbl_dir) else 0
        n_lmks = len(os.listdir(lmk_dir)) if os.path.isdir(lmk_dir) else 0

        has_lbl = "✓" if n_lbls > 0 else "✗ (none)"
        print(f"  {split:6s}  images={n_imgs:6d}  "
              f"labels={has_lbl:10s}  "
              f"landmarks={n_lmks:6d}")
    print("=" * 48)
    print("  train → supervised training  (image + label)")
    print("  val   → visual inspection only (image only, NO label)")
    print("  test  → final mIoU evaluation (image + label)")
    print()
