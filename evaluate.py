"""
evaluate.py
"""

import argparse, os, sys, csv
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (CKPT_DEEPLAB, CKPT_CLIPUNET, RESULT_DIR,
                    ACTIVE_MODEL, LAPA_CLASS_NAMES, BATCH_SIZE, NUM_WORKERS)
from src.dataset import (
    _collect_labeled, _collect_images,
    LapaSegDataset, LapaInferenceDataset,
    get_val_transforms,
    dataset_summary,
)
from src.metrics import SegMetrics
from torch.utils.data import DataLoader

def build_model(model_name: str, device: torch.device):
    if model_name == "deeplab":
        from src.models.system_1_deeplabv3 import DeepLabV3
        model = DeepLabV3(pretrained=False)
        ckpt  = CKPT_DEEPLAB
    else:
        from src.models.system_2_clipunet import ClipUNet
        model = ClipUNet(freeze_clip=False)
        ckpt  = CKPT_CLIPUNET

    if not os.path.exists(ckpt):
        print(f"WARNING: checkpoint not found at {ckpt}.")
        print("Running with random weights (for debugging only).")
    else:
        state = torch.load(ckpt, map_location=device)
        # Support both plain state-dict and wrapped checkpoints
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {ckpt}")

    return model.to(device).eval()


@torch.no_grad()
def evaluate_segmentation(model_name: str, device: torch.device):
    model = build_model(model_name, device)

    te_imgs, te_lbls = _collect_labeled("test")
    test_ds = LapaSegDataset(te_imgs, te_lbls, get_val_transforms())
    test_dl = DataLoader(test_ds, BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

    metrics = SegMetrics()
    for imgs, masks in test_dl:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs).argmax(dim=1)
        metrics.update(preds, masks)

    results = metrics.compute()

    print(f"\n{'═'*55}")
    print(f"  {model_name.upper()} — Test split segmentation metrics")
    print(f"{'═'*55}")
    metrics.print_results()

    # Save to CSV
    out_path = os.path.join(RESULT_DIR, f"{model_name}_results.csv")
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["mIoU",      f"{results['mIoU']:.4f}"])
        w.writerow(["pixel_acc", f"{results['pixel_acc']:.4f}"])
        w.writerow(["mean_acc",  f"{results['mean_acc']:.4f}"])
        for cls, val in results["per_class_iou"].items():
            w.writerow([f"iou_{cls}", f"{val:.4f}"])

    print(f"\nSaved → {out_path}")
    return results

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  default=ACTIVE_MODEL,
                   choices=["deeplab", "clipunet"])
    p.add_argument("--device",  default="auto")
    args = p.parse_args()

    dataset_summary()

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device : {device}")
    print(f"Model  : {args.model}")

    evaluate_segmentation(args.model, device)

if __name__ == "__main__":
    main()
