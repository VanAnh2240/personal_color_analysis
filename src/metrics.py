"""
src/metrics.py
"""

import numpy as np
import torch
from config import LAPA_NUM_CLASSES, LAPA_CLASS_NAMES


class SegMetrics:
    def __init__(self, num_classes: int = LAPA_NUM_CLASSES,
                 ignore_index: int = 255):
        self.num_classes   = num_classes
        self.ignore_index  = ignore_index
        self.reset()

    def reset(self):
        self.conf_mat = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        p = preds.cpu().numpy().astype(np.int64).flatten()
        t = targets.cpu().numpy().astype(np.int64).flatten()

        mask = t != self.ignore_index
        p, t = p[mask], t[mask]

        valid = (p >= 0) & (p < self.num_classes) & \
                (t >= 0) & (t < self.num_classes)
        p, t = p[valid], t[valid]

        np.add.at(self.conf_mat, (t, p), 1)

    def compute(self) -> dict:
        cm = self.conf_mat.astype(np.float64)
        tp  = np.diag(cm)
        fp  = cm.sum(0) - tp
        fn  = cm.sum(1) - tp

        iou = np.where((tp + fp + fn) > 0,
                       tp / (tp + fp + fn + 1e-10),
                       np.nan)
        miou = float(np.nanmean(iou))

        per_class_acc = np.where(cm.sum(1) > 0,
                                 tp / (cm.sum(1) + 1e-10),
                                 np.nan)
        mean_acc = float(np.nanmean(per_class_acc))
        pixel_acc = float(tp.sum() / (cm.sum() + 1e-10))

        per_class = {LAPA_CLASS_NAMES[i]: float(iou[i])
                     for i in range(self.num_classes)}

        return {
            "mIoU":       miou,
            "pixel_acc":  pixel_acc,
            "mean_acc":   mean_acc,
            "per_class_iou": per_class,
        }

    def print_results(self, prefix=""):
        res = self.compute()
        print(f"{prefix}mIoU={res['mIoU']:.4f}  "
              f"pixel_acc={res['pixel_acc']:.4f}  "
              f"mean_acc={res['mean_acc']:.4f}")
        for cls, val in res["per_class_iou"].items():
            bar = "█" * int(val * 30)
            print(f"  {cls:>15s}: {val:.4f}  {bar}")


class DiceLoss(torch.nn.Module):
    def __init__(self, num_classes=LAPA_NUM_CLASSES, smooth=1.0):
        super().__init__()
        self.C      = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1) 
        t_oh  = torch.zeros_like(probs)
        t_oh.scatter_(1, targets.unsqueeze(1), 1)

        dims  = (0, 2, 3)
        num   = 2 * (probs * t_oh).sum(dims) + self.smooth
        denom = probs.sum(dims) + t_oh.sum(dims) + self.smooth
        return 1 - (num / denom).mean()


class ComboLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, ignore_index=255, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.ce    = torch.nn.CrossEntropyLoss(
            weight=class_weights,          
            ignore_index=ignore_index
        )
        self.dice  = DiceLoss()
 
    def forward(self, logits, targets):
        return self.alpha * self.ce(logits, targets) + \
               (1 - self.alpha) * self.dice(logits, targets)
