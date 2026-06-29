"""
train.py
python train.py --model clipunet --epochs 50
python train.py --model deeplab  --epochs 50
"""

import argparse, os, sys, time, random, json
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LinearLR, SequentialLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ACTIVE_MODEL, NUM_EPOCHS, LR, WEIGHT_DECAY,
    SCHEDULER, SEED,
    CKPT_DEEPLAB, CKPT_CLIPUNET, RESULT_DIR,
    BATCH_SIZE, NUM_WORKERS, LAPA_NUM_CLASSES,
)
from src.dataset import (
    _collect_labeled, LapaSegDataset,
    get_train_transforms, get_val_transforms,
)
from src.metrics import SegMetrics, ComboLoss
from src.utils   import save_checkpoint, load_checkpoint, save_best, log_epoch
from torch.utils.data import DataLoader


LAPA_CLASS_WEIGHTS = torch.tensor([
    0.5,   # 0  background
    0.7,   # 1  skin
    5.0,   # 2  left_eyebrow
    5.0,   # 3  right_eyebrow
    5.0,   # 4  left_eye
    5.0,   # 5  right_eye
    3.0,   # 6  nose
    6.0,   # 7  upper_lip
    6.0,   # 8  inner_mouth
    6.0,   # 9  lower_lip
    1.5,   # 10 hair
], dtype=torch.float32)


def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def sanity_check(loader, device, num_classes=LAPA_NUM_CLASSES):
    print("\n  [sanity] Checking first batch...")
    imgs, masks = next(iter(loader))
    imgs, masks = imgs.to(device), masks.to(device)

    print(f"  [sanity] Image  shape : {imgs.shape}  "
          f"min={imgs.min():.2f}  max={imgs.max():.2f}")
    print(f"  [sanity] Mask   shape : {masks.shape}  "
          f"dtype={masks.dtype}")

    unique_classes = masks.unique().tolist()
    print(f"  [sanity] Unique classes in batch : {[int(c) for c in unique_classes]}")

    if len(unique_classes) < 4:
        print(f"  [sanity] WARNING: Only {len(unique_classes)} classes in batch — "
              f"possible label loading issue!")
    else:
        print(f"  [sanity] OK — {len(unique_classes)} classes found.")

    class_counts = [(masks == i).sum().item() for i in range(num_classes)]
    total_px = masks.numel()
    print("  [sanity] Pixel distribution:")
    for i, cnt in enumerate(class_counts):
        if cnt > 0:
            bar = "█" * int(cnt / total_px * 40)
            print(f"    class {i:2d}: {cnt:8d} px  ({cnt/total_px*100:4.1f}%)  {bar}")
    print()


def build_model(name):
    if name == "deeplab":
        from src.models.system_1_deeplabv3 import DeepLabV3
        return DeepLabV3(pretrained=True), CKPT_DEEPLAB

    from src.models.system_2_clipunet import ClipUNet
    return ClipUNet(freeze_clip=False), CKPT_CLIPUNET


def build_optimizer(model, model_name: str, lr: float = LR):
    if model_name == "clipunet":
        encoder_params = [p for n, p in model.named_parameters()
                          if "encoder" in n and p.requires_grad]
        decoder_params = [p for n, p in model.named_parameters()
                          if "encoder" not in n and p.requires_grad]

        n_enc = sum(p.numel() for p in encoder_params) / 1e6
        n_dec = sum(p.numel() for p in decoder_params) / 1e6
        print(f"  [opt] Encoder trainable: {n_enc:.1f}M  lr={lr*0.1:.1e}")
        print(f"  [opt] Decoder trainable: {n_dec:.1f}M  lr={lr:.1e}")

        return optim.AdamW([
            {"params": encoder_params, "lr": lr * 0.1},
            {"params": decoder_params, "lr": lr},
        ], weight_decay=WEIGHT_DECAY)

    trainable = filter(lambda p: p.requires_grad, model.parameters())
    return optim.AdamW(trainable, lr=lr, weight_decay=WEIGHT_DECAY)


def build_scheduler(opt, epochs: int, warmup_epochs: int = 5):
    warmup = LinearLR(opt, start_factor=0.1, end_factor=1.0,
                      total_iters=warmup_epochs)
    if SCHEDULER == "cosine":
        main = CosineAnnealingLR(opt, T_max=max(1, epochs - warmup_epochs),
                                 eta_min=1e-6)
    else:
        main = StepLR(opt, step_size=max(1, (epochs - warmup_epochs) // 3),
                      gamma=0.1)
    return SequentialLR(opt, schedulers=[warmup, main],
                        milestones=[warmup_epochs])


class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = 0.0
        self.counter   = 0

    def step(self, miou: float) -> bool:
        if miou > self.best + self.min_delta:
            self.best    = miou
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def train_one(model, loader, opt, crit, device, scaler=None):
    model.train()
    total = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        opt.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                loss = crit(model(imgs), masks)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            loss = crit(model(imgs), masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def eval_labeled(model, loader, crit, device, metrics):
    model.eval(); metrics.reset(); total = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        total += crit(logits, masks).item()
        metrics.update(logits.argmax(1), masks)
    res = metrics.compute()
    return total / max(len(loader), 1), res["mIoU"], res


def run_standard(model_name, epochs, device, resume=None):
    from sklearn.model_selection import train_test_split

    model, ckpt_path = build_model(model_name)
    model = model.to(device)

    all_imgs, all_lbls = _collect_labeled("train")
    tr_imgs, mo_imgs, tr_lbls, mo_lbls = train_test_split(
        all_imgs, all_lbls, test_size=0.10, random_state=SEED, shuffle=True
    )
    print(f"\n  Train={len(tr_imgs)}  Monitor={len(mo_imgs)}")

    train_ds   = LapaSegDataset(tr_imgs, tr_lbls, get_train_transforms())
    monitor_ds = LapaSegDataset(mo_imgs, mo_lbls, get_val_transforms())
    train_dl   = DataLoader(train_ds,   BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            drop_last=True)
    monitor_dl = DataLoader(monitor_ds, BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    sanity_check(train_dl, device)

    class_w = LAPA_CLASS_WEIGHTS.to(device)
    crit    = ComboLoss(class_weights=class_w)

    opt    = build_optimizer(model, model_name, LR)
    sched  = build_scheduler(opt, epochs)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    metrics    = SegMetrics()
    early_stop = EarlyStopping(patience=15)
    start_ep   = 1
    best_miou  = 0.0

    if resume and os.path.exists(resume):
        state = load_checkpoint(resume, model, opt, None, device)
        sched = build_scheduler(opt, epochs)
        start_ep  = state.get("epoch", 0) + 1
        best_miou = state.get("best_miou", 0.0)
        print(f"  Resumed from epoch {start_ep - 1}, best mIoU={best_miou:.4f}")

    #log_path = os.path.join(RESULT_DIR, f"{model_name}_train_log.csv")
    #mode = "a" if (resume and os.path.exists(log_path)) else "w"

    # with open(log_path, mode, newline="") as f:
    #     w = csv.writer(f)
    #     if mode == "w":
    #         w.writerow(["epoch", "train_loss", "monitor_loss",
    #                     "monitor_mIoU", "lr"])

    for ep in range(start_ep, epochs + 1):
        t0      = time.time()
        tr_loss = train_one(model, train_dl, opt, crit, device, scaler)
        mo_loss, mo_miou, mo_res = eval_labeled(
            model, monitor_dl, crit, device, metrics)
        lr = opt.param_groups[0]["lr"]
        sched.step()

        log_epoch(model_name, ep, epochs,
                  tr_loss, mo_loss, mo_miou, lr,
                  time.time() - t0, best_miou)
        # w.writerow([ep, f"{tr_loss:.5f}", f"{mo_loss:.5f}",
        #             f"{mo_miou:.5f}", f"{lr:.2e}"])

        best_miou = save_best(
            model, ckpt_path, mo_miou, best_miou,
            extra={"epoch": ep, "best_miou": mo_miou,
                   "optimizer": opt.state_dict(),
                   "scheduler": sched.state_dict()})

        if ep % 10 == 0:
            print("  Per-class IoU:")
            for cls, val in mo_res["per_class_iou"].items():
                bar = "█" * int(val * 20)
                print(f"    {cls:>15s}: {val:.4f}  {bar}")

        if early_stop.step(mo_miou):
            print(f"\n  [early stop] No improvement for {early_stop.patience} "
                  f"epochs. Stopping at epoch {ep}.")
            break

    print(f"\nTraining complete.")
    print(f"  Best monitor mIoU : {best_miou:.4f}")
    #print(f"  Log               → {log_path}")
    print(f"  Best checkpoint   → {ckpt_path}")

    # try:
    #     from src.utils import plot_training_curves
    #     # plot_training_curves(
    #     #     log_path,
    #     #     os.path.join(RESULT_DIR, f"{model_name}_curves.png"))
    # except Exception:
    #     pass


def main():
    p = argparse.ArgumentParser(
        description="Train DeepLabV3 or ClipUNet on LaPa dataset")
    p.add_argument("--model",  default=ACTIVE_MODEL,
                   choices=["deeplab", "clipunet"])
    p.add_argument("--epochs", default=NUM_EPOCHS, type=int)
    p.add_argument("--resume", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--lr",     default=LR, type=float,
                   help=f"Base learning rate (default {LR}). "
                        f"ClipUNet encoder sẽ dùng lr/10 tự động.")
    args = p.parse_args()

    seed_everything()
    device = None
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"\n{'━'*54}")
    print(f"  Model      : {args.model}")
    print(f"  Device     : {device}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Base LR    : {LR:.1e}  "
          f"(encoder LR = {LR*0.1:.1e} for clipunet)")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  Class weights: ON (inverse-frequency)")
    print(f"  Early stop : patience=15")
    if args.resume:
        print(f"  Resume     : {args.resume}")
    print(f"{'━'*54}\n")

    run_standard(args.model, args.epochs, device, resume=args.resume)


if __name__ == "__main__":
    main()