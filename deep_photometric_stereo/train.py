"""
Training script for UNetPS — Uncalibrated Photometric Stereo.

Usage:
    # 1. Joint Pipeline (Train on Synthetic + DiLiGenT, evaluate every 5 epochs)
    python train.py --mode joint_888 --json_path data/synthetic/train.json \
                    --val_json data/synthetic/val_split.json \
                    --data_root data/training --epochs 200

    # 2. Train with leave-one-object-out cross-validation (DiLiGenT)
    python train.py --mode logo_cv --epochs 200

    # 3. Train on specific objects, test on another (DiLiGenT)
    python train.py --mode train --train_objects buddha bunny --test_object cat --epochs 200

    # 4. Train Synapse segmentation (following TransUNet paper exactly)
    python train.py --mode synapse --model_type transunet --epochs 150
"""
import argparse
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm 
import csv

from config import Config, DataConfig, ModelConfig, TrainConfig
from dataset import (
    DiLiGentDataset, DiLiGentTestDataset,
    SyntheticDataset, SyntheticTestDataset
)
from model import get_model
from losses import CombinedNormalLoss, SegmentationLoss
from utils import (
    mean_angular_error, save_checkpoint, load_checkpoint,
    count_parameters, normal_to_rgb,
)

def poly_lr(optimizer, base_lr, iter_num, max_iters, power=0.9):
    """Polynomial LR decay (TransUNet trainer.py)."""
    lr = base_lr * (1.0 - iter_num / max_iters) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# =====================================================================
# SHARED PHOTOMETRIC STEREO FUNCTIONS
# =====================================================================

def train_one_epoch_ps(model, loader, criterion, optimizer, device, grad_clip=1.0, 
                       base_lr=0.01, iter_offset=0, max_iters=10000, 
                       lr_power=0.9, use_poly_lr=True):
    model.train()
    total_loss = 0.0
    n_batches = 0
    iter_num = iter_offset

    pbar = tqdm(loader, desc="Training", leave=False)

    for batch_idx, (images, normals, masks) in enumerate(pbar):
        images = images.to(device)    
        normals = normals.to(device)  
        masks = masks.to(device)      

        optimizer.zero_grad()
        pred = model(images) 
        loss = criterion(pred, normals, masks)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        iter_num += 1
        if use_poly_lr:
            poly_lr(optimizer, base_lr, iter_num, max_iters, lr_power)

        total_loss += loss.item()
        n_batches += 1

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return total_loss / max(n_batches, 1), iter_num


@torch.no_grad()
def evaluate_and_save_top3(model, test_dataset, epoch, save_dir, device):
    """Evaluate on validation set and save the top 3 and bottom 3 predictions."""
    model.eval()
    results = []
    
    print("\n  [Evaluate] Running evaluation on validation set...")
    for idx in range(len(test_dataset)):
        images, normal_gt, mask, obj_name = test_dataset[idx]
        if isinstance(obj_name, str) and obj_name == "error": continue
            
        images = images.unsqueeze(0).to(device) # (1, 3, H, W)
        
        pred_normal = model(images)[0]
        pred_normal = F.normalize(pred_normal, p=2, dim=0)

        pred_np = pred_normal.cpu().permute(1, 2, 0).numpy()
        gt_np = normal_gt.permute(1, 2, 0).numpy()
        mask_np = mask[0].numpy()

        mae = mean_angular_error(pred_np, gt_np, mask_np)
        results.append({
            'name': obj_name, 'mae': mae,
            'pred': pred_np, 'gt': gt_np, 'input': images[0].cpu()
        })

    results.sort(key=lambda x: x['mae'])
    best_3 = results[:3]
    worst_3 = results[-3:]
    avg_mae = np.mean([r['mae'] for r in results])
    print(f"  => Average MAE Validation: {avg_mae:.2f}°")

    eval_dir = os.path.join(save_dir, "eval_images")
    os.makedirs(eval_dir, exist_ok=True)
    
    def save_figure(items, prefix):
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        for i, item in enumerate(items):
            # Column 1: Input
            rgb_input = item['input'].permute(1, 2, 0).numpy()
            rgb_input = np.clip((rgb_input - rgb_input.min()) / (rgb_input.max() - rgb_input.min() + 1e-8), 0, 1)
            axes[i, 0].imshow(rgb_input)
            axes[i, 0].set_title(f"Input ({item['name']})")
            axes[i, 0].axis('off')
            
            # Column 2: Ground Truth
            gt_vis = np.clip((item['gt'] + 1.0) / 2.0, 0, 1)
            axes[i, 1].imshow(gt_vis)
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis('off')
            
            # Column 3: Prediction
            pred_vis = np.clip((item['pred'] + 1.0) / 2.0, 0, 1)
            axes[i, 2].imshow(pred_vis)
            axes[i, 2].set_title(f"Prediction (MAE: {item['mae']:.2f}°)")
            axes[i, 2].axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(eval_dir, f"epoch_{epoch}_{prefix}.png"))
        plt.close()

    if best_3: save_figure(best_3, "best_3")
    if worst_3: save_figure(worst_3, "worst_3")

    return avg_mae

@torch.no_grad()
def evaluate_diligent_holdout(model, test_dataset, device):
    """Hold-out evaluation on specific DiLiGenT objects."""
    model.eval()
    print("  [DiLiGenT Hold-out Test]:")
    for idx in range(len(test_dataset)):
        images, normal_gt, mask, name = test_dataset[idx]
        images = images.unsqueeze(0).to(device)
        pred = F.normalize(model(images)[0], p=2, dim=0)
        mae = mean_angular_error(pred.cpu().permute(1,2,0).numpy(), normal_gt.permute(1,2,0).numpy(), mask[0].numpy())
        print(f"    - {name}: {mae:.2f}°")


# =====================================================================
# PIPELINE 1: JOINT PIPELINE (SYNTHETIC + DILIGENT)
# =====================================================================

def train_joint_888_pipeline(config, args):
    """Main training pipeline for joint dataset (Synthetic + DiLiGenT)."""
    device = config.resolve_device()
    print(f"\n{'='*60}")
    print(f"  TRAINING PIPELINE (Synthetic + DiLiGenT)")
    print(f"  Model: {args.model_type} | Patch: {args.patch_size} | Batch: {args.batch_size}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # 1. Initialize and merge datasets
    synth_train = SyntheticDataset(args.json_path, patch_size=args.patch_size, augment=True)
    
    all_diligent_objs = config.data.objects
    diligent_train_objs = [o for o in all_diligent_objs if o not in ["pot2PNG", "bearPNG"]]
    diligent_train = DiLiGentDataset(
        data_root=config.data.data_root, 
        objects=diligent_train_objs, 
        patch_size=args.patch_size, 
        patches_per_epoch=args.patches_per_epoch // 2, 
        augment=True
    )
    
    combined_dataset = ConcatDataset([synth_train, diligent_train])
    train_loader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, drop_last=True)
    
    # Validation sets
    synth_test = SyntheticTestDataset(args.val_json)
    diligent_test = DiLiGentTestDataset(config.data.data_root, objects=["bearPNG", "pot2PNG"])

    # 2. Initialize Model & Optimizer
    model = get_model(config.model, model_type=args.model_type).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = CombinedNormalLoss(angular_w=0.5, cosine_w=0.3, l1_w=0.2)

    save_dir = os.path.join(args.save_dir, "joint_888_run")
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "training_log.csv")
    
    best_mae = float("inf")
    history_loss = []
    history_mae = []
    mae_epochs = []
    
    # Load previous CSV log to resume plotting history
    if os.path.exists(csv_path):
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Use dictionary to overwrite duplicated epochs when resuming
            epoch_to_loss = {}
            epoch_to_mae = {}
            
            for row in reader:
                ep = int(row['Epoch'])
                epoch_to_loss[ep] = float(row['Train_Loss'])
                
                val_mae = row.get('Val_MAE', '')
                if val_mae is not None and val_mae.strip() != '':
                    epoch_to_mae[ep] = float(val_mae.strip())
                    
        # Reconstruct history arrays sorted by epoch
        for ep in sorted(epoch_to_loss.keys()):
            history_loss.append(epoch_to_loss[ep])
            if ep in epoch_to_mae:
                history_mae.append(epoch_to_mae[ep])
                mae_epochs.append(ep)
                
        print(f"[*] LOADED: {len(history_loss)} epochs from history.")
            
    else:
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train_Loss", "Val_MAE"])

    # 3. Training Loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, _ = train_one_epoch_ps(
            model, train_loader, criterion, optimizer, device,
            grad_clip=1.0, base_lr=args.lr, use_poly_lr=False
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{args.epochs} — loss={train_loss:.4f}, lr={lr:.2e}, time={elapsed:.1f}s")
        history_loss.append(train_loss)
        
        val_mae_str = ""

        # Evaluate every 5 epochs
        if epoch % 5 == 0 or epoch == args.epochs:
            val_mae = evaluate_and_save_top3(model, synth_test, epoch, save_dir, device)
            evaluate_diligent_holdout(model, diligent_test, device)
            
            history_mae.append(val_mae)
            mae_epochs.append(epoch)
            val_mae_str = f"{val_mae:.4f}"

            if val_mae < best_mae:
                best_mae = val_mae
                ckpt_path = os.path.join(save_dir, "best_joint.pt")
                save_checkpoint(model, optimizer, epoch, val_mae, ckpt_path, model_type=args.model_type)
                print(f"  ★ New best: {val_mae:.2f}° (saved)")

            # Plot training progress
            fig, (ax_mae, ax_loss) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            # --- Top subplot: Validation MAE ---
            ax_mae.plot(mae_epochs, history_mae, 'r--o', label='Val MAE')
            ax_mae.set_ylabel('MAE (degrees)', color='r')
            ax_mae.tick_params('y', colors='r')
            ax_mae.set_title('Training Progress')
            ax_mae.grid(True, linestyle='--', alpha=0.5) 
            ax_mae.legend(loc='upper right')

            # --- Bottom subplot: Train Loss ---
            ax_loss.plot(range(1, len(history_loss) + 1), history_loss, 'b-', label='Train Loss')
            ax_loss.set_xlabel('Total Epochs')
            ax_loss.set_ylabel('Loss', color='b')
            ax_loss.tick_params('y', colors='b')
            ax_loss.grid(True, linestyle='--', alpha=0.5)
            ax_loss.legend(loc='upper right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "training_curve.png"), dpi=150)
            plt.close(fig)
            
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", val_mae_str])

    print(f"\nTraining done. Best MAE: {best_mae:.2f}°")


# =====================================================================
# PIPELINE 2: DILIGENT LEAVE-ONE-OBJECT-OUT (LOGO_CV)
# =====================================================================

@torch.no_grad()
def evaluate_logo(model, test_dataset, device):
    model.eval()
    results = {}
    for idx in range(len(test_dataset)):
        images, normal_gt, mask, obj_name = test_dataset[idx]
        images = images.unsqueeze(0).to(device)
        
        pred_normal = model(images)[0]
        pred_normal = F.normalize(pred_normal, p=2, dim=0)

        pred_np = pred_normal.cpu().permute(1, 2, 0).numpy()
        gt_np = normal_gt.permute(1, 2, 0).numpy()
        mask_np = mask[0].numpy()

        mae = mean_angular_error(pred_np, gt_np, mask_np)
        results[obj_name] = mae
        print(f"  {obj_name}: MAE = {mae:.2f}°")
    return results

def train_fold(train_objects, test_object, config, fold_name="fold"):
    device = config.resolve_device()
    print(f"\n{'='*60}\n  {fold_name}: train={train_objects}, test={test_object}\n{'='*60}\n")

    train_ds = DiLiGentDataset(
        data_root=config.data.data_root, objects=train_objects,
        patch_size=config.data.patch_size, patches_per_epoch=config.data.patches_per_epoch, augment=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=config.data.batch_size, shuffle=True,
        num_workers=config.data.num_workers, drop_last=True,
    )
    test_ds = DiLiGentTestDataset(data_root=config.data.data_root, objects=[test_object])

    model = get_model(config.model, model_type=config.model.model_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.epochs, eta_min=1e-6)
    criterion = CombinedNormalLoss(angular_w=config.train.angular_weight, cosine_w=config.train.cosine_weight, l1_w=config.train.l1_weight)

    best_mae = float("inf")
    save_dir = os.path.join(config.train.save_dir, fold_name)
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, config.train.epochs + 1):
        t0 = time.time()
        train_loss, _ = train_one_epoch_ps(
            model, train_loader, criterion, optimizer, device,
            grad_clip=config.train.grad_clip, log_interval=config.train.log_interval,
            use_poly_lr=False,
        )
        scheduler.step()
        
        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{config.train.epochs} — loss={train_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}, time={elapsed:.1f}s")

        if epoch % 10 == 0 or epoch == config.train.epochs:
            results = evaluate_logo(model, test_ds, device)
            mae = list(results.values())[0] if results else float("inf")
            if mae < best_mae:
                best_mae = mae
                ckpt_path = os.path.join(save_dir, "best.pt")
                save_checkpoint(model, optimizer, epoch, mae, ckpt_path, model_type=config.model.model_type)
                print(f"  ★ New best: {mae:.2f}° (saved)")

    return best_mae

def logo_cv(config):
    objects = config.data.objects
    results = {}
    for test_obj in objects:
        train_objs = [o for o in objects if o != test_obj]
        mae = train_fold(train_objs, test_obj, config, fold_name=f"logo_{test_obj}")
        results[test_obj] = mae

    print(f"\n{'='*60}\n  LOGO Cross-Validation Results\n{'='*60}")
    for obj, mae in results.items(): print(f"  {obj:>10s}: {mae:.2f}°")
    print(f"  {'Average':>10s}: {np.mean(list(results.values())):.2f}°\n{'='*60}\n")
    return results


# =====================================================================
# PIPELINE 3: SYNAPSE SEGMENTATION
# =====================================================================

class SynapseDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, list_file, img_size=224):
        self.base_dir = base_dir
        self.img_size = img_size
        with open(list_file, "r") as f:
            self.sample_list = [line.strip() for line in f if line.strip()]
        print(f"SynapseDataset: {len(self.sample_list)} slices from {base_dir}")

    def __len__(self): return len(self.sample_list)

    def __getitem__(self, idx):
        slice_name = self.sample_list[idx]
        data = np.load(os.path.join(self.base_dir, slice_name + ".npz"))
        image = data["image"].astype(np.float32)
        label = data["label"].astype(np.float32)

        from PIL import Image as PILImage
        image = np.array(PILImage.fromarray(image).resize((self.img_size, self.img_size), PILImage.BICUBIC))
        label = np.array(PILImage.fromarray(label).resize((self.img_size, self.img_size), PILImage.NEAREST))

        if np.random.random() > 0.5:
            k = np.random.randint(0, 4)
            image = np.rot90(image, k).copy()
            label = np.rot90(label, k).copy()
        if np.random.random() > 0.5:
            from scipy.ndimage import rotate as scipy_rotate
            angle = np.random.uniform(-20, 20)
            image = scipy_rotate(image, angle, order=3, reshape=False)
            label = scipy_rotate(label, angle, order=0, reshape=False)

        image = torch.from_numpy(image).float().unsqueeze(0)
        label = torch.from_numpy(label).float()
        return image, label

def train_synapse(config, args):
    device = config.resolve_device()
    synapse_dir = os.path.abspath(args.synapse_root)
    num_classes = args.num_classes
    img_size = 224

    print(f"\n{'='*60}\n  Synapse Segmentation Training\n{'='*60}\n")

    train_list = os.path.join(synapse_dir, "train.txt")
    train_npz_dir = os.path.join(synapse_dir, "train_npz")
    train_ds = SynapseDataset(train_npz_dir, train_list, img_size=img_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    model_cfg = ModelConfig(model_type=args.model_type, mode="segmentation", num_classes=num_classes, in_channels=1)
    model = get_model(model_cfg, model_type=args.model_type).to(device)

    base_lr = args.lr if args.lr != 1e-3 else 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
    criterion = SegmentationLoss(num_classes=num_classes)

    max_epochs = args.epochs if args.epochs != 200 else 150
    max_iters = max_epochs * len(train_loader)
    iter_num = 0

    save_dir = os.path.join(args.save_dir, "synapse")
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, max_epochs + 1):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            iter_num += 1
            poly_lr(optimizer, base_lr, iter_num, max_iters, power=0.9)
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch}/{max_epochs} — loss={avg_loss:.4f}, time={time.time() - t0:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch, avg_loss, os.path.join(save_dir, "best.pt"), model_type=args.model_type)
            print(f"  New best loss: {avg_loss:.4f} (saved)")

    print(f"\nSynapse training done. Best loss: {best_loss:.4f}")


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Train TransUNetPS")
    parser.add_argument("--mode", type=str, default="joint_888",
                        choices=["train", "logo_cv", "synapse", "joint_888", "synthetic"],
                        help="Training mode: train, logo_cv, synapse, or joint_888")
    parser.add_argument("--data_root", type=str, default="./data/training")
    parser.add_argument("--synapse_root", type=str, default="./data/Synapse")
    parser.add_argument("--json_path", type=str, default="data/synthetic/train.json")
    parser.add_argument("--val_json", type=str, default="data/synthetic/val_split.json")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=9)
    parser.add_argument("--train_objects", nargs="+", default=None)
    parser.add_argument("--test_object", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--patches_per_epoch", type=int, default=2000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model_type", type=str, default="lightweight",
                        choices=["transunet", "lightweight"])
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "sgd"])
    args = parser.parse_args()

    if args.mode == "synapse":
        config = Config(device=args.device)
        train_synapse(config, args)
    else:
        config = Config(
            data=DataConfig(
                data_root=os.path.abspath(args.data_root),
                batch_size=args.batch_size,
                patch_size=args.patch_size,
                patches_per_epoch=args.patches_per_epoch,
            ),
            model=ModelConfig(model_type=args.model_type),
            train=TrainConfig(
                epochs=args.epochs, lr=args.lr, save_dir=args.save_dir, optimizer=args.optimizer,
            ),
            device=args.device,
        )
        config.data.num_workers = args.num_workers 

        if args.mode in ["joint_888", "synthetic"]:
            train_joint_888_pipeline(config, args)
        elif args.mode == "logo_cv":
            logo_cv(config)
        elif args.mode == "train":
            all_objs = config.data.objects
            train_objs = args.train_objects or all_objs[:-1]
            test_obj = args.test_object or all_objs[-1]
            train_fold(train_objs, test_obj, config, fold_name="train")

if __name__ == "__main__":
    main()