"""
Evaluation / inference script for TransUNetPS.

Usage:
    # Evaluate ONE checkpoint on ALL test objects (recommended)
    python test.py --mode eval_all --checkpoint checkpoints/train/best.pt --data_root ./data/testing
    python test.py --mode eval_all --checkpoint checkpoints/train/best.pt --data_root ./data/training

    # Evaluate on a single test object
    python test.py --mode eval --checkpoint checkpoints/train/best.pt --test_object ballPNG

    # Evaluate and save predicted normal maps
    python test.py --mode eval --checkpoint checkpoints/train/best.pt --test_object ballPNG --save_output ./output

    # Run on all objects (using LOGO checkpoints)
    python test.py --mode logo_eval --checkpoint_dir checkpoints/

    # Test on ALL training objects (10 objects)
    python test.py --mode eval_all --checkpoint checkpoints/train/best.pt --data_root ./data/training

    # Test on ALL testing objects (5 objects)
    python test.py --mode eval_all --checkpoint checkpoints/train/best.pt --data_root ./data/testing

    # Test on BOTH training + testing, save output images
    python test.py --mode eval_all --checkpoint checkpoints/train/best.pt --data_root ./data/training --save_output ./output/training
    python test.py --mode eval_all --checkpoint checkpoints/train/best.pt --data_root ./data/testing --save_output ./output/testing

    # Test on Synapse segmentation (following TransUNet test.py)
    # Evaluate on all 12 test volumes — prints per-organ Dice scores
    python test.py --mode synapse --checkpoint checkpoints/synapse/best.pt

    # Save predictions as .npy
    # python test.py --mode synapse --checkpoint checkpoints/synapse/best.pt --save_output ./output/synapse
"""
import argparse
import os
import glob
import csv
import numpy as np
import torch
import torch.nn.functional as F  # <--- BỔ SUNG DÒNG NÀY VÀO ĐÂY
from PIL import Image

from config import Config, DataConfig, ModelConfig
from dataset import DiLiGentTestDataset
from model import get_model
from utils import (
    mean_angular_error, angular_error_map, load_checkpoint, detect_model_type,
    count_parameters, normal_to_rgb,
)


@torch.no_grad()
def predict_full_resolution(model, images, device, tile_size=128, max_images=96):
    """
    Predict normal map using Sliding Window with Hanning Blending.

    Same approach as predict.py: generous padding + 50% overlap + Hanning window.
    All tiles are tile_size x tile_size (matching training patch size).

    Args:
        model: TransUNetPS or LightweightUNetPS model
        images: (N, 1, H, W) tensor
        device: torch device
        tile_size: must match training patch size (default 128)
        max_images: max images to use

    Returns:
        pred_normal: (3, H, W) tensor on CPU
    """
    import numpy as _np

    model.eval()
    N, C, H, W = images.shape
    tile_size = max(16, (tile_size // 16) * 16)
    overlap = 0.5
    stride = int(tile_size * (1 - overlap))

    # Subsample N
    if N > max_images:
        indices = torch.linspace(0, N - 1, max_images).long()
        images = images[indices]
        n_used = max_images
    else:
        n_used = N

    # Generous padding: tile_size//2 on all sides
    pad_top = tile_size // 2
    pad_bottom = (tile_size - H % stride) % stride + tile_size // 2
    pad_left = tile_size // 2
    pad_right = (tile_size - W % stride) % stride + tile_size // 2
    images = torch.nn.functional.pad(images, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")

    _, _, pH, pW = images.shape

    # Hanning window
    window_1d = _np.hanning(tile_size).astype(_np.float32)
    window_2d = _np.outer(window_1d, window_1d)
    window_t = torch.from_numpy(window_2d)  # (tile_size, tile_size)

    pred_accum = torch.zeros(3, pH, pW)
    weight_accum = torch.zeros(1, pH, pW)
    counts = torch.tensor([n_used], dtype=torch.long, device=device)

    y_positions = list(range(0, pH - tile_size + 1, stride))
    x_positions = list(range(0, pW - tile_size + 1, stride))

    for y in y_positions:
        for x in x_positions:
            tile_imgs = images[:, :, y:y + tile_size, x:x + tile_size]
            tile_imgs = tile_imgs.unsqueeze(0).to(device)

            pred_tile = model(tile_imgs, counts)[0].cpu()  # (3, tile_size, tile_size)

            pred_accum[:, y:y + tile_size, x:x + tile_size] += pred_tile * window_t.unsqueeze(0)
            weight_accum[:, y:y + tile_size, x:x + tile_size] += window_t.unsqueeze(0)

    # Weighted average
    weight_accum = weight_accum.clamp(min=1e-5)
    pred_normal = pred_accum / weight_accum

    # Crop padding
    pred_normal = pred_normal[:, pad_top:pad_top + H, pad_left:pad_left + W]

    # Re-normalize
    norm = pred_normal.norm(dim=0, keepdim=True).clamp(min=1e-8)
    return pred_normal / norm


@torch.no_grad()
def evaluate_object(model, test_ds, obj_idx, device, save_dir=None):
    """Evaluate on one object, optionally save outputs."""
    model.eval()
    images, normal_gt, mask, obj_name = test_ds[obj_idx]
    C, H, W = images.shape

    print(f"\nEvaluating: {obj_name} ({C} channels, {H}x{W})")

    images = images.unsqueeze(0).to(device) # Ép lên shape (1, 3, H, W)
    pred_normal = model(images)[0]
    pred_normal = F.normalize(pred_normal, p=2, dim=0)

    # Compute metrics
    pred_np = pred_normal.cpu().permute(1, 2, 0).numpy()
    gt_np = normal_gt.permute(1, 2, 0).numpy()
    mask_np = mask[0].numpy()

    mae = mean_angular_error(pred_np, gt_np, mask_np)
    err_map = angular_error_map(pred_np, gt_np, mask_np)

    # Statistics
    valid_errors = err_map[mask_np > 0.5]
    print(f"  Mean Angular Error: {mae:.2f}")
    print(f"  Median Error:       {np.median(valid_errors):.2f}")
    print(f"  Max Error:          {np.max(valid_errors):.2f}")
    print(f"  <10 pixels:         {(valid_errors < 10).mean() * 100:.1f}%")
    print(f"  <20 pixels:         {(valid_errors < 20).mean() * 100:.1f}%")
    print(f"  <30 pixels:         {(valid_errors < 30).mean() * 100:.1f}%")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        pred_rgb = normal_to_rgb(pred_np * mask_np[..., None])
        Image.fromarray(pred_rgb).save(os.path.join(save_dir, f"{obj_name}_pred_normal.png"))
        gt_rgb = normal_to_rgb(gt_np * mask_np[..., None])
        Image.fromarray(gt_rgb).save(os.path.join(save_dir, f"{obj_name}_gt_normal.png"))
        err_vis = (err_map / 90.0 * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(err_vis).save(os.path.join(save_dir, f"{obj_name}_error_map.png"))
        np.save(os.path.join(save_dir, f"{obj_name}_pred_normal.npy"), pred_np)
        print(f"  Saved outputs to {save_dir}/")

    # [ĐÃ SỬA]: Trả về dictionary để xuất CSV
    return {
        "mae": mae, "median": float(np.median(valid_errors)), "max": float(np.max(valid_errors)),
        "p10": float((valid_errors < 10).mean() * 100), "p20": float((valid_errors < 20).mean() * 100),
        "p30": float((valid_errors < 30).mean() * 100)
    }
    

def discover_objects(data_root):
    """Find all object directories that have Normal_gt.npy (prepared data)."""
    objects = []
    if not os.path.isdir(data_root):
        return objects
    for name in sorted(os.listdir(data_root)):
        obj_dir = os.path.join(data_root, name)
        if not os.path.isdir(obj_dir):
            continue
        if os.path.exists(os.path.join(obj_dir, "Normal_gt.npy")):
            objects.append(name)
    return objects


# Synapse evaluation (following TransUNet test.py)

SYNAPSE_ORGAN_NAMES = {
    1: "Aorta", 2: "Gallbladder", 3: "Kidney(L)", 4: "Kidney(R)",
    5: "Liver", 6: "Pancreas", 7: "Spleen", 8: "Stomach",
}


@torch.no_grad()
def test_synapse(checkpoint_path, synapse_root, device_str="auto", save_output=None):
    """
    Evaluate on Synapse test volumes (following TransUNet test.py).
    Processes each 3D volume slice-by-slice, computes per-class Dice.
    """
    import h5py

    config = Config(device=device_str)
    device = config.resolve_device()

    # Load model
    mt = detect_model_type(checkpoint_path)
    print(f"Auto-detected model_type: {mt}")
    model_cfg = ModelConfig(model_type=mt, mode="segmentation", num_classes=9, in_channels=1)
    model = get_model(model_cfg, model_type=mt).to(device)
    epoch, val_loss = load_checkpoint(checkpoint_path, model)
    model.eval()
    print(f"Loaded checkpoint: epoch={epoch}, loss={val_loss:.4f}")
    print(f"Parameters: {count_parameters(model):,}")

    # Load test volume list
    test_list_path = os.path.join(synapse_root, "test_vol.txt")
    with open(test_list_path, "r") as f:
        test_cases = [line.strip() for line in f if line.strip()]

    test_h5_dir = os.path.join(synapse_root, "test_vol_h5")
    img_size = 224

    all_dice = {c: [] for c in SYNAPSE_ORGAN_NAMES}

    for case_name in test_cases:
        h5_path = os.path.join(test_h5_dir, f"{case_name}.npy.h5")
        if not os.path.exists(h5_path):
            print(f"  WARNING: {h5_path} not found, skipping")
            continue

        data = h5py.File(h5_path, "r")
        image_vol = data["image"][:]   # (D, H, W) float32
        label_vol = data["label"][:]   # (D, H, W) uint8
        data.close()

        D, H, W = image_vol.shape
        pred_vol = np.zeros_like(label_vol, dtype=np.uint8)

        # Process slice-by-slice
        for d in range(D):
            img_slice = image_vol[d]  # (H, W)

            # Resize to model input size
            from PIL import Image as PILImage
            img_resized = np.array(PILImage.fromarray(img_slice).resize(
                (img_size, img_size), PILImage.BICUBIC))

            # To tensor
            inp = torch.from_numpy(img_resized).float().unsqueeze(0).unsqueeze(0).to(device)
            out = model(inp)  # (1, num_classes, 224, 224)
            pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()  # (224, 224)

            # Resize prediction back to original
            pred_resized = np.array(PILImage.fromarray(pred.astype(np.uint8)).resize(
                (W, H), PILImage.NEAREST))
            pred_vol[d] = pred_resized

        # Compute per-class Dice for this volume
        print(f"\n  {case_name} ({D} slices, {H}x{W}):")
        for cls_id, cls_name in SYNAPSE_ORGAN_NAMES.items():
            pred_mask = (pred_vol == cls_id).astype(np.float32)
            gt_mask = (label_vol == cls_id).astype(np.float32)
            intersect = (pred_mask * gt_mask).sum()
            total = pred_mask.sum() + gt_mask.sum()
            if total == 0:
                dice = 1.0 if intersect == 0 else 0.0
            else:
                dice = (2.0 * intersect) / total
            all_dice[cls_id].append(dice)
            print(f"    {cls_name:>15s}: {dice * 100:.2f}%")

        # Save prediction if requested
        if save_output:
            out_dir = os.path.join(save_output, case_name)
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, "prediction.npy"), pred_vol)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Synapse Evaluation Summary — {len(test_cases)} volumes")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")
    avg_all = []
    for cls_id, cls_name in SYNAPSE_ORGAN_NAMES.items():
        scores = all_dice[cls_id]
        if scores:
            mean_dice = np.mean(scores) * 100
            avg_all.append(mean_dice)
            print(f"  {cls_name:>15s}: {mean_dice:.2f}%")
    if avg_all:
        print(f"  {'':>15s}  --------")
        print(f"  {'Average Dice':>15s}: {np.mean(avg_all):.2f}%")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate TransUNetPS")
    parser.add_argument("--mode", type=str, default="eval_all",
                        choices=["eval", "eval_all", "logo_eval", "synapse"],
                        help="eval: single object, eval_all: all objects, synapse: Synapse segmentation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Checkpoint directory (for --mode logo_eval)")
    parser.add_argument("--test_object", type=str, default=None,
                        help="Object to evaluate on (for --mode eval)")
    parser.add_argument("--data_root", type=str, default="./data/training",
                        help="Directory containing test objects")
    parser.add_argument("--synapse_root", type=str, default="./data/Synapse",
                        help="Synapse dataset root (for --mode synapse)")
    parser.add_argument("--save_output", type=str, default=None,
                        help="Directory to save prediction outputs")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--csv_out", type=str, default=None, help="Path to save the CSV metrics file")
    args = parser.parse_args()

    # synapse: segmentation evaluation
    if args.mode == "synapse":
        if not args.checkpoint:
            print("ERROR: --checkpoint required")
            return
        test_synapse(args.checkpoint, os.path.abspath(args.synapse_root),
                     device_str=args.device, save_output=args.save_output)
        return

    config = Config(device=args.device)
    device = config.resolve_device()

    # eval: single object ======
    if args.mode == "eval":
        if not args.checkpoint:
            print("ERROR: --checkpoint required")
            return
        if not args.test_object:
            print("ERROR: --test_object required for --mode eval")
            return

        mt = detect_model_type(args.checkpoint)
        print(f"Auto-detected model_type: {mt}")
        model = get_model(config.model, model_type=mt).to(device)
        epoch, val_loss = load_checkpoint(args.checkpoint, model)
        print(f"Loaded checkpoint from epoch {epoch} (val_loss={val_loss:.4f})")
        print(f"Parameters: {count_parameters(model):,}")

        test_ds = DiLiGentTestDataset(
            data_root=os.path.abspath(args.data_root),
            objects=[args.test_object],
        )

        if len(test_ds) == 0:
            print(f"ERROR: Could not load test object '{args.test_object}'")
            return

        evaluate_object(model, test_ds, 0, device, save_dir=args.save_output)

    # eval_all: one checkpoint, all objects in data_root ==
    elif args.mode == "eval_all":
        if not args.checkpoint:
            print("ERROR: --checkpoint required")
            return

        data_root = os.path.abspath(args.data_root)
        objects = discover_objects(data_root)
        if not objects:
            print(f"ERROR: No prepared objects found in {data_root}")
            print("  (Each object needs Normal_gt.npy — run prepare_data.py first)")
            return

        mt = detect_model_type(args.checkpoint)
        print(f"Auto-detected model_type: {mt}")
        model = get_model(config.model, model_type=mt).to(device)
        epoch, val_loss = load_checkpoint(args.checkpoint, model)
        print(f"Loaded checkpoint from epoch {epoch} (val_loss={val_loss:.4f})")
        print(f"Parameters: {count_parameters(model):,}")
        print(f"Data root: {data_root}")
        print(f"Objects found: {len(objects)} — {objects}")

        all_results = {}
        for obj_name in objects:
            test_ds = DiLiGentTestDataset(
                data_root=data_root,
                objects=[obj_name],
            )
            if len(test_ds) == 0:
                print(f"\n  WARNING: Could not load {obj_name}, skipping")
                continue

            save_dir = os.path.join(args.save_output, obj_name) if args.save_output else None
            mae = evaluate_object(model, test_ds, 0, device, save_dir=save_dir)
            all_results[obj_name] = mae

        # Print summary table
        # Print summary table
        if all_results:
            print(f"\n{'='*60}")
            print(f"  Evaluation Summary — {len(all_results)} objects")
            print(f"  Checkpoint: {args.checkpoint}")
            print(f"  Data root:  {data_root}")
            print(f"{'='*60}")
            
            # 1. In từng vật thể (trích xuất 'mae' từ cuốn từ điển metrics)
            for obj, metrics in all_results.items():
                print(f"  {obj:>20s}: {metrics['mae']:.2f}°")
            
            # 2. Tính toán tổng hợp
            avg = np.mean([m['mae'] for m in all_results.values()])
            median = np.median([m['median'] for m in all_results.values()])
            
            # Tìm Best và Worst dựa trên giá trị 'mae'
            best_obj = min(all_results, key=lambda k: all_results[k]['mae'])
            worst_obj = max(all_results, key=lambda k: all_results[k]['mae'])
            
            print(f"  {'':>20s}  --------")
            print(f"  {'Average MAE':>20s}: {avg:.2f}°")
            print(f"  {'Median Error':>20s}: {median:.2f}°")
            print(f"  {'Best':>20s}: {all_results[best_obj]['mae']:.2f}° ({best_obj})")
            print(f"  {'Worst':>20s}: {all_results[worst_obj]['mae']:.2f}° ({worst_obj})")
            print(f"{'='*60}")

    # logo_eval: one checkpoint per fold
    elif args.mode == "logo_eval":
        data_root = os.path.abspath(args.data_root)
        objects = config.data.objects
        all_results = {}

        all_results = {}
        for obj_name in objects:
            test_ds = DiLiGentTestDataset(data_root=data_root, objects=[obj_name])
            if len(test_ds) == 0:
                print(f"\n  WARNING: Could not load {obj_name}, skipping")
                continue

            save_dir = os.path.join(args.save_output, obj_name) if args.save_output else None
            metrics = evaluate_object(model, test_ds, 0, device, save_dir=save_dir)
            all_results[obj_name] = metrics

        # XUẤT RA FILE CSV
        csv_filename = args.csv_out if args.csv_out else f"metrics_{mt}_test.csv"
        print(f"\nĐang xuất dữ liệu ra file: {csv_filename}")
        
        if len(all_results) > 0:
            with open(csv_filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Object", "Mean_Angular_Error", "Median_Angular_Error", "Max_Error", "Pixels_under_10", "Pixels_under_20", "Pixels_under_30"])
                
                for obj, m in all_results.items():
                    writer.writerow([obj, f"{m['mae']:.4f}", f"{m['median']:.4f}", f"{m['max']:.4f}", f"{m['p10']:.4f}", f"{m['p20']:.4f}", f"{m['p30']:.4f}"])
                
                avg_mae = np.mean([m['mae'] for m in all_results.values()])
                avg_median = np.mean([m['median'] for m in all_results.values()])
                writer.writerow(["Average", f"{avg_mae:.4f}", f"{avg_median:.4f}", "", "", "", ""])
            print(f"✅ Đã xuất file CSV thành công!")

        if all_results:
            print(f"\n{'='*50}")
            print("  LOGO Evaluation Summary")
            print(f"{'='*50}")
            for obj, mae in all_results.items():
                print(f"  {obj:>20s}: {mae:.2f}")
            avg = np.mean(list(all_results.values()))
            print(f"  {'Average':>20s}: {avg:.2f}")
            print(f"{'='*50}")


if __name__ == "__main__":
    main()
