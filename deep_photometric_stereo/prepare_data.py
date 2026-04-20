"""
Prepare training data: convert various formats to standardized .npy files.

Supports three data formats:
  - DiLiGenT:   Normal_gt.mat, mask.png, *PNG/ subfolder or flat numbered PNGs
  - Synthetic:  gt_normal.npy, mask.png, subdirs with .npy images
  - PRPS:       gt_normal.tif, inboundary.png, images_specular/ or images_metallic/ with .tif

Usage:
    python prepare_data.py                          # default ./data/training
    python prepare_data.py --data_root ./data/training --force
"""
import argparse
import os
import glob
import numpy as np
from PIL import Image

try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


def detect_format(obj_dir: str) -> str:
    """Detect which data format this object uses."""
    # PRPS format: gt_normal.tif + images_specular/ or images_metallic/
    if os.path.exists(os.path.join(obj_dir, "gt_normal.tif")):
        return "prps"
    # DiLiGenT format: Normal_gt.mat or *PNG subfolder or flat numbered PNGs
    if os.path.exists(os.path.join(obj_dir, "Normal_gt.mat")):
        return "diligent"
    if glob.glob(os.path.join(obj_dir, "*PNG")):
        return "diligent"
    if glob.glob(os.path.join(obj_dir, "[0-9]*.png")):
        return "diligent"
    # Synthetic format: gt_normal.npy
    if os.path.exists(os.path.join(obj_dir, "gt_normal.npy")):
        return "synthetic"
    for entry in os.listdir(obj_dir):
        sub = os.path.join(obj_dir, entry)
        if os.path.isdir(sub) and glob.glob(os.path.join(sub, "*.npy")):
            return "synthetic"
    # PRPS without gt_normal.tif but with images_specular/ or images_metallic/
    if (os.path.isdir(os.path.join(obj_dir, "images_specular")) or
            os.path.isdir(os.path.join(obj_dir, "images_metallic"))):
        return "prps"
    return "unknown"


def _read_tif(path: str) -> np.ndarray:
    """Read a .tif file, using tifffile if available, else PIL."""
    if HAS_TIFFFILE:
        return tifffile.imread(path)
    else:
        return np.array(Image.open(path))


def convert_normal_gt(obj_dir: str, fmt: str, force: bool = False) -> None:
    """Convert ground truth normals to standardized Normal_gt.npy (H, W, 3) float32."""
    npy_path = os.path.join(obj_dir, "Normal_gt.npy")

    if os.path.exists(npy_path) and not force:
        arr = np.load(npy_path)
        if arr.ndim == 3 and arr.shape[2] == 3 and arr.dtype in (np.float32, np.float64):
            print(f"  Normal_gt.npy OK: shape={arr.shape}, dtype={arr.dtype}")
            return
        else:
            print(f"  Normal_gt.npy INVALID (shape={arr.shape}, dtype={arr.dtype}), re-converting...")

    if fmt == "diligent":
        mat_path = os.path.join(obj_dir, "Normal_gt.mat")
        if not os.path.exists(mat_path):
            print(f"  WARNING: No Normal_gt.mat found")
            return
        if not HAS_SCIPY:
            raise ImportError("scipy required: pip install scipy")
        mat = sio.loadmat(mat_path)
        for key in ["Normal_gt", "normal_gt", "Normal", "normal"]:
            if key in mat:
                normals = mat[key].astype(np.float32)
                np.save(npy_path, normals)
                print(f"  Converted Normal_gt.mat[{key}] -> Normal_gt.npy: shape={normals.shape}")
                return
        data_keys = [k for k in mat.keys() if not k.startswith("__")]
        print(f"  WARNING: No normal key in .mat. Keys: {data_keys}")

    elif fmt == "synthetic":
        gt_path = os.path.join(obj_dir, "gt_normal.npy")
        if os.path.exists(gt_path):
            normals = np.load(gt_path).astype(np.float32)
            if normals.ndim == 3 and normals.shape[2] == 3:
                np.save(npy_path, normals)
                print(f"  Copied gt_normal.npy -> Normal_gt.npy: shape={normals.shape}")
            else:
                print(f"  WARNING: gt_normal.npy unexpected shape {normals.shape}")
        else:
            print(f"  WARNING: No gt_normal.npy found")

    elif fmt == "prps":
        tif_path = os.path.join(obj_dir, "gt_normal.tif")
        if os.path.exists(tif_path):
            normals = _read_tif(tif_path).astype(np.float32)
            # TIF normals may be (H, W, 3) or (H, W, 4) with alpha
            if normals.ndim == 3 and normals.shape[2] >= 3:
                normals = normals[:, :, :3]  # take first 3 channels
                # PRPS normals may be in [0, 1] or [0, 255] range — normalize to [-1, 1]
                if normals.max() > 1.5:
                    # Encoded as uint8/uint16: map [0, 255] -> [-1, 1]
                    normals = normals / 255.0 * 2.0 - 1.0
                elif normals.min() >= 0.0 and normals.max() <= 1.0:
                    # Encoded as [0, 1]: map to [-1, 1]
                    normals = normals * 2.0 - 1.0
                # else already in [-1, 1] range
                # Re-normalize to unit length
                norms = np.linalg.norm(normals, axis=-1, keepdims=True)
                norms = np.maximum(norms, 1e-8)
                normals = normals / norms
                np.save(npy_path, normals)
                print(f"  Converted gt_normal.tif -> Normal_gt.npy: shape={normals.shape}, "
                      f"range=[{normals.min():.2f}, {normals.max():.2f}]")
            else:
                print(f"  WARNING: gt_normal.tif unexpected shape {normals.shape}")
        else:
            print(f"  WARNING: No gt_normal.tif found")


def convert_mask(obj_dir: str, fmt: str, force: bool = False) -> None:
    """Convert mask to mask.npy (H, W) float32 binary."""
    npy_path = os.path.join(obj_dir, "mask.npy")

    if os.path.exists(npy_path) and not force:
        arr = np.load(npy_path)
        if arr.dtype == np.float32 and arr.ndim == 2:
            print(f"  mask.npy OK: shape={arr.shape}, foreground={int(arr.sum())}")
            return
        else:
            print(f"  mask.npy INVALID (shape={arr.shape}, dtype={arr.dtype}), re-converting...")

    # Try multiple mask file names
    mask_candidates = ["mask.png", "inboundary.png"]
    for mask_name in mask_candidates:
        mask_path = os.path.join(obj_dir, mask_name)
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert("L"))
            mask = (mask > 128).astype(np.float32)
            np.save(npy_path, mask)
            print(f"  Converted {mask_name} -> mask.npy: shape={mask.shape}, foreground={int(mask.sum())}")
            return

    print(f"  WARNING: No mask file found (tried: {mask_candidates})")


def verify_images(obj_dir: str, fmt: str) -> None:
    """Check that images exist and report count."""
    non_image = {"Normal_gt.png", "mask.png", "inboundary.png", "onboundary.png", "gt_normal.tif"}

    # Check for .tif images in subdirectories (PRPS format)
    if fmt == "prps":
        for subname in ["images_specular", "images_metallic"]:
            sub = os.path.join(obj_dir, subname)
            if os.path.isdir(sub):
                tifs = sorted(glob.glob(os.path.join(sub, "*.tif")))
                if tifs:
                    img = _read_tif(tifs[0])
                    print(f"  Images: {len(tifs)} TIFs in {subname}/, "
                          f"sample shape={img.shape}, dtype={img.dtype}")
        return

    # Check flat layout (numbered PNGs in obj_dir)
    direct_pngs = sorted([
        p for p in glob.glob(os.path.join(obj_dir, "*.png"))
        if os.path.basename(p) not in non_image and os.path.basename(p)[0].isdigit()
    ])
    if direct_pngs:
        img = np.array(Image.open(direct_pngs[0]))
        print(f"  Images: {len(direct_pngs)} PNGs (flat), "
              f"sample shape={img.shape}, dtype={img.dtype}")
        return

    # Check *PNG subfolder
    if fmt == "diligent":
        png_dirs = glob.glob(os.path.join(obj_dir, "*PNG"))
        if png_dirs:
            pngs = sorted(glob.glob(os.path.join(png_dirs[0], "*.png")))
            if pngs:
                img = np.array(Image.open(pngs[0]))
                print(f"  Images: {len(pngs)} PNGs in {os.path.basename(png_dirs[0])}/, "
                      f"sample shape={img.shape}, dtype={img.dtype}")
                return

    # Check .npy subdirectory
    if fmt == "synthetic":
        for name in sorted(os.listdir(obj_dir)):
            sub = os.path.join(obj_dir, name)
            if os.path.isdir(sub):
                npys = sorted(glob.glob(os.path.join(sub, "*.npy")))
                if npys:
                    img = np.load(npys[0])
                    print(f"  Images: {len(npys)} NPYs in {name}/, "
                          f"sample shape={img.shape}, dtype={img.dtype}")
                    return

    print(f"  WARNING: No images found")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for TransUNetPS")
    parser.add_argument("--data_root", type=str, default="./data/training")
    parser.add_argument("--force", action="store_true",
                        help="Re-convert even if .npy already exists")
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    print(f"Data root: {data_root}\n")

    objects = []
    for name in sorted(os.listdir(data_root)):
        obj_dir = os.path.join(data_root, name)
        if not os.path.isdir(obj_dir):
            continue
        fmt = detect_format(obj_dir)
        if fmt != "unknown":
            objects.append((name, fmt))

    if not objects:
        print("No object directories found. Check --data_root path.")
        return

    print(f"Found {len(objects)} objects: {[(n, f) for n, f in objects]}\n")

    for obj_name, fmt in objects:
        obj_dir = os.path.join(data_root, obj_name)
        print(f"[{obj_name}] (format: {fmt})")
        convert_normal_gt(obj_dir, fmt, force=args.force)
        convert_mask(obj_dir, fmt, force=args.force)
        verify_images(obj_dir, fmt)
        print()

    print("Data preparation complete.")


if __name__ == "__main__":
    main()
