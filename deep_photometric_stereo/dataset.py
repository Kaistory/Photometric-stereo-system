import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import glob
import random
import cv2
import numpy as np
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

class DiLiGentDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        objects: list,
        patch_size: int = 128,
        patches_per_epoch: int = 2000,
        augment: bool = True,
    ):
        self.data_root = data_root
        self.objects = objects
        self.patch_size = patch_size
        self.patches_per_epoch = patches_per_epoch
        self.augment = augment
        self.K_train = 3 

        # IMPORTANT: Store ONLY file paths, do not load data (images, numpy arrays) into RAM
        self.object_info = []
        for obj_name in objects:
            obj_dir = os.path.join(data_root, obj_name)
            info = self._scan_object(obj_dir, obj_name)
            if info is not None:
                self.object_info.append(info)

        if not self.object_info:
            raise RuntimeError(f"No valid objects found in {data_root} for {objects}")

        print(f"DiLiGentDataset: {len(self.object_info)} objects, N=3, patch={patch_size}, patches/epoch={patches_per_epoch}")

    def _scan_object(self, obj_dir: str, obj_name: str):
        """Scans the directory to retrieve file paths without loading image data."""
        normal_path = os.path.join(obj_dir, "Normal_gt.npy")
        if not os.path.exists(normal_path): return None

        mask_path = os.path.join(obj_dir, "mask.npy")
        if not os.path.exists(mask_path):
            mask_path = None
            for mask_name in ["mask.png", "inboundary.png"]:
                mp = os.path.join(obj_dir, mask_name)
                if os.path.exists(mp):
                    mask_path = mp
                    break

        image_paths = self._scan_images(obj_dir)
        if not image_paths: return None

        return {
            "name": obj_name,
            "dir": obj_dir,
            "normal_path": normal_path,
            "mask_path": mask_path,
            "image_paths": image_paths
        }

    @staticmethod
    def _scan_images(obj_dir):
        """Scans and returns a list of image paths instead of loading images."""
        non_image_names = {"Normal_gt.png", "mask.png", "inboundary.png", "onboundary.png", "gt_normal.tif"}
        def _is_numbered_image(path):
            name = os.path.basename(path)
            return name not in non_image_names and name[0].isdigit()

        direct_pngs = sorted(glob.glob(os.path.join(obj_dir, "*.png")))
        image_pngs = [p for p in direct_pngs if _is_numbered_image(p)]
        if image_pngs: return image_pngs

        png_dirs = glob.glob(os.path.join(obj_dir, "*PNG"))
        if png_dirs:
            pngs = sorted(glob.glob(os.path.join(png_dirs[0], "*.png")))
            if pngs: return pngs

        for subname in ["images_specular", "images_metallic"]:
            sub = os.path.join(obj_dir, subname)
            if os.path.isdir(sub):
                tifs = sorted(glob.glob(os.path.join(sub, "*.tif")))
                if tifs: return tifs

        for subname in sorted(os.listdir(obj_dir)):
            sub = os.path.join(obj_dir, subname)
            if not os.path.isdir(sub): continue
            npy_paths = sorted(glob.glob(os.path.join(sub, "*.npy")))
            if npy_paths: return npy_paths
            tif_paths = sorted(glob.glob(os.path.join(sub, "*.tif")))
            if tif_paths: return tif_paths
        return None

    @staticmethod
    def _load_single_image(path):
        """Loads a physical image from disk."""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npy":
            img = np.load(path).astype(np.float32)
            if img.ndim == 3: img = img.mean(axis=-1)
            if img.max() > 1.5: img = img / 255.0
        elif ext in [".tif", ".tiff"]:
            try:
                import tifffile
                img = tifffile.imread(path).astype(np.float32)
            except ImportError:
                img = np.array(Image.open(path)).astype(np.float32)
            if img.ndim == 3:
                if img.shape[2] >= 3:
                    img = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
                else: img = img[:, :, 0]
            if img.max() > 1.5:
                img = img / img.max() if img.max() > 0 else img
        else:
            img = np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0
        return img

    def __len__(self):
        return self.patches_per_epoch

    def __getitem__(self, idx):
        # 1. Randomly select 1 object
        obj_info = random.choice(self.object_info)
        
        # 2. LOAD NORMAL AND MASK DATA FROM DISK TO RAM
        normal_gt = np.load(obj_info["normal_path"]).astype(np.float32)
        
        if obj_info["mask_path"]:
            ext = os.path.splitext(obj_info["mask_path"])[1].lower()
            if ext == ".npy":
                mask = np.load(obj_info["mask_path"]).astype(np.float32)
            else:
                mask = np.array(Image.open(obj_info["mask_path"]).convert("L")).astype(np.float32)
                mask = (mask > 128).astype(np.float32)
        else:
            mask = np.ones(normal_gt.shape[:2], dtype=np.float32)

        if mask.ndim != 2: mask = mask.reshape(normal_gt.shape[:2])
        if mask.dtype != np.float32:
            mask = (mask > 0.5).astype(np.float32) if mask.max() <= 1.0 else (mask > 128).astype(np.float32)

        H, W = normal_gt.shape[:2]
        ps = self.patch_size

        # 3. MASK-AWARE CROP ALGORITHM
        valid_coords = np.argwhere(mask > 0.5)

        if len(valid_coords) > 0:
            rand_idx = random.randint(0, len(valid_coords) - 1)
            center_y, center_x = valid_coords[rand_idx]
            y = center_y - (ps // 2)
            x = center_x - (ps // 2)
            y = max(0, min(y, H - ps))
            x = max(0, min(x, W - ps))
        else:
            y = random.randint(0, max(0, H - ps))
            x = random.randint(0, max(0, W - ps))
            
        mask_patch = mask[y:y+ps, x:x+ps]
        normal_patch = normal_gt[y:y+ps, x:x+ps, :]

        # 4. SELECTIVELY LOAD IMAGES (MEMORY OPTIMIZATION)
        image_paths = obj_info["image_paths"]
        total_imgs = len(image_paths)
        indices = sorted(random.sample(range(total_imgs), min(self.K_train, total_imgs)))
        
        # Load exactly 3 images, crop immediately, and discard the rest
        img_patches = []
        for i in indices:
            full_img = self._load_single_image(image_paths[i])
            img_patches.append(full_img[y:y+ps, x:x+ps])
            
        img_patches = np.stack(img_patches, axis=0) # (3, ps, ps)
        
        # 5. NORMALIZATION AND AUGMENTATION
        mask_bool = mask_patch > 0.5
        for i in range(img_patches.shape[0]):
            if mask_bool.any():
                m = img_patches[i][mask_bool].mean()
                s = img_patches[i][mask_bool].std() + 1e-8
                img_patches[i] = (img_patches[i] - m) / s

        if self.augment and random.random() > 0.5:
            img_patches = img_patches[:, :, ::-1].copy()
            normal_patch = normal_patch[:, ::-1, :].copy()
            normal_patch[:, :, 0] = -normal_patch[:, :, 0]  
            mask_patch = mask_patch[:, ::-1].copy()

        img_tensor = torch.from_numpy(img_patches).float()
        normal_tensor = torch.from_numpy(normal_patch).float().permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_patch).float().unsqueeze(0)

        return img_tensor, normal_tensor, mask_tensor

class DiLiGentTestDataset(Dataset):
    def __init__(self, data_root: str, objects: list):
        self.data_root = data_root
        self.objects = objects
        self.object_info = []

        # Scan and store paths only, avoid loading data into RAM
        for obj_name in objects:
            obj_dir = os.path.join(data_root, obj_name)
            info = self._scan_object(obj_dir, obj_name)
            if info is not None:
                self.object_info.append(info)

    def _scan_object(self, obj_dir, obj_name):
        normal_path = os.path.join(obj_dir, "Normal_gt.npy")
        if not os.path.exists(normal_path): return None

        mask_path = os.path.join(obj_dir, "mask.npy")
        if not os.path.exists(mask_path):
            mask_path = None # If no mask.npy exists, a dummy mask will be created during __getitem__

        # Reuse image scanning function from DiLiGentDataset
        image_paths = DiLiGentDataset._scan_images(obj_dir)
        if not image_paths: return None

        return {
            "name": obj_name,
            "normal_path": normal_path,
            "mask_path": mask_path,
            "image_paths": image_paths
        }

    def __len__(self):
        return len(self.object_info)

    def __getitem__(self, idx):
        obj_info = self.object_info[idx]
        
        # 1. Load Normal
        normal_gt = np.load(obj_info["normal_path"]).astype(np.float32)
        
        # 2. Load Mask
        if obj_info["mask_path"]:
            ext = os.path.splitext(obj_info["mask_path"])[1].lower()
            if ext == ".npy":
                mask = np.load(obj_info["mask_path"]).astype(np.float32)
            else:
                mask = np.array(Image.open(obj_info["mask_path"]).convert("L")).astype(np.float32)
                mask = (mask > 128).astype(np.float32)
        else:
            mask = np.ones(normal_gt.shape[:2], dtype=np.float32)

        if mask.ndim != 2: mask = mask.reshape(normal_gt.shape[:2])
        if mask.dtype != np.float32:
            mask = (mask > 0.5).astype(np.float32) if mask.max() <= 1.0 else (mask > 128).astype(np.float32)

        # 3. Load exactly 3 fixed images for testing
        image_paths = obj_info["image_paths"]
        total_imgs = len(image_paths)
        
        # Test: Fix seed to ensure the same 3 images are sampled for consistent results
        random.seed(42 + idx)
        indices = sorted(random.sample(range(total_imgs), min(3, total_imgs)))
        random.seed()
        
        test_images = []
        for i in indices:
            img = DiLiGentDataset._load_single_image(image_paths[i])
            test_images.append(img)
        test_images = np.stack(test_images, axis=0)

        # 4. Normalize
        mask_bool = mask > 0.5
        for i in range(test_images.shape[0]):
            if mask_bool.any():
                m = test_images[i][mask_bool].mean()
                s = test_images[i][mask_bool].std() + 1e-8
                test_images[i] = (test_images[i] - m) / s

        # Shape: (3, H, W)
        img_tensor = torch.from_numpy(test_images).float()
        normal_tensor = torch.from_numpy(normal_gt).float().permute(2, 0, 1)  
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)  
        
        return img_tensor, normal_tensor, mask_tensor, obj_info["name"]
      

class SyntheticDataset(Dataset):
    def __init__(self, json_path, patch_size=256, alpha_min=0.35, augment=True): 
        with open(json_path, 'r') as f:
            self.folder_list = json.load(f)
        self.patch_size = patch_size
        self.alpha_min = alpha_min
        self.augment = augment
        self.K_train = 3 

    def __len__(self): 
        return len(self.folder_list)

    def load_exr(self, path):
        if not os.path.exists(path): return None
        try:
            img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img is None: return None
            if len(img.shape) == 2: img = cv2.merge([img, img, img])
            return img.astype(np.float32)
        except: return None

    def read_light_means(self, folder_path):
        mean_ip, mean_ienv = 1.0, 1.0
        clean_path = folder_path.rstrip('/\\')
        parent_dir = os.path.dirname(clean_path)
        config_path = os.path.join(parent_dir, "light_means.config")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split() 
                        if len(parts) >= 2:
                            if 'point_mean' in parts[0]: mean_ip = float(parts[1])
                            elif 'env_mean' in parts[0]: mean_ienv = float(parts[1])
            except: pass
        return mean_ip, mean_ienv

    def __getitem__(self, idx):
        attempts = 0
        ps = self.patch_size
        
        while attempts < 20:
            folder_path = self.folder_list[idx]
            try:
                normal_gt = self.load_exr(os.path.join(folder_path, "local_normal.exr"))
                mask = self.load_exr(os.path.join(folder_path, "binary_mask.exr"))
                if normal_gt is None or mask is None: raise FileNotFoundError

                normal_gt = normal_gt[:, :, ::-1]       
                normal_gt = normal_gt * 2.0 - 1.0       
                norm = np.linalg.norm(normal_gt, axis=-1, keepdims=True) + 1e-8
                normal_gt = normal_gt / norm
                if mask.ndim == 3: mask = mask[:,:,0]

                H, W = normal_gt.shape[:2]
                y, x = random.randint(0, H-ps), random.randint(0, W-ps)
                for _ in range(30):
                    ty, tx = random.randint(0, H-ps), random.randint(0, W-ps)
                    if mask[ty:ty+ps, tx:tx+ps].sum() > ps*ps*0.05:
                        y, x = ty, tx; break

                mean_ip, mean_ienv = self.read_light_means(folder_path)

                all_files = os.listdir(folder_path)
                all_indices = [int(f.split('_')[2].split('.')[0]) for f in all_files if f.startswith("point_light_") and f.endswith(".exr")]
                
                # Sample exactly 3 illumination directions
                if len(all_indices) > self.K_train: sampled_indices = random.sample(all_indices, self.K_train)
                else: sampled_indices = all_indices[:self.K_train]

                # ======================================================
                # STEP 1: CREATE FOREGROUND IMAGE FROM 3 LIGHT DIRECTIONS
                # ======================================================
                channels = []
                for i in sampled_indices:
                    img_p = self.load_exr(os.path.join(folder_path, f"point_light_{str(i).zfill(5)}.exr"))
                    if img_p is None: raise FileNotFoundError
                    patch_p = img_p[y:y+ps, x:x+ps] / (mean_ip + 1e-8)
                    
                    # Compute Grayscale for object (0 = B, 1 = G, 2 = R in OpenCV format)
                    gray = 0.2989*patch_p[:,:,2] + 0.5870*patch_p[:,:,1] + 0.1140*patch_p[:,:,0]
                    channels.append(gray)

                # Stack into tensor (3, H, W) - Contains only object, no background yet
                object_3_lights = np.stack(channels, axis=0)
                
                # ======================================================
                # STEP 2: LOAD EXACTLY 1 ENVIRONMENT IMAGE FOR BACKGROUND
                # ======================================================
                env_idx = random.randint(1, 10)
                img_e = self.load_exr(os.path.join(folder_path, f"env_light_{str(env_idx).zfill(5)}.exr"))
                if img_e is None: raise FileNotFoundError
                patch_e = img_e[y:y+ps, x:x+ps] / (mean_ienv + 1e-8)
                
                # OpenCV reads EXR in BGR; reverse to RGB and reshape to (3, H, W)
                patch_e_rgb = patch_e[:, :, ::-1].transpose(2, 0, 1)

                # ======================================================
                # STEP 3: COMPOSITE BACKGROUND USING MASK
                # ======================================================
                mask_patch = mask[y:y+ps, x:x+ps]
                # Add channel dimension for broadcasting (1, H, W)
                mask_expanded = np.expand_dims(mask_patch, axis=0) 
                
                # Where mask is active (object), use object_3_lights. Otherwise (background), use patch_e_rgb.
                input_patch = (object_3_lights * mask_expanded) + (patch_e_rgb * (1.0 - mask_expanded))
                
                # Preserve previous Normalization logic
                mask_bool = mask_patch > 0.5
                
                if mask_bool.any():
                    valid_pixels = input_patch[:, mask_bool] 
                    p99 = np.percentile(valid_pixels, 99.0)
                    input_patch = np.clip(input_patch, 0.0, p99)
                    
                    valid_pixels_clipped = input_patch[:, mask_bool]
                    m = valid_pixels_clipped.mean()
                    s = valid_pixels_clipped.std() + 1e-8
                    input_patch = (input_patch - m) / s

                normal_patch = normal_gt[y:y+ps, x:x+ps, :]

                return torch.from_numpy(input_patch.copy()), \
                       torch.from_numpy(normal_patch.copy()).permute(2, 0, 1), \
                       torch.from_numpy(mask_patch.copy()).unsqueeze(0)
            except:
                attempts += 1
                idx = random.randint(0, len(self.folder_list)-1)
        
        return torch.zeros((3, ps, ps)), torch.zeros((3, ps, ps)), torch.zeros((1, ps, ps))


class SyntheticTestDataset(Dataset):
    def __init__(self, json_path, alpha_min=0.35):
        with open(json_path, 'r') as f:
            self.folder_list = json.load(f)
        self.alpha_min = alpha_min

    def __len__(self): return len(self.folder_list)

    def load_exr(self, path):
        if not os.path.exists(path): return None
        try:
            img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img is None: return None
            if len(img.shape) == 2: img = cv2.merge([img, img, img])
            return img.astype(np.float32)
        except: return None

    def read_light_means(self, folder_path):
        mean_ip, mean_ienv = 1.0, 1.0
        clean_path = folder_path.rstrip('/\\')
        parent_dir = os.path.dirname(clean_path)
        config_path = os.path.join(parent_dir, "light_means.config")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split() 
                        if len(parts) >= 2:
                            if 'point_mean' in parts[0]: mean_ip = float(parts[1])
                            elif 'env_mean' in parts[0]: mean_ienv = float(parts[1])
            except: pass
        return mean_ip, mean_ienv

    def __getitem__(self, idx):
        folder_path = self.folder_list[idx]
        normal_gt = self.load_exr(os.path.join(folder_path, "local_normal.exr"))
        mask = self.load_exr(os.path.join(folder_path, "binary_mask.exr"))
        
        if normal_gt is None or mask is None:
             return torch.zeros((3, 512, 512)), torch.zeros((3, 512, 512)), torch.zeros((1, 512, 512)), "error"

        normal_gt = normal_gt[:, :, ::-1]
        normal_gt = normal_gt * 2.0 - 1.0
        norm = np.linalg.norm(normal_gt, axis=-1, keepdims=True) + 1e-8
        normal_gt /= norm
        if mask.ndim == 3: mask = mask[:,:,0]
        
        mean_ip, mean_ienv = self.read_light_means(folder_path)
        
        all_files = os.listdir(folder_path)
        all_indices = [int(f.split('_')[2].split('.')[0]) for f in all_files if f.startswith("point_light_") and f.endswith(".exr")]
        
        # Fix random seed for Test set to guarantee consistent lighting angles across evaluations
        random.seed(42 + idx)
        sampled_indices = random.sample(all_indices, 3) if len(all_indices) >= 3 else all_indices
        random.seed() # Reset random state

        # ======================================================
        # STEP 1: FOREGROUND (FIXED, REMOVED RANDOM ALPHA)
        # ======================================================
        channels = []
        for i in sampled_indices:
            img_p = self.load_exr(os.path.join(folder_path, f"point_light_{str(i).zfill(5)}.exr"))
            if img_p is None: 
                # Prevent shape mismatch crashes caused by missing files (e.g., remaining 2 channels)
                channels.append(np.zeros_like(mask))
                continue
                
            img_p = img_p / (mean_ip + 1e-8)
            gray = 0.2989*img_p[:,:,2] + 0.5870*img_p[:,:,1] + 0.1140*img_p[:,:,0]
            channels.append(gray)

        object_3_lights = np.stack(channels, axis=0)
        
        # ======================================================
        # STEP 2: BACKGROUND (FIXED ENV_LIGHT FOR TESTING)
        # ======================================================
        # Consistently use env_light_00001.exr as test background for deterministic evaluation
        img_e = self.load_exr(os.path.join(folder_path, f"env_light_00001.exr"))
        if img_e is not None:
            img_e = img_e / (mean_ienv + 1e-8)
            patch_e_rgb = img_e[:, :, ::-1].transpose(2, 0, 1) # BGR -> RGB, shape (3, H, W)
        else:
            patch_e_rgb = np.zeros_like(object_3_lights)

        # ======================================================
        # STEP 3: BACKGROUND COMPOSITING VIA MASK
        # ======================================================
        mask_expanded = np.expand_dims(mask, axis=0) # (1, H, W)
        input_patch = (object_3_lights * mask_expanded) + (patch_e_rgb * (1.0 - mask_expanded))
        
        # ======================================================
        # STEP 4: NORMALIZATION (OBJECT REGION ONLY)
        # ======================================================
        mask_bool = mask > 0.5
        if mask_bool.any():
            valid_pixels = input_patch[:, mask_bool]
            p99 = np.percentile(valid_pixels, 99.0)
            input_patch = np.clip(input_patch, 0.0, p99)
            
            valid_pixels_clipped = input_patch[:, mask_bool]
            m = valid_pixels_clipped.mean()
            s = valid_pixels_clipped.std() + 1e-8
            input_patch = (input_patch - m) / s

        name = os.path.basename(os.path.dirname(folder_path)) + "_" + os.path.basename(folder_path)
        
        return torch.from_numpy(input_patch.copy()), \
               torch.from_numpy(normal_gt.copy()).permute(2, 0, 1), \
               torch.from_numpy(mask.copy()).unsqueeze(0), name