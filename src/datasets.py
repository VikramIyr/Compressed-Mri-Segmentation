import os
import glob
import random
import pickle
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# -------------------------
# Utilities
# -------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_nifti(path: str) -> np.ndarray:
    import nibabel as nib
    return np.asarray(nib.load(path).get_fdata())


def robust_normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Robust per-slice normalization:
      - clip to [p1, p99]
      - z-score
    """
    p1, p99 = np.percentile(x, (1, 99))
    x = np.clip(x, p1, p99)
    return (x - x.mean()) / (x.std() + eps)


def resize_bilinear(img: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """
    Bilinear resize for images.
    Supports:
      - (H,W)
      - (C,H,W)
    Returns:
      - (H,W) or (C,H,W) respectively
    """
    import torch.nn.functional as F

    t = torch.from_numpy(img).float()
    if t.ndim == 2:
        t = t[None, None]      # (1,1,H,W)
        t = F.interpolate(t, size=out_hw, mode="bilinear", align_corners=False)
        return t.squeeze(0).squeeze(0).numpy()  # (H,W)

    if t.ndim == 3:
        t = t[None]            # (1,C,H,W)
        t = F.interpolate(t, size=out_hw, mode="bilinear", align_corners=False)
        return t.squeeze(0).numpy()             # (C,H,W)

    raise ValueError(f"resize_bilinear expects 2D or 3D array, got shape {img.shape}")


def resize_nn_mask(mask: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """
    Nearest-neighbor resize for masks.
    Input:  (H,W)
    Output: (H,W)
    """
    import torch.nn.functional as F

    if mask.ndim != 2:
        raise ValueError(f"resize_nn_mask expects (H,W), got {mask.shape}")

    t = torch.from_numpy(mask).float()[None, None]  # (1,1,H,W)
    t = F.interpolate(t, size=out_hw, mode="nearest")
    out = t.squeeze(0).squeeze(0).numpy()           # (H,W)
    return out


# -------------------------
# Case indexing
# -------------------------

@dataclass(frozen=True)
class Case:
    case_id: str
    modalities: Dict[str, str]
    seg: str


def build_brats_index(root: str, modalities: List[str]) -> List[Case]:
    """
    root should contain folders like:
      data/BraTS2021_00000/
        *_flair.nii.gz
        *_t1ce.nii.gz
        *_t1.nii.gz
        *_t2.nii.gz
        *_seg.nii.gz
    """
    cases: List[Case] = []

    for d in sorted(os.listdir(root)):
        case_dir = os.path.join(root, d)
        if not os.path.isdir(case_dir):
            continue

        try:
            mods = {m: glob.glob(os.path.join(case_dir, f"*_{m}.nii.gz"))[0] for m in modalities}
            seg = glob.glob(os.path.join(case_dir, "*_seg.nii.gz"))[0]
        except IndexError:
            # Skip folders that aren't BraTS cases
            continue

        cases.append(Case(case_id=d, modalities=mods, seg=seg))

    if len(cases) == 0:
        raise RuntimeError(f"No BraTS cases found under {root}. Check BRATS_ROOT.")

    return cases


def split_cases(
    cases: List[Case], val_ratio: float = 0.2, seed: int = 0
) -> Tuple[List[Case], List[Case]]:
    rng = random.Random(seed)
    idx = list(range(len(cases)))
    rng.shuffle(idx)
    n_val = int(len(cases) * val_ratio)
    val_idx = set(idx[:n_val])
    train = [c for i, c in enumerate(cases) if i not in val_idx]
    val = [c for i, c in enumerate(cases) if i in val_idx]
    return train, val


# -------------------------
# Dataset
# -------------------------

class Brats2DSliceDataset(Dataset):
    """
    BraTS → 2D slice dataset with compression options.
    Returns:
      img:  FloatTensor (C,H,W)
      mask: LongTensor  (H,W)
      meta: dict
    """

    def __init__(
        self,
        cases: List[Case],
        modalities: List[str],
        slice_axis: int = 2,
        slice_stride: int = 1,
        downsample: int = 1,
        target_hw: Tuple[int, int] = (240, 240),
        keep_empty: bool = False,
        empty_ratio: float = 0.25,
        seed: int = 0,
        cache_dir: str = "cache",
    ):
        assert slice_stride >= 1
        assert downsample >= 1

        self.cases = cases
        self.modalities = modalities
        self.slice_axis = slice_axis
        self.slice_stride = slice_stride
        self.downsample = downsample
        self.target_hw = target_hw
        self.keep_empty = keep_empty
        self.empty_ratio = empty_ratio
        self.seed = seed

        self.rng = np.random.RandomState(seed)

        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        self.index = self._load_or_build_index()

    def _cache_path(self) -> str:
        # Include key params that affect which slices are included
        case_hash = f"n{len(self.cases)}"
        key = f"{case_hash}_axis{self.slice_axis}_stride{self.slice_stride}_keep{int(self.keep_empty)}_er{self.empty_ratio}_seed{self.seed}"
        return os.path.join(self.cache_dir, f"slice_index_{key}.pkl")

    def _load_or_build_index(self) -> List[Tuple[int, int]]:
        path = self._cache_path()
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)

        idx: List[Tuple[int, int]] = []
        empty: List[Tuple[int, int]] = []

        for ci, case in enumerate(self.cases):
            seg = load_nifti(case.seg).astype(np.int16)
            n_slices = seg.shape[self.slice_axis]

            for s in range(0, n_slices, self.slice_stride):
                mask2d = np.take(seg, s, axis=self.slice_axis)
                if mask2d.max() == 0:
                    empty.append((ci, s))
                else:
                    idx.append((ci, s))

        if self.keep_empty and len(empty) > 0:
            n_keep = int(len(empty) * self.empty_ratio)
            n_keep = max(0, min(n_keep, len(empty)))
            if n_keep > 0:
                sel = self.rng.choice(len(empty), n_keep, replace=False)
                idx.extend([empty[i] for i in sel])

        self.rng.shuffle(idx)

        with open(path, "wb") as f:
            pickle.dump(idx, f)

        return idx

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ci, s = self.index[i]
        case = self.cases[ci]

        # Load modalities slice
        imgs = []
        for m in self.modalities:
            vol = load_nifti(case.modalities[m]).astype(np.float32)
            sl = np.take(vol, s, axis=self.slice_axis)  # (H,W)
            sl = robust_normalize(sl)
            imgs.append(sl)

        img = np.stack(imgs, axis=0)  # (C,H,W)

        # Load mask slice
        seg = load_nifti(case.seg).astype(np.int16)
        mask = np.take(seg, s, axis=self.slice_axis)  # (H,W)

        # Optional spatial downsample (compression)
        if self.downsample > 1:
            h, w = mask.shape
            new_hw = (max(1, h // self.downsample), max(1, w // self.downsample))
            img = resize_bilinear(img, new_hw)       # (C,h',w')
            mask = resize_nn_mask(mask, new_hw)      # (h',w')

        # Resize to fixed size for batching
        img = resize_bilinear(img, self.target_hw)   # (C,H,W)
        mask = resize_nn_mask(mask, self.target_hw)  # (H,W)

        # Tensors
        img_t = torch.from_numpy(img).float()
        mask_t = torch.from_numpy(mask).long()

        meta = {
            "case_id": case.case_id,
            "slice_idx": int(s),
            "slice_stride": int(self.slice_stride),
            "downsample": int(self.downsample),
        }
        return img_t, mask_t, meta
