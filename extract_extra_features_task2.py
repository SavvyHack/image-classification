"""
extract_extra_features_task2.py
-------------------------------
Engineer hand-crafted features from the raw 128x128 bird images. Same
spirit as extract_extra_features.py for Task 1, with one Task-2-motivated
adjustment: a finer hue histogram (36 bins instead of 24).

Rationale: many of the 10 bird species are defined by colour rather than
shape -- Cardinal (red), Blue Jay (blue), American Goldfinch and Yellow
Warbler (yellow), Red-Winged Blackbird (black with red shoulder), Wilson
Warbler (yellow + black cap). Hue is therefore likely to be one of the
strongest hand-crafted signals on this task, even though it cannot
distinguish the gull species or the sparrow species (which share colour).

Two feature groups, in this order:
1. LBP (Local Binary Patterns) histogram on grayscale -- 10 bins.
   Captures local texture: feather patterning, breast streaks, wing bars.
2. HSV summary -- 6-dim per-channel mean/std + 36-bin saturation-weighted
   hue histogram. Saturation-weighting suppresses contribution from
   near-grey pixels (sky, plain backgrounds, gull bodies) where hue is
   poorly defined.

Output: outputs/extra_features_task2.csv  (image_id, lbp_*, hsv_*, hue_*)
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage import color
from skimage.feature import local_binary_pattern


DATA_DIR = "task2_data"
OUT_PATH = "outputs/extra_features_task2.csv"

LBP_P, LBP_R = 8, 1
N_LBP_BINS = LBP_P + 2        # 10 bins
N_HUE_BINS = 36               # finer than Task 1's 24 -- bird colours are
                              # diagnostic and we want narrow hue cells


def features_for_image(img_pil):
    """Return a 1D feature vector for one PIL Image (RGB)."""
    rgb = np.asarray(img_pil, dtype=np.float32) / 255.0      # H x W x 3 in [0,1]

    # --- LBP on grayscale
    gray = (color.rgb2gray(rgb) * 255).astype(np.uint8)
    lbp = local_binary_pattern(gray, LBP_P, LBP_R, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=N_LBP_BINS,
                               range=(0, N_LBP_BINS), density=True)

    # --- HSV summary
    hsv = color.rgb2hsv(rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hsv_stats = np.array([h.mean(), h.std(),
                          s.mean(), s.std(),
                          v.mean(), v.std()], dtype=np.float32)

    # --- Saturation-weighted hue histogram. Greys (low s) contribute almost
    # nothing; the histogram therefore reflects the dominant *colourful*
    # pixels of the bird/background, not the white/grey gull bodies.
    hue_hist, _ = np.histogram(h.ravel(), bins=N_HUE_BINS, range=(0, 1),
                               weights=s.ravel(), density=False)
    if hue_hist.sum() > 0:
        hue_hist = hue_hist / hue_hist.sum()

    return np.concatenate([lbp_hist, hsv_stats, hue_hist]).astype(np.float32)


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    tr = pd.read_csv(os.path.join(DATA_DIR, "train_metadata.csv"))
    te = pd.read_csv(os.path.join(DATA_DIR, "test_metadata.csv"))
    all_meta = pd.concat([tr[["image_id", "image_path"]],
                          te[["image_id", "image_path"]]], ignore_index=True)

    feats = []
    for i, row in enumerate(all_meta.itertuples(index=False)):
        path = os.path.join(DATA_DIR, row.image_path)
        img = Image.open(path).convert("RGB")
        feats.append(features_for_image(img))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(all_meta)}")

    feats = np.vstack(feats)
    print(f"Feature matrix: {feats.shape}")

    cols = ([f"lbp_{i}" for i in range(N_LBP_BINS)]
            + ["hsv_h_mean", "hsv_h_std",
               "hsv_s_mean", "hsv_s_std",
               "hsv_v_mean", "hsv_v_std"]
            + [f"hue_{i}" for i in range(N_HUE_BINS)])
    df = pd.DataFrame(feats, columns=cols)
    df.insert(0, "image_id", all_meta["image_id"].values)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
