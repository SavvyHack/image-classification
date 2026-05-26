"""
extract_extra_features.py
-------------------------
Engineer additional hand-crafted features from the raw 64x64 images,
complementing the provided color histogram / HOG-PCA / additional features.

Two feature groups are computed:

  1. LBP (Local Binary Patterns) histograms on the grayscale image.
     LBP captures local texture patterns invariant to monotonic intensity
     changes -- a different texture signal than HOG (which captures gradient
     orientation). Useful for distinguishing fur, feathers, skin, and the
     fine speckling on butterflies and spiders.

  2. HSV colour statistics. The provided color histogram bins RGB, which
     tangles brightness and hue. HSV separates them: hue (what colour),
     saturation (colour purity), value (brightness). Mean and standard
     deviation of each give a compact 6-dim summary, plus a finer 24-bin
     hue histogram for sensitive hue distinctions.

Output:  outputs/extra_features.csv  (image_id, lbp_*, hsv_*)
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage import color
from skimage.feature import local_binary_pattern

DATA_DIR = "/home/claude/task1_data"
OUT_PATH = "/home/claude/project/outputs/extra_features.csv"

# LBP settings -- uniform LBP with P=8 sample points, radius=1.
# 'uniform' LBP gives P+2 distinct labels (P uniform patterns + 1 non-uniform).
LBP_P, LBP_R = 8, 1
N_LBP_BINS  = LBP_P + 2          # 10 bins
N_HUE_BINS  = 24                 # for fine hue histogram


def features_for_image(img_pil):
    """Return a 1D feature vector for one PIL Image (RGB)."""
    rgb = np.asarray(img_pil, dtype=np.float32) / 255.0   # H x W x 3 in [0,1]

    # --- LBP on grayscale (uint8 for stable integer comparisons)
    gray = (color.rgb2gray(rgb) * 255).astype(np.uint8)
    lbp = local_binary_pattern(gray, LBP_P, LBP_R, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=N_LBP_BINS,
                               range=(0, N_LBP_BINS), density=True)

    # --- HSV stats: per-channel mean and std (6 dims)
    hsv = color.rgb2hsv(rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hsv_stats = np.array([h.mean(), h.std(),
                          s.mean(), s.std(),
                          v.mean(), v.std()], dtype=np.float32)

    # --- Hue histogram (24 bins) -- only meaningful where saturation is high
    # (low-sat pixels have no well-defined hue). Weight by saturation.
    hue_hist, _ = np.histogram(h.ravel(), bins=N_HUE_BINS, range=(0, 1),
                               weights=s.ravel(), density=False)
    if hue_hist.sum() > 0:
        hue_hist = hue_hist / hue_hist.sum()    # normalise to a distribution

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
        img  = Image.open(path).convert("RGB")
        feats.append(features_for_image(img))
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(all_meta)}")
    feats = np.vstack(feats)
    print(f"Feature matrix: {feats.shape}")

    cols = ([f"lbp_{i}"     for i in range(N_LBP_BINS)]
            + ["hsv_h_mean","hsv_h_std","hsv_s_mean","hsv_s_std","hsv_v_mean","hsv_v_std"]
            + [f"hue_{i}"   for i in range(N_HUE_BINS)])
    df = pd.DataFrame(feats, columns=cols)
    df.insert(0, "image_id", all_meta["image_id"].values)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
