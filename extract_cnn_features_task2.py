"""
extract_cnn_features_task2.py
-----------------------------
Extract 512-dim feature embeddings from a pretrained ResNet-18 (ImageNet
weights) for every image in Task 2 (fine-grained bird species).

Spec compliance: the spec forbids feature extractors pretrained on
CIFAR-10 or CUB-200-2011 but explicitly allows ImageNet-pretrained models.
ResNet-18 (ImageNet1K v1) qualifies. The classifier head is removed and
we read the 512-dim post-avgpool vector.

Task 2 images are 128x128; we upsample to 224 (the input size ResNet-18
was trained on). For a small/fine-grained dataset like this one, ResNet-18
is actually a reasonable default: its 512-dim embedding is less prone to
overfit a linear/SVM classifier on ~350 training samples than the 2048-dim
ResNet-50 embedding would be. ResNet-50 is worth trying as a separate
extractor if the small-data regularisation works out.

Output: outputs/cnn_features_task2.csv (image_id, cnn_0, ..., cnn_511)
"""

import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image


DATA_DIR = "task2_data"
OUT_PATH = "outputs/cnn_features_task2.csv"
BATCH = 64

# ImageNet preprocessing. Resize(224) preserves aspect ratio (the bird crops
# are roughly square already), then CenterCrop tidies to 224x224 exactly so
# the avgpool sees a fixed spatial grid.
preprocess = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


class ImageList(Dataset):
    """Loads images from a list of (image_id, full_path) pairs."""
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        image_id, path = self.records[i]
        img = Image.open(path).convert("RGB")
        return image_id, preprocess(img)


def build_records():
    """Return [(image_id, path), ...] for all train+test images."""
    tr = pd.read_csv(os.path.join(DATA_DIR, "train_metadata.csv"))
    te = pd.read_csv(os.path.join(DATA_DIR, "test_metadata.csv"))
    records = []
    for _, r in tr.iterrows():
        records.append((r["image_id"], os.path.join(DATA_DIR, r["image_path"])))
    for _, r in te.iterrows():
        records.append((r["image_id"], os.path.join(DATA_DIR, r["image_path"])))
    return records


def build_model():
    """ResNet-18 pretrained on ImageNet, with the final FC removed.
    Output is the 512-dim post-avgpool vector."""
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity()
    m.eval()
    return m


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    records = build_records()
    print(f"Total images: {len(records)}")

    model = build_model()
    loader = DataLoader(ImageList(records), batch_size=BATCH,
                        shuffle=False, num_workers=2)

    ids, feats = [], []
    t0 = time.time()
    with torch.no_grad():
        for i, (batch_ids, batch_imgs) in enumerate(loader):
            out = model(batch_imgs).numpy()
            feats.append(out)
            ids.extend(batch_ids)
            if (i + 1) % 5 == 0:
                done = (i + 1) * BATCH
                print(f"  {min(done, len(records))}/{len(records)}  "
                      f"({time.time() - t0:.1f}s elapsed)")

    feats = np.vstack(feats)
    print(f"Extraction done in {time.time() - t0:.1f}s. Feature matrix: {feats.shape}")

    cols = [f"cnn_{i}" for i in range(feats.shape[1])]
    df = pd.DataFrame(feats, columns=cols)
    df.insert(0, "image_id", ids)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
