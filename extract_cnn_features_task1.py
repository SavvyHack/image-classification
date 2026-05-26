"""
Extract 512-dim feature embeddings from a pretrained ResNet-18 for every image in Task 1.

We use a model pretrained on ImageNet. The classifier head is removed and we take the
output of the global-average-pooling layer. Images are upsampled from 64x64 to 224x224 to match
the input size the network was trained on, and normalised with ImageNet
mean/std.
"""
import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

DATA_DIR  = "task1_data"
OUT_PATH  = "outputs/cnn_features.csv"
BATCH     = 64

# ImageNet preprocessing: resize and normalise.
preprocess = transforms.Compose([
    transforms.Resize(224),
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
            if (i + 1) % 10 == 0:
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
