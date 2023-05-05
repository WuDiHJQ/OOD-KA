import os
from PIL import Image
from torch.utils.data import Dataset


class OOD_Segment(Dataset):
    # OOD ImageSet For Segment

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.files = os.listdir(self.root)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.files[idx])).convert("RGB")
        lbl = img.copy()  # never used
        if self.transform is not None:
            img, lbl = self.transform(img, lbl)
        return img, lbl
