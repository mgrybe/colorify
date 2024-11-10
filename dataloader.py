import glob
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from skimage import color

SIZE=256

class ColorizationDataset(Dataset):
    def __init__(self, paths, split='train'):
        self.transforms = None
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((SIZE, SIZE),  InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip()
            ])
        elif split == 'val':
            self.transforms = transforms.Resize((SIZE, SIZE),  InterpolationMode.BICUBIC)

        self.split = split
        self.size = SIZE
        self.paths = paths

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img) if self.transforms is not None else img
        img = np.array(img)
        img_lab = color.rgb2lab(img).astype("float32") # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1. # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110. # Between -1 and 1

        return L, ab

    def __len__(self):
        return len(self.paths)

if __name__ == '__main__':
    np.random.seed(123)

    train_paths = glob.glob("./dataset/faces-256px/train/**", recursive=True) # Your path for your dataset
    train_paths = [file for file in train_paths if file.endswith(('.png', '.jpg'))]
    val_paths = glob.glob("./dataset/faces-256px/val/**", recursive=True) # Your path for your dataset
    val_paths = [file for file in val_paths if file.endswith(('.png', '.jpg'))]
    print(f'train={len(train_paths)}, val={len(val_paths)}')