import glob
import numpy as np

if __name__ == '__main__':
    np.random.seed(123)

    train_paths = glob.glob("./dataset/faces-256px/train/**", recursive=True) # Your path for your dataset
    train_paths = [file for file in train_paths if file.endswith(('.png', '.jpg'))]
    val_paths = glob.glob("./dataset/faces-256px/val/**", recursive=True) # Your path for your dataset
    val_paths = [file for file in val_paths if file.endswith(('.png', '.jpg'))]
    print(f'train={len(train_paths)}, val={len(val_paths)}')