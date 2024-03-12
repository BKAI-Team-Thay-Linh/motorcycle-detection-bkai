import json
import os
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset
import numpy as np
import polars as pl

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import motorbike_project as mp


class MotorBikeDataset(Dataset):
    def __init__(self, session: str = 'train', folder_path: str = ''):
        """
            In this class, there are two data mode to choose from:
            - `csv`: You need to provide a folder containing the images, and a csv file containing the labels

            - `ssl`: In this mode, you just need to import list of folder paths, which are divided into classes already

        Args:
            `config_path` (str): The path to the config file
            `session` (str, optional): The session of the dataset, must be in [`train`, `val`, `test`]
            `folder_path` (str, optional): The path to the folder containing the images. Hence the label is in folder "labels"

        """

        self.session = session
        self.transform = mp.Transform(session=session)
        self.folder_path = folder_path
        self.load_dataset()

    def _get_label(self, img, label_dir):
        # Read the label
        name = img.split('.')[0]
        with open(os.path.join(label_dir, f'{name}.txt'), 'r') as f:
            label = int(f.read())
        return label

    def load_dataset(self):
        self.labels = {}

        if self.session == 'train':
            img_dir = os.path.join(self.folder_path, 'train', 'images')
            label_dir = os.path.join(self.folder_path, 'train', 'labels')
        elif self.session == 'val':
            img_dir = os.path.join(self.folder_path, 'valid', 'images')
            label_dir = os.path.join(self.folder_path, 'valid', 'labels')
        else:
            img_dir = os.path.join(self.folder_path, 'test', 'images')
            label_dir = os.path.join(self.folder_path, 'test', 'labels')

        # Read the csv file
        images_path = tuple(os.listdir(img_dir))
        labels_path = tuple(os.listdir(label_dir))
        futures = {}

        with ThreadPoolExecutor(max_workers=100) as executor:
            print('Start processing images')
            for idx, img in enumerate(images_path):
                print(f'{idx:>6}|{len(images_path):<6} - Submitting {img}', end='\r')
                futures[executor.submit(self._get_label, img, label_dir)] = img

            print()
            print('Start getting results')
            for idx, future in enumerate((as_completed(futures))):
                label = future.result()
                img = futures[future]
                print(f'{idx:>6}|{len(images_path):<6} - Processing {img} - {label}', end='\r')
                self.labels[os.path.join(img_dir, img)] = label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path = list(self.labels.keys())[index]
        label = self.labels[img_path]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img_np = np.array(img)
            img = self.transform(img_np)

        return img, label


if __name__ == '__main__':
    train_dataset = MotorBikeDataset(
        session='train',
        folder_path='/home/linhdang/workspace/PAPER_Material/FINAL-DATASET/train'
    )

    print(train_dataset[0])
