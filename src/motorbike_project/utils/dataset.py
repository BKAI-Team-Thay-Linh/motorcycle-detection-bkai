import json
import os
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset
import numpy as np
import polars as pl

import motorbike_project as mp


class MotorBikeDataset(Dataset):
    def __init__(self, config_path: str, session: str = 'train', data_mode: str = 'ssl', **kwargs):
        """
            In this class, there are two data mode to choose from:
            - `csv`: You need to provide a folder containing the images, and a csv file containing the labels

            - `ssl`: In this mode, you just need to import list of folder paths, which are divided into classes already

        Args:
            `config_path` (str): The path to the config file
            `session` (str, optional): The session of the dataset, must be in [`train`, `val`, `test`]
            `data_mode` (str, optional): The data mode. Defaults to `csv`.
            `kwargs`: Other arguments:

            - For `csv` mode:
                `folder_path` (str): The folder containing the images
                `csv_path` (str): The csv file containing the labels
            - For `ssl` mode:
                'folder_paths' (list): The list of folder paths, each folder path is a string to the folder containing the images of a class
        """

        assert session in ['train', 'val', 'test'], 'Invalid session, must be in [train, val, test]'

        assert data_mode in ['csv', 'ssl'], 'Invalid data mode, must be in [csv, ssl]'

        self.data_mode = data_mode
        self.session = session
        self.kwargs = kwargs
        self.config_path = config_path

        if not os.path.exists(config_path):
            raise ValueError(f'Config path {config_path} does not exist')

        with open(os.path.join(self.config_path, 'class.json'), 'r') as f:
            self.config_class: dict = json.load(f)

        # Define the image transform
        self.transform = mp.Transform(session)

        # Load the dataset in the folder
        self.load_dataset()

    def load_dataset(self):
        if self.data_mode == 'csv':
            self._csv_mode()
        else:
            self._ssl_mode()

    def __read_csv(self, csv_path: str) -> dict:
        if not os.path.exists(csv_path):
            raise ValueError(f'CSV path {csv_path} does not exist')

        df = pl.read_csv(csv_path, infer_schema_length=0)
        mapping_output = {}

        rows = df.rows(named=True)

        for row in rows:
            if row['imagename'] not in mapping_output:
                mapping_output[row['imagename']] = row['answer']

        return mapping_output

    def _csv_mode(self):
        folder_path = self.kwargs.get('folder_path', None)
        csv_path = self.kwargs.get('csv_path', None)

        if not os.path.exists(folder_path):
            raise ValueError(f'Folder path {folder_path} does not exist')
        if not os.path.exists(csv_path):
            raise ValueError(f'CSV path {csv_path} does not exist')

        # Read the csv file and get the mapping
        self.labels = {}
        self.mapping_result = self.__read_csv(csv_path)
        count = 0

        for img in os.listdir(folder_path):
            folder, idx, _ = img.split('_')
            img_name = f'{folder}_image_{idx}.jpg'
            img_path = os.path.join(folder_path, img)

            if img_name not in self.mapping_result:
                with open('log.txt', 'a') as f:
                    f.write(f'Image {img} ---> {img_name} does not exist in the csv file\n')
                print(f'Image {img} ---> {img_name} does not exist in the csv file')
                count += 1
                continue

        self.labels[img_path] = int(self.mapping_result[img_name]) if self.mapping_result[img_name] in ('0', '1') else 2

        print(f'Number of images not in the csv file: {count}')

    def _ssl_mode(self):
        self.folder_paths = self.kwargs.get('folder_paths', None)
        self.labels = {}

        # 1 folder is "xe_so", 2 folder is "xe_ga", 3, 4, 5 folder is "others"
        self.classes = os.listdir(self.folder_paths[0])

        for folder in self.folder_paths:
            for folder_class in self.classes:
                for img in os.listdir(os.path.join(folder, folder_class)):
                    img_path = os.path.join(folder, folder_class, img)
                    self.labels[img_path] = int(folder_class) - 1 if folder_class in ('1', '2') else 2

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path = list(self.labels.keys())[index]
        label = self.labels[img_path]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            # Convert the image to numpy array
            img_np = np.array(img)

            img = self.transform(img_np)

        return img, label


if __name__ == '__main__':
    train_dataset = MotorBikeDataset(
        config_path=r'C:\Users\QUANPC\Documents\GitHub\Motocycle-Detection-BKAI\src\motorbike_project\config',
        session='train',
        data_mode='csv',
        folder_path=r'D:\Data Deep Learning\FINAL-DATASET\final_dataset',
        csv_path=r'D:\Data Deep Learning\FINAL-DATASET\result.csv'
    )

    # print(train_dataset._read_csv(r'D:\Data Deep Learning\FINAL-DATASET\result.csv'))

    print(train_dataset[0])
