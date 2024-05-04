import torch
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as trans
from pathlib import Path
import matplotlib.pyplot as plt

# Our Data is to be 
#   Size:   160 X 120
#   Format: CAM_IMAGE_PIX_FMT_RGB565

class DataPreparation:

    def __init__(self, data_save_path):
        self.data_save_path = Path(data_save_path)

        self.train_dir = self.data_save_path / "train"
        self.test_dir = self.data_save_path / "test"

        self.classes = ['Bird', 'Empty', 'Squirrel'] # Its loaded alphabeticlyh
        #self.width = 96 #smalest our sensor can do
        #self.height = 96
        self.width = 160 #QQVGA
        self.height = 120

    def get_data(self):
        trainTrans = trans.Compose([
            trans.Resize((320, 240)),
            trans.CenterCrop(96),
            #trans.Resize((self.height, self.width)),
            trans.ToTensor()
        ])

        # Setup training data
        train_data = datasets.ImageFolder(root=self.train_dir, transform=trainTrans)
        test_data = datasets.ImageFolder(root=self.test_dir, transform=trainTrans)

        img = train_data[0][0]
        plt.figure(figsize=(10, 10))
        plt.imshow(img.permute(1, 2,0))
        plt.show()

        return train_data, test_data