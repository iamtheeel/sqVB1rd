import torch
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as trans
from pathlib import Path
import matplotlib.pyplot as plt
from BGR2RGB565 import RGB2RGB565
import cv2

# Our Data is to be 
#   Size:   160 X 120
#   Format: CAM_IMAGE_PIX_FMT_RGB565

class DataPreparation:

    def __init__(self, data_save_path):
        self.data_save_path = Path(data_save_path)

        self.train_dir = self.data_save_path / "train"
        self.test_dir = self.data_save_path / "test"

        self.classes = ['Bird', 'Empty', 'Squirrel'] # Its loaded alphabeticlyh
        self.width = 96 #smalest our sensor can do
        self.height = 96
        self.camWidth = 320 #QQVGA
        self.camHeight = 240
        self.depth = 2 # RGB565 is 2 bits for 3 colors

    def get_data(self, displayImages=False):
        Trans = trans.Compose([
            #trans.Resize((self.camHeight, self.camWidth)), #Resize is H/w
            trans.CenterCrop(self.width),
            RGB2RGB565(),
            trans.ToTensor()
        ])

        # Setup training data
        train_data = datasets.ImageFolder(root=self.train_dir, transform=Trans)
        test_data = datasets.ImageFolder(root=self.test_dir,  transform=Trans)

        # From https://stackoverflow.com/questions/54897646/pytorch-datasets-converting-entire-dataset-to-numpy
        from torch.utils.data import DataLoader
        loaderForRepData = DataLoader(train_data, batch_size=1)
        #loaderForRepData = DataLoader(train_data, batch_size=len(train_data))
        #np.save(file='data/calibdata.npy', arr=calib_datas)
        #calib_datas = np.vstack(img_datas)
        np.save("../output/representive_data",next(iter(loaderForRepData))[0].numpy() ) # itertes over the batch... not the image



        return train_data, test_data