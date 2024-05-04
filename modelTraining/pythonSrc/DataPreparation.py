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
        self.depth = 2 # BGR2RGB565

    def get_data(self, displayImages=False):
        Trans = trans.Compose([
            trans.Resize((self.camWidth, self.camHeight)),
            trans.CenterCrop(self.width),
            #trans.Resize((self.height, self.width)),
            RGB2RGB565(),
            trans.ToTensor()

        ])

        # Setup training data
        train_data = datasets.ImageFolder(root=self.train_dir, transform=Trans)
        test_data = datasets.ImageFolder(root=self.test_dir,  transform=Trans)

        if(displayImages):
            startImg = 0
            dataLen = len(train_data)
            print(f" display training data. Image count: {dataLen}")
            for i in range(startImg, dataLen):
                print(f"len: {len(train_data[i][0].numpy())}")
                print(f"Displaying: train_data {i} out of {dataLen}. it is {train_data[i][1]}")
                plt.figure(figsize=(10, 10))
                #plt.imshow(np.dstack(train_data[i][0]))
                numpy_array = np.asarray(train_data[i][0].permute(1,2,0))
                print(f"numpy array len: {len(numpy_array)}")
                print(f"numpy array: {numpy_array}")
                #numpy_array = cv2.cvtColor(train_data[i][0], cv2.COLOR_RGB2BGR565) # convert RGB565 to rgb
                #numpy_array = numpy_array.reshape(1,2,self.width, self.height)
                numpy_array = cv2.cvtColor(numpy_array, cv2.COLOR_BGR5652RGB) # convert RGB565 to rgb
                #image = numpy_array.reshape(1,3,self.camWidth,self.camWidth)# 2 is from RGB565, 2 bytes for 3 colors
                plt.imshow(numpy_array)
                #plt.imshow(numpy_array.permute(1, 2,0))
                #plt.imshow(train_data[i][0].permute(1, 2,0))
                plt.xlabel(f"train_data {i} out of {dataLen}. it is {train_data[i][1]}")

                #plt.canvas.mpl_connect('key_press_event', on_press)
                plt.show()
                plt.close()

        return train_data, test_data