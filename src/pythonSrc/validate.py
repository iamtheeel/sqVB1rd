###
# validate.py
#
# Joshua Mehlman
# ENGR 859 Spring 2024
# Term Project
#
# Squirrl Or Bird Detector
#
###
# Validate saved data
###

from pathlib import Path


hiddenNerons = 30
MODEL_NAME = "leNetV5.pth"
MODEL_PATH = Path("../models/leNetV5_mod/13_BRG565")


print(f"INIT: Set Loger")
import sys
import os
from ConfigParser import ConfigParser
stdout_logger = logging.getLogger('STDOUT')
config = ConfigParser(os.path.join(os.getcwd(), 'config.yaml'))
meta_config = config.get_config()["meta"]
model_config = config.get_config()["model"]

print(f"INIT: Load Config")
from ConfigParser import ConfigParser

"""
Data Preparation
"""
print(f"INIT: Get Images")
from DataPreparation import DataPreparation
data_preparation = DataPreparation(meta_config["data_save_path"])
train_data, test_data = data_preparation.get_data(displayImages=False)
image_width = data_preparation.width
image_height = data_preparation.height
image_depth = data_preparation.depth

# Must be after Get Images
config = ConfigParser(os.path.join(os.getcwd(), 'config.yaml'))
training_config = config.get_config()["training"]

"""
Load the model
"""
from pathlib import Path
from Model import leNetV5
import torch
MODEL_SAVE_PATH = MODEL_PATH/MODEL_NAME
model = leNetV5(input_shape=image_depth,hidden_units=hiddenNerons,output_shape=len(data_preparation.classes))
model.load_state_dict(torch.load(f=MODEL_SAVE_PATH), strict=False)

runBatch = True
if(runBatch):
    # Run the validation set
    from Trainer import Trainer
    training_config['model'] = model
    training_config['training_data'] = train_data
    training_config['testing_data'] = test_data
    
    trainer = Trainer(**training_config)
    testLoss, testAcc = trainer.test(data_preparation.classes) # Unit test reqires singletion 

else:
    # Run the images one at a time
    from torch.utils.data import DataLoader
    import cv2
    import numpy as np

    model_labels = {0: 'Nobody', 1: 'Bird', 2: 'Squi'}
    model.eval()
    dataloader = DataLoader(test_data,batch_size=25,shuffle=False)

    for images, labels in dataloader:
        #print(f"data len: {len(labels)}")
        for i in range(len(labels)-1):  
            '''
            print(f"images[{i}] size: {images[i].shape}")
            numpy_array = np.asarray(images[i].permute(1,2,0), dtype='>i2') # '>' = bigEndian, 'i' = int, '2' = 2 bytes
            print(f"numpy_array size: {numpy_array.shape}")
            cv2.imshow('', numpy_array)
            cv2.waitKey()
            '''

            # Get a printout of the predictions
            predictions = model(images).squeeze()
            predicted_labels = torch.argmax(predictions, axis=1) # get the index of the largest predic

            true_label = labels[i].item()
            predicted_label = predicted_labels[i].item()

            #print(f"true: {true_label}, pred: {predicted_label}, prdictes: {predictions[i,:]}" )
            with open("../output/predicts.txt", "a") as outFile:
                print(f"true: {true_label}, pred: {predicted_label}, prdictes: {predictions[i,:]}", file=outFile)