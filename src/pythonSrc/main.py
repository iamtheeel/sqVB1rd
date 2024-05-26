###
# saveModel.py
#
# Joshua Mehlman
# ENGR 859 Spring 2024
# Term Project
#
# Squirrl Or Bird Detector
#
###
# Main training entry point
###


print(f"INIT: SQB, V0.1Alpha ")
import torch
import os
import sys
import logging
from DataPreparation import DataPreparation
from Model import leNetV5, MobileNetV3, AlexNet,  mobileNetV1
from ConfigParser import ConfigParser
from Trainer import Trainer

from OpCounter import countOperations, saveInfo, timeStrFromS
from torchinfo import summary
import csv
from timeit import default_timer as timer

## Optimise
from optimize import optimiser


print(f"INIT: Load Config")
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename="mylog.log",
    filemode='a'
)

print(f"INIT: Check Log")
if os.path.exists("mylog.log"):
    log_file_size = os.path.getsize("mylog.log")
    if log_file_size > 2 * 1024 * 1024:
        os.remove("mylog.log")
        print("The log file was greater than 2MB and has been deleted.")

config = ConfigParser(os.path.join(os.getcwd(), 'config.yaml'))
meta_config = config.get_config()["meta"]

"""
Data Preparation
"""
print(f"INIT: Get Images")
data_preparation = DataPreparation(meta_config["data_save_path"])
train_data, test_data = data_preparation.get_data(displayImages=False)
image_width = data_preparation.width
image_height = data_preparation.height
image_depth = data_preparation.depth
#exit()
"""
Hyperparameters preparation
"""
config = ConfigParser(os.path.join(os.getcwd(), 'config.yaml'))
training_config = config.get_config()["training"]

model_name = config.get_config()["model"]
if model_name == "leNetV5":
    model = leNetV5(input_shape=image_depth,hidden_units=30,output_shape=len(data_preparation.classes))
elif model_name == "MobileNetV3":
    model = MobileNetV3(num_classes=len(data_preparation.classes))
elif model_name == "AlexNet":
    model = AlexNet(num_classes=len(data_preparation.classes))
elif model_name == "MobileNetV1":
    model = mobileNetV1(input_shape=image_depth, output_shape=len(data_preparation.classes))
else: 
    print(f"{model_name} is not a model that we have")
    exit()

csvColNames = ['model', 'epochs', 'lr', 'batchSize', 'MACs', 'modelPerams', 'trainTimeStr','trainTimeS', 'trainLoss', 'testLoss', 'testAcc']  #trainLoss, testLoss, 
with open('../output/runSumary.csv', 'w', newline='') as csvfile: # make a new file
    csvWriter = csv.DictWriter(csvfile, fieldnames=csvColNames)
    csvWriter.writeheader()

nEpochs_list = [1]
lr_list = [0.01]
bSize_list = [16]

runStartTime = timer()
for epochs in nEpochs_list:
  training_config['epochs'] = epochs
  for batchSize in bSize_list:
    training_config['batch_size'] = batchSize
    for lr in lr_list:
        training_config['learning_rate'] = lr

        #opt= optimiser(model, shalPruneRatio=0.0, midPruneRatio=0.0, deepPruneRatio=0.5)
        #model = opt.prune() 

        training_config['model'] = model
        training_config['training_data'] = train_data
        training_config['testing_data'] = test_data

        hasPrinttedInfo = True
        testImage, testLabel = train_data[0]
        testImSz = len(testImage)

        modelSum = summary(model=model, 
            input_size=(1, image_depth, image_width, image_height), # make sure this is "input_size", not "input_shape"
            # col_names=["input_size"], # uncomment for smaller output
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
        saveInfo(model=model, thingOne=modelSum, fileName='_sumary.txt')
        saveInfo(model=model, thingOne=model, fileName='_modelInfo.txt')
        MACs, mPerams = countOperations(model=model, image=testImage)

        """
        Training and Testing
        """
        trainer = Trainer(**training_config)

        startTime = timer()
        trainLoss = trainer.train()
        trainTime = timer() - startTime
        trainTimeStr = timeStrFromS(trainTime)
        #testLoss = trainer.test(data_preparation.classes)
        testLoss, testAcc = trainer.test(data_preparation.classes) # Unit test reqires singletion 

        with open('../output/runSumary.csv', 'a', newline='') as csvfile: 
            csvWriter = csv.DictWriter(csvfile, fieldnames=csvColNames)
            csvWriter.writerow({'model': model_name, 
                                'epochs':epochs, 'lr':lr, 'batchSize':batchSize, 'MACs':MACs, 'modelPerams':mPerams, 
                                'trainTimeStr':trainTimeStr, 'trainTimeS':trainTime, 'trainLoss':trainLoss, 'testLoss':testLoss, 'testAcc':'na'})
                                #'trainTimeStr':trainTimeStr, 'trainTimeS':trainTime, 'trainLoss':trainLoss, 'testLoss':testLoss, 'testAcc':testAcc})
       

runEndTime = timer()
runTime = timer() - runStartTime
runTimeStr = timeStrFromS(runTime)
print(f"Total run time: {runTime} seconds, {runTimeStr}")

# Save the model
from pathlib import Path
modelDir = "../output"
modelPath = Path(modelDir)
modelPath.mkdir(parents=True, exist_ok=True)
name = model.__class__.__name__

fileName = name+".pth"
modelFile = modelPath/fileName
print(f"Saving PyTorch Model State Dict: {modelFile}")
torch.save(obj=model.state_dict(), f=modelFile)