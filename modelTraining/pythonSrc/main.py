print(f"INIT: SQB, V0.1Alpha ")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import logging
from DataPreparation import DataPreparation
from Model import leNetV5, MobileNetV3, AlexNet
from ConfigParser import ConfigParser
from Trainer import Trainer
from Logger import Logger
from Plot import Plot

from OpCounter import countOperations, saveInfo, timeStrFromS
from torchinfo import summary
import csv
from timeit import default_timer as timer

## Optimise
from optimize import optimiser

## For save

from saveModel import saveModel

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

print(f"INIT: Set Loger")
stdout_logger = logging.getLogger('STDOUT')
sys.stdout = Logger(stdout_logger, logging.DEBUG)
config = ConfigParser(os.path.join(os.getcwd(), 'config.yaml'))
meta_config = config.get_config()["meta"]
model_config = config.get_config()["model"]

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
#model_config = config.get_config()["model"]

csvColNames = ['model', 'epochs', 'lr', 'hiddenNer', 'batchSize', 'MACs', 'modelPerams', 'trainTimeStr','trainTimeS', 'trainLoss', 'testLoss', 'testAcc']  #trainLoss, testLoss, 
with open('../output/runSumary.csv', 'w', newline='') as csvfile: # make a new file
    csvWriter = csv.DictWriter(csvfile, fieldnames=csvColNames)
    csvWriter.writeheader()

model_list = ["leNetV5"]
#model_list = ["MobileNetV3"] # Cant Prune
#model_list = ["AlexNet"]   # way too big
#model_list = ["leNetV5", "TinyVGG", "logisticRegression", "multilayerPerceptron"]
nEpochs_list = [1]
#nEpochs_list = [1, 10, 20]
NumHidden_list = [30]
#NumHidden_list = [15, 30, 60]
lr_list = [0.01]
#lr_list = [0.1, 0.01]
bSize_list = [5]
#bSize_list = [16, 32, 64]

runStartTime = timer()
for model_name in model_list:           # Set in model call
 nHidden_list = NumHidden_list
 for epochs in nEpochs_list:            # Set in training conf
  training_config['epochs'] = epochs
  for hiddenNerons in nHidden_list:     # Set in model call
   for batchSize in bSize_list:
    training_config['batch_size'] = batchSize
    for lr in lr_list:
        training_config['learning_rate'] = lr
        if model_name == "leNetV5":
            model = leNetV5(input_shape=image_depth,hidden_units=hiddenNerons,output_shape=len(data_preparation.classes))
        elif model_name == "MobileNetV3":
            model = MobileNetV3(num_classes=len(data_preparation.classes))
        elif model_name == "AlexNet":
            model = AlexNet(num_classes=len(data_preparation.classes))
        else: 
            print(f"{model_name} is not a model that we have")
            exit()

        #opt= optimiser(model, shalPruneRatio=0.0, midPruneRatio=0.0, deepPruneRatio=0.5)
        #model = opt.prune() 

        training_config['model'] = model
        training_config['training_data'] = train_data
        training_config['testing_data'] = test_data

        ####
        # Josh adds the op counter here
        ####
        hasPrinttedInfo = True
        testImage, testLabel = train_data[0]
        testImSz = len(testImage)

        modelSum = summary(model=model, 
            input_size=(32, image_depth, image_width, image_height), # make sure this is "input_size", not "input_shape"
            # col_names=["input_size"], # uncomment for smaller output
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
        saveInfo(model=model, thingOne=modelSum, fileName='_sumary.txt')
        saveInfo(model=model, thingOne=model, fileName='_modelInfo.txt')
    
        MACs, mPerams = countOperations(model=model, image=testImage)
        #exit() # Stop here for now
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
                                'epochs':epochs, 'lr':lr, 'hiddenNer':hiddenNerons, 'batchSize':batchSize, 'MACs':MACs, 'modelPerams':mPerams, 
                                'trainTimeStr':trainTimeStr, 'trainTimeS':trainTime, 'trainLoss':trainLoss, 'testLoss':testLoss, 'testAcc':'na'})
                                #'trainTimeStr':trainTimeStr, 'trainTimeS':trainTime, 'trainLoss':trainLoss, 'testLoss':testLoss, 'testAcc':testAcc})
       

runEndTime = timer()
runTime = timer() - runStartTime
runTimeStr = timeStrFromS(runTime)
print(f"Total run time: {runTime} seconds, {runTimeStr}")

saveModel(model=model, imgLayers=image_depth, imgWidth=image_width, imgHeight=image_height)
#exit()
"""
Plotting

"""
# Figure out how to plot RGB565
#Plot().plot_prediction(training_config['model'],test_data,os.path.join(os.getcwd(), "../output/predictions.png") )