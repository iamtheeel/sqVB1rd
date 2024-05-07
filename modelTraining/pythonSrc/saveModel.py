###
# saveModel.py
#
# Joshua Mehlman
# ENGR 859 Spring 2024
# Term Project
#
# Squirrl Or Bird Detector
#
# Saving the model as a tensor flow lite:
#   https://www.learnpytorch.io/01_pytorch_workflow/#loading-a-saved-pytorch-models-state_dict
#   Save as PyTorch Model State
#   PyTorch     --> Onnx
#   Onnx        --> TensorFlow
#   TensorFlow  --> TensorFlow Lite
#   https://www.tensorflow.org/lite/microcontrollers/build_convert#operation_support
#   TF Lite     --> C header
#**
from pathlib import Path
import numpy as np
import tensorflow
import torch

# For representative_dataset
import torchvision.transforms as trans
from torchvision import datasets


## add numbers
# From MIC
def representative_dataset():
    """
    Prepare representive data for quantization activation layer
    """

    ### Make a function
    imgW = 96
    imgH = 96
    camWidth = 320 #QQVGA
    camHeight = 240

    data = np.load("representive_data.npy",allow_pickle=True) #Created in DataPreperation
    print(f"Data size: {data.shape}") # Batch size: 87, 2, 96, 96
    for i in range(len(data)):
        temp_data = data[i]
        temp_data = temp_data.reshape(1,2,imgW,imgH)# 2 is from RGB565, 2 bytes for 3 colors
        yield [temp_data.astype(np.float32)]

def saveModel(model, imgLayers, imgWidth, imgHeight, mean, std):
    modelDir = "../output"
    modelPath = Path(modelDir)
    modelPath.mkdir(parents=True, exist_ok=True)

    name = model.__class__.__name__

    fileName = name+".pth"
    modelFile = modelPath/fileName

    # Save the PyTorch Model
    '''
    print(f"Saving PyTorch Model State Dict: {modelFile}")
    torch.save(obj=model.state_dict(), f=modelFile)

    '''
    # or open it
    model.load_state_dict(torch.load(modelFile), strict=False)
    #model.load_state_dict(torch.load(modelFile)["state_dict"])

    # Save and Load as Onnx
    onnxFileName = name+".onnx"
    onnxFile = modelPath/onnxFileName
    print(f"Saveing to Onnx: {onnxFile}")
    input_shape = (1, imgLayers, imgWidth, imgHeight)
    torch.onnx.export(model, torch.randn(input_shape), onnxFile, verbose=True,
                      input_names=["input"],
                      output_names=["output"])

    '''
    import onnx_tf
    import onnx
    # Convert to Tensor Flow using onnx_tf
    tfFileName = name+"_tf"
    tfFile = modelDir+"/"+name+"_tf"
    print(f"Saveing to Tensor FLow: {tfFile} with onnx_tf")
    onnx_model = onnx.load(onnxFile)  # Load the file we just saved
    tf_rep = onnx_tf.backend.prepare(onnx_model)
    tf_rep.export_graph(tfFile) # Export as tf


    # Convert to TF Lite
    tfLiteFileName = name+".tflite"
    tfLiteFile = modelDir+"/"+tfLiteFileName
    print(f"Saveing to Tensor FLow Lite: {tfLiteFile}")
    converter = tensorflow.lite.TFLiteConverter.from_saved_model(tfFile)
    # Some settings for the tfLite from MIC
    converter.optimizations = [tensorflow.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tensorflow.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tensorflow.float32  
    #converter.inference_input_type = tensorflow.uint8  # this is giving headach
    converter.inference_output_type = tensorflow.float32

    tflite_model = converter.convert()
    open(tfLiteFile, 'wb').write(tflite_model)
    tflite_modelSize = len(tflite_model)

    print(f"TF Lite model size: {tflite_modelSize} bytes")
    '''

    # Convert to Tensor Flow using onnx2tf - https://github.com/PINTO0309/onnx2tf
    #onnx2tf -i leNetV5.onnx -oiqt -cind representive_data.npy
    #onnx2tf -i leNetV5.onnx -oiqt -cind "input" representive_data.npy "[[[[0.485,0.456,0.406]]]]" "[[[[0.229,0.224,0.225]]]]"

    #calibrationData=[ ["input", "data/calibdata.npy", [[[0.485,0.456,0.406]]], [[[0.229,0.224,0.225]]]] ]
    calibrationData=[ ["input", "../output/representive_data.npy", mean, std] ]

    #overwrite_input_shape = Union[List[str], NoneType] = None,

    import onnx2tf
    tfFile = modelDir+"/"+name+"_tf"
    print(f"Saveing to Tensor FLow: {tfFile} with onnx2tf")
    onnx2tf.convert(
        input_onnx_file_path=onnxFile,
        output_folder_path=tfFile,
        output_integer_quantized_tflite=True,
        #overwrite_input_shape = input_shape,
        custom_input_op_name_np_data_path=calibrationData,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
    )


    ## The final step os 
    #  TF Lite     --> C header
    # xxd -i leNetV5.tflite > leNetV5.h

def generateMeanStd():
    from torch.utils.data import DataLoader
    from DataPreparation import DataPreparation
    from ConfigParser import ConfigParser
    import os

    config = ConfigParser(os.path.join(os.getcwd(), 'config.yaml'))
    meta_config = config.get_config()["meta"]

    print(f"INIT: Get Images")
    data_preparation = DataPreparation(meta_config["data_save_path"])
    train_data, test_data = data_preparation.get_data(displayImages=False)
    dataloader = DataLoader(test_data,batch_size=10,shuffle=False)

    runningSum = 0
    nImages = 0
    for images, labels in dataloader:
        for i in range(len(images)-1):  
            #print(f"img_data shape: {img_data.shape}")
            nImages += 1
            img_data = np.asarray(images[i], dtype=np.float32)
            runningSum += img_data
            #print(f"running sum: {runningSum[0, 0:4, 0]}") # make sure the images are different

    mean  = runningSum/nImages
    var = 0
    for images, labels in dataloader:
        for i in range(len(images)-1):  
            #print(f"img_data shape: {img_data.shape}")
            img_data = np.asarray(images[i], dtype=np.float32)
            img_data -= mean
            var += np.square(img_data)

    std = np.sqrt(var/nImages)
    #print(f"There are {nImages} images")
    #print(f"mean shape: {mean.shape}")
    #print(f"std shape: {std.shape}")
    #print(f"mean: {mean[0, 0:4, 0]}") # make sure the images are different
    #print(f"std: {std[0, 0:4, 0]}") # make sure the images are different

    return mean, std

mean, std = generateMeanStd()
#exit()
from Model import leNetV5
hiddenNerons = 30
image_depth = 2
print(f"Export presaved model")
model = leNetV5(input_shape=image_depth,hidden_units=hiddenNerons,output_shape=3)
saveModel(model, image_depth, 96, 96, mean, std)