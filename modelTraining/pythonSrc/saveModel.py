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
import onnx_tf
import onnx

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

    data_path = Path("../data")
    data_dir =data_path / "train"
    saveTrans = trans.Compose([
                trans.Resize((camWidth, camHeight)),
                trans.CenterCrop(imgW),
                trans.ToTensor()
        ])

    #saveDataSet = datasets.ImageFolder(root=data_dir, transform=saveTrans)
    #input_fp32_list = []
    #startImg = 0
    #dataLen = len(saveDataSet)
    #for i in range(startImg, dataLen):
    #    input_fp32 = saveDataSet.__getitem__(i)[0]
    #    yield [input_fp32]
    #    #input_fp32_list.append(input_fp32)

    data = np.load("representive_data.npy",allow_pickle=True) #Created in DataPreperation
    print(f"Data size: {data.shape}") # Batch size: 87, 2, 96, 96
    for i in range(len(data)):
        temp_data = data[i]
        temp_data = temp_data.reshape(1,2,imgW,imgH)# 2 is from RGB565, 2 bytes for 3 colors
        yield [temp_data.astype(np.float32)]

def saveModel(model, imgLayers, imgWidth, imgHeight):
    modelDir = "../output"
    modelPath = Path(modelDir)
    modelPath.mkdir(parents=True, exist_ok=True)

    name = model.__class__.__name__


    # Save the PyTorch Model
    fileName = name+".pth"
    modelFile = modelPath/fileName
    print(f"Saving PyTorch Model State Dict: {modelFile}")
    torch.save(obj=model.state_dict(), f=modelFile)

    # Save and Load as Onnx
    onnxFileName = name+".onnx"
    onnxFile = modelPath/onnxFileName
    print(f"Saveing to Onnx: {onnxFile}")
    input_shape = (1, imgLayers, imgWidth, imgHeight)
    torch.onnx.export(model, torch.randn(input_shape), onnxFile, verbose=True,input_names=["input"],output_names=["output"])

    # Convert to Tensor Flow
    tfFileName = name+"_tf"
    tfFile = modelDir+"/"+tfFileName
    print(f"Saveing to Tensor FLow: {tfFile}")
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
    #converter.representative_dataset = representative_dataset
    #converter.target_spec.supported_ops = [tensorflow.lite.OpsSet.TFLITE_BUILTINS_INT8]
    #converter.inference_input_type = tensorflow.int8  # or tf.uint8
    #converter.inference_output_type = tensorflow.float32

    tflite_model = converter.convert()
    open(tfLiteFile, 'wb').write(tflite_model)
    tflite_modelSize = len(tflite_model)

    print(f"TF Lite model size: {tflite_modelSize} bytes")


    ## The final step os 
    #  TF Lite     --> C header
    # xxd -i leNetV5.tflite > leNetV5.h
