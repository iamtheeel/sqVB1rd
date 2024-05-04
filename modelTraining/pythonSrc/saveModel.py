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


## add numbers
def representative_dataset():
    """
    Prepare representive data for quantization activation layer
    """
    imgW = 160
    imgH = 120
    #data = np.load("representive_data.npy",allow_pickle=True)#("representive_data.npy",allow_pickle=True)#("Representive_data.npy") ??????????????????
    for i in range(2):
        #temp_data = data[i]
        temp_data = np.random.rand(1, 3*imgW*imgH)
        temp_data = temp_data.reshape(1,3,imgW,imgH)#temp_data = temp_data.reshape(1,globalSize,globalSize,3)
        yield [temp_data.astype(np.float32)]

def saveModel(model, name, imgLayers, imgWidth, imgHeight):
    modelDir = "../models"
    modelPath = Path(modelDir)
    modelPath.mkdir(parents=True, exist_ok=True)

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
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tensorflow.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tensorflow.int8  # or tf.uint8
    converter.inference_output_type = tensorflow.float32

    tflite_model = converter.convert()
    open(tfLiteFile, 'wb').write(tflite_model)
    tflite_modelSize = len(tflite_model)

    print(f"TF Lite model size: {tflite_modelSize} bytes")


    ## The final step os 
    #  TF Lite     --> C header
    # xxd -i converted_model.tflite > model_data.h