import tensorflow as tf
import numpy as np


#TFLITE_FILE_PATH = '../models/leNetV5_mod/16_inputFloat/leNetV5.tflite'
#TFLITE_FILE_PATH = '../models/leNetV5_mod/17_onnx2tf/leNetV5_tf/leNetV5_dynamic_range_quant.tflite'
#TFLITE_FILE_PATH = '../models/leNetV5_mod/18_onnx2f-TFLiteConverter/leNetV5.tflite'
TFLITE_FILE_PATH = '../models/leNetV5_mod/20_modifyedOnnx2TF/leNetV5.tflite'
#TFLITE_FILE_PATH = '../output/leNetV5_tf/leNetV5_dynamic_range_quant.tflite'
#TFLITE_FILE_PATH = '../output/leNetV5.tflite'


# Load the TFLite model in TFLite Interpreter
interpreter = tf.lite.Interpreter(model_path=TFLITE_FILE_PATH)
interpreter.allocate_tensors()


# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_type = input_details[0]['dtype']
input_shape = input_details[0]['shape']
#print(f"Input shape: {input_shape}, type: {input_type}")

qPerams = input_details[0]['quantization_parameters']['scales']
#input_zPoint = input_details[0]['zero_points']
print(f"Quantize info: {qPerams}")
#exit()

# Test the model on random input data.
#input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)

# Use our dataloader
from torch.utils.data import DataLoader
from DataPreparation import DataPreparation
from ConfigParser import ConfigParser
import os

config = ConfigParser(os.path.join(os.getcwd(), 'config.yaml'))
meta_config = config.get_config()["meta"]
data_preparation = DataPreparation(meta_config["data_save_path"])
train_data, test_data = data_preparation.get_data(displayImages=False)
#model_labels = {0: 'Nobody', 1: 'Bird', 2: 'Squi'}
dataloader = DataLoader(train_data,batch_size=25,shuffle=False)

for images, labels in dataloader:
    for i in range(len(images)-1):  
                                #plt.imshow(images[i].permute(1, 2,0))
            input_data = np.asarray(images[i].permute(1,2,0), dtype=np.float32)
            #input_data = np.asarray(images[i], dtype=np.uint8)
            input_data = np.expand_dims(input_data, axis=0) 

            #print(f"images[{i}] size: {images[i].shape}")
            #print(f"numpy_array size: {input_data.shape}")
            #print(f"image: {images[i]}")
            #print(f"image: {input_data[0, 0, 0:4, 0]}") # make sure the images are different

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
            output_data = interpreter.get_tensor(output_details[0]['index'])
            print(output_data)
