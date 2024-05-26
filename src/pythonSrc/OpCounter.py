from thop import profile, clever_format
import sys
#from Model import leNetV5

def countOperations(model, image, showLayers:bool = False):
        image = image.unsqueeze(0)
        MACs, parameters, layerData = profile(model=model, inputs=(image,) , ret_layer_info=True)
        #MACs, parameters = clever_format([MACs, parameters], "%.1f")
        #print('Model: ', model.__class__.__name__, sep='', end='', flush=True) wtf don't this work?

        #MACs, parameters = profile(model, inputs=(input,), custom_ops={logisticRegression: model.count_your_model})
        print(70*"-")
        print(f"  **** Model: {model.__class__.__name__}, params, MACs: {parameters}, {MACs}\n")
        print(70*"-")
        if(showLayers):
            opCountLayRes(0, layerData['pretrain_model'])
            print(70*"-")

        #return f"Model: {model.__class__.__name__}, MACs, params: {MACs}, {parameters}"
        return MACs, parameters

def opCountLayRes(layerNum, item):
    #print(f"{layerNum*' '}")
    if(isinstance(item, dict)):
        for key, dicItem in item.items():
            print(f"{layerNum*' '}%{key}", end='')
            opCountLayRes(layerNum+1, dicItem)
    else:
        print(f"{layerNum*' '}-MACs: {item[0]}, Params: {item[1]}")
        opCountLayRes(layerNum+1, item[2])


def saveInfo(model, thingOne, fileName):
    modelName = model.__class__.__name__
    print(f"Saveing : {modelName}{fileName}")
    fileName = open('../output/'+modelName+fileName, 'w')
    stdOut = sys.stdout
    sys.stdout = fileName
    print(modelName)
    print(thingOne)
    sys.stdout = stdOut
    fileName.close()

def timeStrFromS(sec):
    from math import floor

    min = 0
    hur = 0
    if sec > 60:
        min  = floor(sec/60)
        sec = sec - (min*60)
    if min > 60:
        hur = floor(min/60)
        min = min - (hur*60)

    sec = floor(sec)
    return f"{hur}:{min:02d}:{sec:02d}"
