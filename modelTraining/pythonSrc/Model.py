import torch
from torch import nn

from thop import profile, clever_format

#
# Note we will be going to TF Lite so:
# https://www.tensorflow.org/lite/guide/ops_compatibility
#

from torchvision.models import AlexNet, AlexNet_Weights
class AlexNet(nn.Module):
    def __init__(self,num_classes=3):
        super(AlexNet,self).__init__()
        print(f"Init AlexNet:")
        torch.manual_seed(420)
        weights = AlexNet_Weights.DEFAULT
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', weights=weights)
        print(self.model.eval())

		# Freeze the model parameters
        for param in self.model.features.parameters(): 
            param.requires_grad = False 
            # Make a new classifyer
            # before:                                                          Input shape          Output shape
            #└─Sequential (classifier)           [32, 9216]           [32, 1000]           --                   True
            #└─Dropout (0)                  [32, 9216]           [32, 9216]           --                   --
            #└─Linear (1)                   [32, 9216]           [32, 4096]           37,752,832           True
            #└─ReLU (2)                     [32, 4096]           [32, 4096]           --                   --
            #└─Dropout (3)                  [32, 4096]           [32, 4096]           --                   --
            #└─Linear (4)                   [32, 4096]           [32, 4096]           16,781,312           True
            #└─ReLU (5)                     [32, 4096]           [32, 4096]           --                   --
            #└─Linear (6)                   [32, 4096]           [32, 1000]           4,097,000            True
            self.model.classifier = nn.Sequential(
						nn.Linear(9216, num_classes)) #Initial: In 32x576, Out 32x1000


    def forward(self,x):
        return self.model(x)   


from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
class MobileNetV3(nn.Module):
    def __init__(self,num_classes=3):
        super(MobileNetV3,self).__init__()
		#TODO: Load the pretrained model 
        torch.manual_seed(420)
        weights = MobileNet_V3_Small_Weights.DEFAULT
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v3_small', weights=weights)

        # For optimization, from HW6
        self.conv2d_layers = [0,3,7,10]
        self.bn_layers = [1,4,8,11]
        self.shaLayEnd = 2
        self.midLayEnd = 3
		
		# Freeze the model parameters
        for param in self.model.features.parameters(): 
            param.requires_grad = False 
            # Make a new classifyer
            # before:                                                          Input shape          Output shape
            #│    └─Sequential (classifier)                                    [32, 576]            [32, 1000]           --                   True
            #│    │    └─Linear (0)                                            [32, 576]            [32, 1024]           590,848              True
            #│    │    └─Hardswish (1)                                         [32, 1024]           [32, 1024]           --                   --
            #│    │    └─Dropout (2)                                           [32, 1024]           [32, 1024]           --                   --
            #│    │    └─Linear (3)                                            [32, 1024]           [32, 1000]           1,025,000            True

            self.model.classifier = nn.Sequential(
						#nn.Linear(576, num_classes)) #Initial: In 32x576, Out 32x1000
                        nn.Linear(576, 1024),
                        nn.Hardswish(),
                        nn.Dropout(0.5),
						nn.Linear(1024, num_classes)) #Initial: In 32x576, Out 32x1000
            

    def forward(self,x):
        return self.model(x)   


    
class leNetV5(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        """
        LeNet-5:
            Convolution kernal = 5x5, stride=1, tanh
            Pooling, kernal = 2x2, stride 2, tanh

            Convolution kernal = 5x5, stride=1, tanh
            Pooling, kernal = 2x2, stride 2, tanh

            ## Fully connected
            Convolution kernal = 5x5, stride=1, tanh

            FC, tanh
            FC, softmax
        """
        super().__init__() 

        # For optimization, from HW6
        self.conv2d_layers = [0,4,7]
        self.bn_layers = [1,5,8]
        self.shaLayEnd = 1
        self.midLayEnd = 2

        conv_1Lay = 12#
        conv_2Lay = 24
        torch.manual_seed(420)


        #self.layer = nn.Sequential(
        self.features = nn.Sequential(
                                        nn.Conv2d(in_channels=input_shape, out_channels=conv_1Lay, kernel_size=3, stride=1, padding=2),
                                        nn.BatchNorm2d(conv_1Lay),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2, stride=2),

                                        nn.Conv2d(in_channels=conv_1Lay, out_channels=conv_2Lay, kernel_size=3, stride=1, padding=0),
                                        nn.BatchNorm2d(conv_2Lay),
                                        nn.ReLU(),

                                        nn.Conv2d(in_channels=conv_2Lay, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
                                        nn.BatchNorm2d(hidden_units),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2, stride=2)
                                     )

        self.dropout = nn.Dropout(0.25)
        linConnections = 32
        self.linear = nn.Sequential( nn.Flatten(),
                                      #nn.Linear(605*conv_2Lay, linConnections), # conv_1Lay = 6, conv_2Lay = 16
                                      nn.Linear(605*conv_2Lay, linConnections), # conv_1Lay = 6, conv_2Lay = 16
                                      nn.ReLU(),
                                      nn.Linear(linConnections, linConnections), # conv_1Lay = 6, conv_2Lay = 16
                                      nn.ReLU()
                                      )  

        self.clasifyer = nn.Sequential(nn.Linear(linConnections, output_shape)  )


    def forward(self, x: torch.Tensor):
        #x = self.layer(x)
        x = self.features(x)
        #x = self.hiddenLayer(x)

        #x = self.dropout(x)
        x = self.linear(x)
        x = self.clasifyer(x)

        return x 

class mobileNetV1(nn.Module):
    def __init__(self, input_shape: int, output_shape: int):
        super().__init__() 
        outCh = 32
        self.input = nn.Sequential(
                            nn.Conv2d(in_channels=input_shape, out_channels=outCh, kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(outCh),
                            nn.ReLU(),
        )
        inCh = 32
        outCh = 64
        self.conv32_64 = nn.Sequential(
                            nn.Conv2d(in_channels=inCh, out_channels=inCh, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(inCh),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=inCh, out_channels=outCh, kernel_size=1, stride=1, padding=0, bias=False),
                            nn.BatchNorm2d(outCh),
                            nn.ReLU()
        )
        inCh = outCh
        outCh = 128
        self.conv64_128 = nn.Sequential(
                            nn.Conv2d(in_channels=inCh, out_channels=inCh, kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(inCh),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=inCh, out_channels=outCh, kernel_size=1, stride=1, padding=0, bias=False),
                            nn.BatchNorm2d(outCh),
                            nn.ReLU()
        )
        inCh = outCh
        self.conv128_128 = nn.Sequential(
                            nn.Conv2d(in_channels=inCh, out_channels=inCh, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(inCh),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=inCh, out_channels=outCh, kernel_size=1, stride=1, padding=0, bias=False),
                            nn.BatchNorm2d(outCh),
                            nn.ReLU()
        )
        outCh = 256
        self.conv128_256 = nn.Sequential(
                            nn.Conv2d(in_channels=inCh, out_channels=inCh, kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(inCh),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=inCh, out_channels=outCh, kernel_size=1, stride=1, padding=0, bias=False),
                            nn.BatchNorm2d(outCh),
                            nn.ReLU()
        )
        inCh = outCh
        self.conv256_256 = nn.Sequential(
                            nn.Conv2d(in_channels=inCh, out_channels=inCh, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(inCh),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=inCh, out_channels=outCh, kernel_size=1, stride=1, padding=0, bias=False),
                            nn.BatchNorm2d(outCh),
                            nn.ReLU()
        )
        outCh = 512
        self.conv256_512 = nn.Sequential(
                            nn.Conv2d(in_channels=inCh, out_channels=inCh, kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(inCh),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=inCh, out_channels=outCh, kernel_size=1, stride=1, padding=0, bias=False),
                            nn.BatchNorm2d(outCh),
                            nn.ReLU()
        )
        inCh = outCh
        self.conv512_512 = nn.Sequential(
                            nn.Conv2d(in_channels=inCh, out_channels=inCh, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(inCh),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=inCh, out_channels=outCh, kernel_size=1, stride=1, padding=0, bias=False),
                            nn.BatchNorm2d(outCh),
                            nn.ReLU()
        )
        outCh = 1024
        self.conv512_1024 = nn.Sequential(
                            nn.Conv2d(in_channels=inCh, out_channels=inCh, kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(inCh),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=inCh, out_channels=outCh, kernel_size=1, stride=1, padding=0, bias=False),
                            nn.BatchNorm2d(outCh),
                            nn.ReLU()
        )
        inCh = outCh
        self.conv1024_1024 = nn.Sequential(
                            # note on final layer stride: https://github.com/CellEight/PytorchMobileNet-v1
                            nn.Conv2d(in_channels=inCh, out_channels=inCh, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(inCh),
                            nn.ReLU(),
                            nn.Conv2d(in_channels=inCh, out_channels=outCh, kernel_size=1, stride=1, padding=0, bias=False),
                            nn.BatchNorm2d(outCh),
                            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool2d(1) 
        self.fullCon = nn.Linear(1024, output_shape) 

    def forward(self, x: torch.Tensor):
        x = self.input(x)
        x = self.conv32_64(x)
        x = self.conv64_128(x)
        x = self.conv128_128(x)
        x = self.conv128_256(x)
        x = self.conv256_256(x)
        x = self.conv256_512(x)
        x = self.conv512_512(x) # 5x
        x = self.conv512_512(x) # 5x
        x = self.conv512_512(x) # 5x
        x = self.conv512_512(x) # 5x
        x = self.conv512_512(x) # 5x
        x = self.conv512_1024(x) 
        x = self.conv1024_1024(x) 
        x = self.pool(x)
        x = x.view(-1,1024) # Reshape to ?x1024
        x = self.fullCon(x)

        return x 