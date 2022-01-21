"""
Implementation of the U-Net model architecture.

Copyright (c), Aarón Espasandín - All Rights Reserved

This source code is licensed under the BSD 3-Clause license found in the
LICENSE file in the root directory of this source tree:
https://github.com/aaronespasa/Wall-Painting/blob/main/LICENSE
"""
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConvolution, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """UNet model implementation for image segmentation """
    def __init__(self, in_channels:int = 3, out_channels:int = 2, features:list = [64, 128, 256, 512]):
        """
        Args:
            in_channels (int): Number of channels of each input image.
                                    - For RGB images: in_channels = 3
                                    - For B&W images: in_channels = 1
            out_channels (int): Number of different classes our model is segmenting. 
                                    - For binary segmentation (background & car): out_channels = 2
            features (int list): Number of features ("out_channels") that will have 
                                 each Double Convolution.
                                    - Example: features = [64, 128, 256]

        """
        super(UNet, self).__init__()
        # Main attributes
        self.in_channels = in_channels
        self.NUM_OF_CLASSES = out_channels
        self.features = features

        # Main pieces of the UNet model
        self.decreasing_layers = self.create_decreasing_layers()
        self.bottleneck = self.create_bottleneck_layers()
        self.increasing_layers = self.create_increasing_layers()
        self.last_conv_layer = nn.Conv2d(in_channels=features[0], out_channels=self.NUM_OF_CLASSES, kernel_size=1)

        # Other layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def create_decreasing_layers(self, moduleList = nn.ModuleList()):
        """"Create the decreasing part of the UNet model architecture"""
        in_channels = self.in_channels

        for feature in self.features:
            moduleList.append(DoubleConvolution(in_channels, feature))
            # Actual num. of input channels == previous num. of output channels
            in_channels = feature

        return moduleList
    
    def create_bottleneck_layers(self):
        """"Create the bottleneck of the UNet model architecture"""
        return DoubleConvolution(
                    in_channels=self.features[-1],
                    out_channels=self.features[-1]*2
                )

    def create_increasing_layers(self, moduleList = nn.ModuleList()):
        """Create the increasing part of the UNet model architecture"""
        for feature in reversed(self.features):
            moduleList.append(
                nn.ConvTranspose2d(in_channels=feature*2, out_channels=feature, kernel_size=2, stride=2)
            )
            # After "going up" we have to make 2 convolutions
            moduleList.append(
                DoubleConvolution(in_channels=feature*2, out_channels=feature)
            )

        return moduleList
    
    def forward(self, x):
        # the skip connections list stores the last layer of each double convolution
        skip_connections = []
        
        ### DECREASING PART ###
        for decreasing_layer in self.decreasing_layers:
            x = decreasing_layer(x) # input x is processed by the code of the 'decreasing_layer'
            skip_connections.append(x)
            x = self.pool(x)

        skip_connections = skip_connections[::-1] # we want with the last skip connection

        ###   BOTTLENECK    ###
        x = self.bottleneck(x)

        ### INCREASING PART ###
        for i in range(0, len(self.increasing_layers), 2):
            # Apply the conv2d transpose
            x = self.increasing_layers[i](x)

            # We're taking steps of two, so we have to divide the index by 2
            skip_connection = skip_connections[i // 2]

            if x.shape != skip_connection.shape:
                # Example: x.shape = 30x30 and skip_connection.shape = 31 x 31
                #          That means, we lost a pixel and now we can't concatenate them.
                #          Therefore, we have to make them equal.
                x = TF.resize(x, size=skip_connection.shape[2:])

            # The concatenation of the skip connection layer + input layer -> new input layer
            concatenate_skip = torch.cat((skip_connection, x), dim=1)

            # Double convolution of the skip connection layer + input layer
            x = self.increasing_layers[i+1](concatenate_skip)

        ### FINAL CONVOLUTION ###
        return self.last_conv_layer(x)

def test():
    x = torch.rand((3, 1, 160, 160))
    model = UNet(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == '__main__':
    test()