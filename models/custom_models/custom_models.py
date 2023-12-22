from __future__ import print_function

import argparse
from curses import KEY_LEFT
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from torchvision.models import resnet18

def save_hook(module, input, output):
    setattr(module, 'output', output)

class OneLayer(nn.Module):
    def __init__(self, in_dim=100, out_dim=100):
        super(OneLayer, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim*3)
        self.fc2 = nn.Linear(out_dim*3, out_dim)
        self.act = nn.ReLU()
        self.input_dim = in_dim
        self.layer_norm_mid = nn.LayerNorm(300)
        self.layer_norm_out = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        x = self.layer_norm_out(x)
        return x

class MultiLayer(nn.Module):
    def __init__(self, in_dim=100, out_dim=100):
        super(OneLayer, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim*3)
        self.fc2 = nn.Linear(out_dim*3, out_dim*3)
        self.fc3 = nn.Linear(out_dim*3, out_dim)
        self.act = nn.ReLU()
        self.input_dim = in_dim
        self.layer_norm_mid = nn.LayerNorm(300)
        self.layer_norm_out = nn.LayerNorm(100)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.layer_norm_mid(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.layer_norm_mid(x)
        x = self.fc3(x)
        x = self.layer_norm_out(x)
        return x


class PenmenshipCNN(nn.Module):
    def __init__(self):
        super(PenmenshipCNN, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Dropout2d(p=0.1),

            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Dropout2d(p=0.1),

            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout2d(p=0.1),

            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout2d(p=0.1),
                        
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.main(x)
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        logits = self.classifier(x)
        return logits

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Dropout2d(p=0.1),

            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Dropout2d(p=0.1),

            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout2d(p=0.1),

            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout2d(p=0.1),
                        
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 10)
        )

    def forward(self, x, return_feature=False):
        x = self.main(x)
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        logits = self.classifier(x)
        if return_feature==True:
            return logits, x
        return logits
    

class CNN_cifar10(nn.Module):
    def __init__(self):
        super(CNN_cifar10, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Dropout2d(p=0.1),

            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Dropout2d(p=0.1),

            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout2d(p=0.1),

            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout2d(p=0.1),
                        
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.main(x)
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        logits = self.classifier(x)
        return logits, x


class Generator(nn.Module):
    def __init__(self, ngpu=1, nc=1, nz=100, ngf=64):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input : Z
            nn.ConvTranspose2d(nz, ngf*8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=2, bias=False),
            nn.Tanh()

        )

    def forward(self, input):
        if len(input.shape) <= 3:
            input = input.reshape(input.shape + torch.Size([1,1]))
        output = self.main(input)
        return output
    

class Generator_cifar10(nn.Module):
    def __init__(self, nz, ngf, nc, ngpu):
        super(Generator_cifar10, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.ReLU(True),
            # # state size. (ngf) x 32 x 32
            # nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input, return_only_logit=True):
        input = input.reshape(input.shape + torch.Size([1,1]))
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        if return_only_logit == True:
            return output
        else:
            return output, 0


class Discriminator(nn.Module):
    def __init__(self, nc=1, ngf=64):
        super(Discriminator, self).__init__()

    #     self.conv_1 = nn.Conv2d(nc, ngf, kernel_size=4, stride=2, padding=1, bias=True)
    #     self.conv_2 = nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2, padding=1, bias=False)
    #     self.conv_3 = nn.Conv2d(ngf*2, ngf*4, kernel_size=4, stride=2, padding=1, bias=False)
    #     self.conv_4 = nn.Conv2d(ngf*4, ngf*8, kernel_size=4, stride=2, padding=1, bias=False)
    #     self.conv_5 = nn.Conv2d(ngf*8, 1, kernel_size=3, stride=1, padding=1, bias=False)
    #     self.relu = nn.LeakyReLU(0.2)
    #     self.batch_norm_1 = nn.BatchNorm2d(ngf)
    #     self.batch_norm_2 = nn.BatchNorm2d(ngf*2)
    #     self.batch_norm_3 = nn.BatchNorm2d(ngf*4)
    #     self.batch_norm_4 = nn.BatchNorm2d(ngf*8)
    #     self.sig = nn.Sigmoid()

    # def forward(self, x):
    #     x = self.conv_1(x)
    #     x = self.batch_norm_1(x)
    #     x = self.relu(x)

    #     x = self.conv_2(x)
    #     x = self.batch_norm_2(x)
    #     x = self.relu(x)

    #     x = self.conv_3(x)
    #     x = self.batch_norm_3(x)
    #     x = self.relu(x)
        
    #     x = self.conv_4(x)
    #     x = self.batch_norm_4(x)
    #     x = self.relu(x)

    #     x = self.conv_5(x)
    #     x = self.sig(x)
    #     return x
        

        self.main = nn.Sequential(
            # input:I (1*28*28), range = [-1, 1]
            nn.Conv2d(nc, ngf, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            # feature size = ngf, 14, 14
            nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),

            # feature size = ngf*2, 7, 7
            nn.Conv2d(ngf*2, ngf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(0.2, inplace=True),

            # feature size = ngf*4, 3, 3
            nn.Conv2d(ngf*4, ngf*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # feature size = ngf*4, 1, 1
            nn.Conv2d(ngf*8, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid()
            # output size = 1, 1, 1, range=[0, 1]
        )
    
    def forward(self, x):
        output = self.main(x)
        return output
    
class Discriminator_cifar10(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(Discriminator_cifar10, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            # nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, mean=1.0, std=0.02)
        nn.init.zeros_(m.bias)



class AE(nn.Module):
	def __init__(self):
		super().__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
			nn.ReLU(True),
			nn.Conv2d(16, 16, kernel_size=2, stride=2, padding=0),
			nn.ReLU(True),
			nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
			nn.ReLU(True),
			nn.Conv2d(16, 16, kernel_size=2, stride=2, padding=0),
			nn.ReLU(True)
		)

		self.decoder = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='bilinear'),
			nn.ReLU(True),
			nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
			nn.ReLU(True),

			nn.Upsample(scale_factor=2, mode='bilinear'),
			nn.ReLU(True),
			nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
		)

	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		x = torch.sigmoid(x)
		return x



class AE_cifar10(nn.Module):
    def __init__(self):
        super(AE_cifar10, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(12),
            nn.Conv2d(12, 24, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 48, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(48),           
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

def get_configs(arch='resnet50'):

    # True or False means wether to use BottleNeck

    if arch == 'resnet18':
        return [2, 2, 2, 2], False
    elif arch == 'resnet34':
        return [3, 4, 6, 3], False
    elif arch == 'resnet50':
        return [3, 4, 6, 3], True
    elif arch == 'resnet101':
        return [3, 4, 23, 3], True
    elif arch == 'resnet152':
        return [3, 8, 36, 3], True
    else:
        raise ValueError("Undefined model")

class ResNetAutoEncoder(nn.Module):

    def __init__(self, configs, bottleneck):

        super(ResNetAutoEncoder, self).__init__()

        self.encoder = ResNetEncoder(configs=configs,       bottleneck=bottleneck)
        self.decoder = ResNetDecoder(configs=configs[::-1], bottleneck=bottleneck)
    
    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x

class ResNet(nn.Module):

    def __init__(self, configs, bottleneck=False, num_classes=1000):
        super(ResNet, self).__init__()

        self.encoder = ResNetEncoder(configs, bottleneck)

        self.avpool = nn.AdaptiveAvgPool2d((1,1))

        if bottleneck:
            self.fc = nn.Linear(in_features=2048, out_features=num_classes)
        else:
            self.fc = nn.Linear(in_features=512, out_features=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):

        x = self.encoder(x)

        x = self.avpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


class ResNetEncoder(nn.Module):

    def __init__(self, configs, bottleneck=False):
        super(ResNetEncoder, self).__init__()
        if len(configs) != 4:
            raise ValueError("Only 4 layers can be configued")

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        )

        if bottleneck:

            self.conv2 = EncoderBottleneckBlock(in_channels=64,   hidden_channels=64,  up_channels=256,  layers=configs[0], downsample_method="pool")
            self.conv3 = EncoderBottleneckBlock(in_channels=256,  hidden_channels=128, up_channels=512,  layers=configs[1], downsample_method="conv")
            self.conv4 = EncoderBottleneckBlock(in_channels=512,  hidden_channels=256, up_channels=1024, layers=configs[2], downsample_method="conv")
            self.conv5 = EncoderBottleneckBlock(in_channels=1024, hidden_channels=512, up_channels=2048, layers=configs[3], downsample_method="conv")

        else:

            self.conv2 = EncoderResidualBlock(in_channels=64,  hidden_channels=64,  layers=configs[0], downsample_method="pool")
            self.conv3 = EncoderResidualBlock(in_channels=64,  hidden_channels=128, layers=configs[1], downsample_method="conv")
            self.conv4 = EncoderResidualBlock(in_channels=128, hidden_channels=256, layers=configs[2], downsample_method="conv")
            self.conv5 = EncoderResidualBlock(in_channels=256, hidden_channels=512, layers=configs[3], downsample_method="conv")

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x

class ResNetDecoder(nn.Module):

    def __init__(self, configs, bottleneck=False):
        super(ResNetDecoder, self).__init__()

        if len(configs) != 4:
            raise ValueError("Only 4 layers can be configued")

        if bottleneck:

            self.conv1 = DecoderBottleneckBlock(in_channels=2048, hidden_channels=512, down_channels=1024, layers=configs[0])
            self.conv2 = DecoderBottleneckBlock(in_channels=1024, hidden_channels=256, down_channels=512,  layers=configs[1])
            self.conv3 = DecoderBottleneckBlock(in_channels=512,  hidden_channels=128, down_channels=256,  layers=configs[2])
            self.conv4 = DecoderBottleneckBlock(in_channels=256,  hidden_channels=64,  down_channels=64,   layers=configs[3])


        else:

            self.conv1 = DecoderResidualBlock(hidden_channels=512, output_channels=256, layers=configs[0])
            self.conv2 = DecoderResidualBlock(hidden_channels=256, output_channels=128, layers=configs[1])
            self.conv3 = DecoderResidualBlock(hidden_channels=128, output_channels=64,  layers=configs[2])
            self.conv4 = DecoderResidualBlock(hidden_channels=64,  output_channels=64,  layers=configs[3])

        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False),
        )

        self.gate = nn.Sigmoid()

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gate(x)

        return x

class EncoderResidualBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, layers, downsample_method="conv"):
        super(EncoderResidualBlock, self).__init__()

        if downsample_method == "conv":

            for i in range(layers):

                if i == 0:
                    layer = EncoderResidualLayer(in_channels=in_channels, hidden_channels=hidden_channels, downsample=True)
                else:
                    layer = EncoderResidualLayer(in_channels=hidden_channels, hidden_channels=hidden_channels, downsample=False)
                
                self.add_module('%02d EncoderLayer' % i, layer)
        
        elif downsample_method == "pool":

            maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.add_module('00 MaxPooling', maxpool)

            for i in range(layers):

                if i == 0:
                    layer = EncoderResidualLayer(in_channels=in_channels, hidden_channels=hidden_channels, downsample=False)
                else:
                    layer = EncoderResidualLayer(in_channels=hidden_channels, hidden_channels=hidden_channels, downsample=False)
                
                self.add_module('%02d EncoderLayer' % (i+1), layer)
    
    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x

class EncoderBottleneckBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, up_channels, layers, downsample_method="conv"):
        super(EncoderBottleneckBlock, self).__init__()

        if downsample_method == "conv":

            for i in range(layers):

                if i == 0:
                    layer = EncoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, up_channels=up_channels, downsample=True)
                else:
                    layer = EncoderBottleneckLayer(in_channels=up_channels, hidden_channels=hidden_channels, up_channels=up_channels, downsample=False)
                
                self.add_module('%02d EncoderLayer' % i, layer)
        
        elif downsample_method == "pool":

            maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.add_module('00 MaxPooling', maxpool)

            for i in range(layers):

                if i == 0:
                    layer = EncoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, up_channels=up_channels, downsample=False)
                else:
                    layer = EncoderBottleneckLayer(in_channels=up_channels, hidden_channels=hidden_channels, up_channels=up_channels, downsample=False)
                
                self.add_module('%02d EncoderLayer' % (i+1), layer)
    
    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x


class DecoderResidualBlock(nn.Module):

    def __init__(self, hidden_channels, output_channels, layers):
        super(DecoderResidualBlock, self).__init__()

        for i in range(layers):

            if i == layers - 1:
                layer = DecoderResidualLayer(hidden_channels=hidden_channels, output_channels=output_channels, upsample=True)
            else:
                layer = DecoderResidualLayer(hidden_channels=hidden_channels, output_channels=hidden_channels, upsample=False)
            
            self.add_module('%02d EncoderLayer' % i, layer)
    
    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x

class DecoderBottleneckBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, down_channels, layers):
        super(DecoderBottleneckBlock, self).__init__()

        for i in range(layers):

            if i == layers - 1:
                layer = DecoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, down_channels=down_channels, upsample=True)
            else:
                layer = DecoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels, down_channels=in_channels, upsample=False)
            
            self.add_module('%02d EncoderLayer' % i, layer)
    
    
    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x


class EncoderResidualLayer(nn.Module):

    def __init__(self, in_channels, hidden_channels, downsample):
        super(EncoderResidualLayer, self).__init__()

        if downsample:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )

        self.weight_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=hidden_channels),
        )

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
            )
        else:
            self.downsample = None

        self.relu = nn.Sequential(
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x = x + identity

        x = self.relu(x)

        return x

class EncoderBottleneckLayer(nn.Module):

    def __init__(self, in_channels, hidden_channels, up_channels, downsample):
        super(EncoderBottleneckLayer, self).__init__()

        if downsample:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
            )

        self.weight_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.weight_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=up_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=up_channels),
        )

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=up_channels, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(num_features=up_channels),
            )
        elif (in_channels != up_channels):
            self.downsample = None
            self.up_scale = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=up_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=up_channels),
            )
        else:
            self.downsample = None
            self.up_scale = None

        self.relu = nn.Sequential(
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        x = self.weight_layer3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)
        elif self.up_scale is not None:
            identity = self.up_scale(identity)

        x = x + identity

        x = self.relu(x)

        return x

class DecoderResidualLayer(nn.Module):

    def __init__(self, hidden_channels, output_channels, upsample):
        super(DecoderResidualLayer, self).__init__()

        self.weight_layer1 = nn.Sequential(
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

        if upsample:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)                
            )
        else:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            )

        if upsample:
            self.upsample = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=1, stride=2, output_padding=1, bias=False)   
            )
        else:
            self.upsample = None
    
    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)

        if self.upsample is not None:
            identity = self.upsample(identity)

        x = x + identity

        return x

class DecoderBottleneckLayer(nn.Module):

    def __init__(self, in_channels, hidden_channels, down_channels, upsample):
        super(DecoderBottleneckLayer, self).__init__()

        self.weight_layer1 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )

        self.weight_layer2 = nn.Sequential(
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

        if upsample:
            self.weight_layer3 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=down_channels, kernel_size=1, stride=2, output_padding=1, bias=False)
            )
        else:
            self.weight_layer3 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=down_channels, kernel_size=1, stride=1, padding=0, bias=False)
            )

        if upsample:
            self.upsample = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=down_channels, kernel_size=1, stride=2, output_padding=1, bias=False)
            )
        elif (in_channels != down_channels):
            self.upsample = None
            self.down_scale = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=in_channels, out_channels=down_channels, kernel_size=1, stride=1, padding=0, bias=False)
            )
        else:
            self.upsample = None
            self.down_scale = None
    
    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        x = self.weight_layer3(x)

        if self.upsample is not None:
            identity = self.upsample(identity)
        elif self.down_scale is not None:
            identity = self.down_scale(identity)

        x = x + identity

        return x

if __name__ == "__main__":

    configs, bottleneck = get_configs("resnet152")

    encoder = ResNetEncoder(configs, bottleneck)

    input = torch.randn((5,3,224,224))

    print(input.shape)

    output = encoder(input)

    print(output.shape)

    decoder = ResNetDecoder(configs[::-1], bottleneck)

    output = decoder(output)

    print(output.shape)