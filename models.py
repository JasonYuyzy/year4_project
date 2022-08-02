# -*- encoding: utf-8 -*-
# -------------------------------------------
# Year 4 personal project code work network model part
# -------------------------------------------
# Zhengyu Yu

import torch
import torch.nn as nn
import torch.nn.functional as F


# the data-binding STNs
class Where_stn(nn.Module):
    def __init__(self):
        super(Where_stn, self).__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            # conv1
            nn.Conv2d(6, 32, kernel_size=4, stride=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            # conv2
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # conv3
            nn.Conv2d(64, 128, kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # conv4
            nn.Conv2d(128, 256, kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # conv5
            nn.Conv2d(256, 512, kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            # linear1
            nn.Linear(in_features=25088, out_features=512),
            nn.Dropout(0.5),
            nn.ReLU(True),
            # linear2
            nn.Linear(512, 3 * 2)
        )

    # Spatial transformer network forward function
    def forward(self, FG, BG):
        x = torch.cat((BG, FG), dim=1)
        out = self.localization(x)
        out = out.view(out.size(0), -1)
        theta = self.fc_loc(out)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, FG.size(), align_corners=True)
        x = F.grid_sample(FG, grid, padding_mode="border", align_corners=True)

        return x


# the feature-binding STNs
class Where_nstn(nn.Module):
    def __init__(self):
        super(Where_nstn, self).__init__()
        # FG img localization out 256
        self.FGconv2dlayers = nn.Sequential(
            # conv1
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            # conv2
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # conv3
            nn.Conv2d(64, 128, kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # conv4
            nn.Conv2d(128, 256, kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # conv5
            nn.Conv2d(256, 512, kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # BG img localization out 256
        self.BGconv2dlayers = nn.Sequential(
            # conv1
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            # conv2
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # conv3
            nn.Conv2d(64, 128, kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # conv4
            nn.Conv2d(128, 256, kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # conv5
            nn.Conv2d(256, 512, kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Linear layer1 output 512 to 6
        self.Linearlayers = nn.Sequential(
            # linear1
            nn.Linear(in_features=50176, out_features=512),
            nn.Dropout(0.5),
            nn.ReLU(True),
            # linear2
            nn.Linear(512, 3 * 2)
        )

        # transform forward

    def forward(self, FG, BG):
        outFG = self.FGconv2dlayers(FG)
        outBG = self.BGconv2dlayers(BG)
        out = torch.cat((outBG, outFG), dim=1)
        out = out.view(out.size(0), -1)  # resize
        theta = self.Linearlayers(out)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, FG.size(), align_corners=True)
        x = F.grid_sample(FG, grid, padding_mode="border", align_corners=True)

        return x


# the traditional version with sigmoid
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # layer1 random noise, output 32
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, bias=False),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # layer2 output 64
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # layer3 output 128
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # layer4 output 256
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer5 output 512
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer6 output 1
        self.LayerScore = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=3, stride=1, bias=False),
            nn.Sigmoid()  # True False detection
        )

    # transform forward
    def forward(self, img):
        out = self.layer1(img)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        score = self.LayerScore(out)

        return score


# the WGANs version without sigmoid()
class DiscriminatorW(nn.Module):
    def __init__(self):
        super(DiscriminatorW, self).__init__()
        # conv1
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=6, stride=3, bias=False),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # conv2
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=6, stride=3, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # conv3
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # conv4
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # conv5
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # conv6
        self.LayerScore = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=2, bias=False),
        )

    # transform forward
    def forward(self, img):
        out = self.layer1(img)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        score = self.LayerScore(out)

        return score


# for Dis net
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)