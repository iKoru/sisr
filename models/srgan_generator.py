import torch
import torch.nn as nn


class block(nn.Module):
    def __init(self, channel, kernal, stride):
        super(block, self).__init()
        self.residual = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channel), nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channel))

    def forward(self, x):
        return self.residual(x) + x


class Generator(nn.Module):
    def __init__(self, initialSize, outputSize):
        super(Generator, self).__init__()
        self.outputChannel = 64
        self.generate = nn.Sequential(
            # input to output channel 1
            nn.Conv2d(initialSize, outputChannel, 9, 1, 4, bias=False),
            nn.LeakyReLU(0.02, inplace=True),
        )
        self.trim = nn.Sequential(
            nn.Conv2d(outputChannel, outputChannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outputChannel))
        self.finish = nn.Sequential(
            nn.Conv2d(outputChannel, outputChannel * 4, 3, 1, 1, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(outputChannel, outputChannel * 4, 3, 1, 1, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(outputChannel, outputSize, 9, 1, 4, bias=False))

    def forward(self, input):
        input2 = self.generate(input)
        input2_clone = input2.clone()
        for i in range(5):
            input2_clone = block(self.outputChannel, 3, 1)(input2_clone)
        input2 = self.trim(input2_clone) + input2
        return self.finish(input2)
