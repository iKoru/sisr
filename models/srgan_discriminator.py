import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, initialSize):
        super(Discriminator, self).__init__()
        outputChannel = 64
        self.shoot = nn.Sequential(
            # input to output channel 1
            nn.Conv2d(initialSize, outputChannel, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # output channel 1 to output channel 2
            nn.Conv2d(outputChannel, outputChannel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(outputChannel),
            nn.LeakyReLU(0.2, inplace=True),
            # output channel 2 to output channel 3
            nn.Conv2d(outputChannel, outputChannel * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outputChannel * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # output channel 3 to output channel 4
            nn.Conv2d(outputChannel * 2,
                      outputChannel * 2,
                      3,
                      2,
                      1,
                      bias=False),
            nn.BatchNorm2d(outputChannel * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # output channel 4 to output channel 5
            nn.Conv2d(outputChannel * 2,
                      outputChannel * 4,
                      3,
                      1,
                      1,
                      bias=False),
            nn.BatchNorm2d(outputChannel * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # output channel 5 to output channel 6
            nn.Conv2d(outputChannel * 4,
                      outputChannel * 4,
                      3,
                      2,
                      1,
                      bias=False),
            nn.BatchNorm2d(outputChannel * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # output channel 6 to output channel 7
            nn.Conv2d(outputChannel * 4,
                      outputChannel * 8,
                      3,
                      1,
                      1,
                      bias=False),
            nn.BatchNorm2d(outputChannel * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # output channel 5 to output channel 6
            nn.Conv2d(outputChannel * 8,
                      outputChannel * 8,
                      3,
                      1,
                      1,
                      bias=False),
            nn.BatchNorm2d(outputChannel * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # output channel 6 to discriminator
            nn.Linear(outputChannel * 8, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.shoot(input).view(-1, 1).squeeze(1)
