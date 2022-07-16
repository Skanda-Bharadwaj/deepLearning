import torch
from Inception import InceptionModule, ConvBlock, Auxiliary

class GoogLeNet(torch.nn.Module):

    def __init__(self, num_Classes, im_channels=3, use_auxiliary=True):
        super(GoogLeNet, self).__init__()

        self.conv1 = ConvBlock(im_channels, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = torch.nn.Sequential(
            ConvBlock(64, 64, kernel_size=1, stride=1, padding=0),
            ConvBlock(64, 192, kernel_size=3, stride=1, padding=1))

        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = torch.nn.AvgPool2d(kernel_size=7, stride=1)

        self.dropout = torch.nn.Dropout(0.4)
        self.linear  = torch.nn.Linear(1024, num_Classes)

        self.use_auxiliary = use_auxiliary
        if self.use_auxiliary:
            self.auxiliary4a = Auxiliary(512, num_Classes)
            self.auxiliary4d = Auxiliary(528, num_Classes)

        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        for m in self.modules():
            if isinstance(m, torch.torch.nn.Conv2d) or isinstance(m, torch.torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.detach().zero_()

    def forward(self, x):
        y = None
        z = None

        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool(x)

        x = self.inception4a(x)
        if self.training and self.use_auxiliary:
            y = self.auxiliary4a(x)

        x = self.inception4b(x)
        x = self.inception4c(x)

        x = self.inception4d(x)
        if self.training and self.use_auxiliary:
            z = self.auxiliary4d(x)

        x = self.inception4e(x)
        x = self.maxpool(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)

        x = self.dropout(x)
        x = self.linear(x)

        return x, y, z