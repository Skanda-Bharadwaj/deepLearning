import torch

class Auxiliary(torch.nn.Module):

    def __init__(self, in_channels, num_classes):
        super(Auxiliary, self).__init__()

        self.avgpool = torch.nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv1x1 = ConvBlock(in_channels, 128, kernel_size=1)

        self.fc1 = torch.nn.Linear(2048, 1024)
        self.fc2 = torch.nn.Linear(1024, num_classes)

        self.dropout = torch.nn.Dropout(0.7)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1x1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class ConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(ConvBlock, self).__init__()

        self.Conv = torch.nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    **kwargs)

        self.BatchNorm = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.Conv(x)
        x = self.BatchNorm(x)
        x = self.relu(x)

        return x


class InceptionModule(torch.nn.Module):

    def __init__(self, im_channels, num_1C, num_3C_reduce, num_3C, num_5C_reduce, num_5C, num_pool_proj):
        super(InceptionModule, self).__init__()

        self.Conv_1x1 = ConvBlock(in_channels=im_channels, out_channels=num_1C, kernel_size=1)

        self.Conv_3x3_reduce = ConvBlock(in_channels=im_channels, out_channels=num_3C_reduce, kernel_size=1)
        self.Conv_3x3 = ConvBlock(in_channels=num_3C_reduce, out_channels=num_3C, kernel_size=3, padding=1)

        self.Conv_5x5_reduce = ConvBlock(in_channels=im_channels, out_channels=num_5C_reduce, kernel_size=1)
        self.Conv_5x5 = ConvBlock(in_channels=num_5C_reduce, out_channels=num_5C, kernel_size=5, padding=2)

        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool_proj = ConvBlock(in_channels=im_channels, out_channels=num_pool_proj, kernel_size=1)

    def forward(self, x):

        x1 = self.Conv_1x1(x)

        x2 = self.Conv_3x3_reduce(x)
        x2 = self.Conv_3x3(x2)

        x3 = self.Conv_5x5_reduce(x)
        x3 = self.Conv_5x5(x3)

        x4 = self.maxpool(x)
        x4 = self.pool_proj(x4)

        x = torch.concat([x1, x2, x3, x4], 1)

        return x


