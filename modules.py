import torch
from torch import nn
from configs import config


class Rnn(nn.Module):
    def __init__(self):
        super(Rnn, self).__init__()
        self.rnn = nn.GRU(input_size=64,
                          hidden_size=128,
                          num_layers=1,
                          bidirectional=False,
                          batch_first=True)

        self.fc = nn.Linear(128, 51)

    def forward(self, inputs):
        out, h = self.rnn(inputs.squeeze(1), self.init_hidden(inputs.size(0)))
        x = self.fc(h)

        return x.squeeze(0)

    def init_hidden(self, bts):
        return torch.zeros((1, bts, 128), dtype=torch.float32, device=config.device)


class char2Word(nn.Module):
    def __init__(self):
        super(char2Word, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.conv = nn.Conv2d(1, 64, (2, config.embedding_size))
        self.a = nn.Conv2d(1, 64, (2, config.embedding_size))
        self.b = nn.MaxPool1d(20)

    def forward(self, inputs):
        x = inputs.view(-1, inputs.size(2))
        x_emb = self.embedding(x).unsqueeze(1)
        out, _ = self.conv(x_emb).max(2)

        return out.view(inputs.size(0), inputs.size(1), -1)


def conv3x64(in_planes, out_planes, stride=1):
    """3x64 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 1), stride=stride,
                     padding=(1, 0), bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x64(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x64(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 1), stride=stride,
                               padding=(1, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
