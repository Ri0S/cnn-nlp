import modules
import math
from torch import nn
from configs import config

class cnnNlp(nn.Module):
    def __init__(self):
        super(cnnNlp, self).__init__()
        self.char2word = modules.char2Word()
        
        if config.model == 'resnet34':
            resnet = ResNet(modules.BasicBlock, [3, 4, 6, 3])
        elif config.model == 'resnet50':
            resnet = ResNet(modules.Bottleneck, [3, 4, 6, 3])
        elif config.model == 'resnet101':
            resnet = ResNet(modules.Bottleneck, [3, 4, 23, 3])
        elif config.model == 'resnet152':
            resnet = ResNet(modules.Bottleneck, [3, 8, 36, 3])
        else:
            resnet = ResNet(modules.BasicBlock, [2, 2, 2, 2])

        self.resnet = resnet

    def forward(self, inputs):
        if config.char2word:
            wordvec = self.char2word(inputs).unsqueeze(1)
        else:
            wordvec = self.embedding(inputs).view(inputs.size(0), 1, -1, 64)
        out = self.resnet(wordvec)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=51):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 64), stride=2, padding=(3, 0),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        if config.char2word:
            self.maxpool = nn.MaxPool2d(kernel_size=(3, 1), stride=2, padding=(1, 0))
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=(4, 1), stride=3, padding=(1, 0))
            self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
            self.layer2 = self._make_layer(block, 128, layers[1], stride=3)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=3)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=3)
        self.avgpool = nn.AvgPool2d((10, 1), stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


