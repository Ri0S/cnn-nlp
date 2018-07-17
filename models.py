import modules
from torch import nn
from configs import config

class cnnNlp(nn.Module):
    def __init__(self):
        super(cnnNlp, self).__init__()
        self.char2word = modules.char2Word()
        
        if config.model == 'resnet34':
            resnet = modules.ResNet(modules.BasicBlock, [3, 4, 6, 3])
        elif config.model == 'resnet50':
            resnet = modules.ResNet(modules.Bottleneck, [3, 4, 6, 3])
        elif config.model == 'resnet101':
            resnet = modules.ResNet(modules.Bottleneck, [3, 4, 23, 3])
        elif config.model == 'resnet152':
            resnet = modules.ResNet(modules.Bottleneck, [3, 8, 36, 3])
        else:
            resnet = modules.ResNet(modules.BasicBlock, [2, 2, 2, 2])

        self.resnet = resnet

    def forward(self, inputs):
        wordvec = self.char2word(inputs).unsqueeze(1)
        out = self.resnet(wordvec)

        return out

