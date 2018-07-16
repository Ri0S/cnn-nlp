import modules
from torch import nn
from configs import config

class cnnNlp(nn.Module):
    def __init__(self):
        super(cnnNlp, self).__init__()
        self.char2word = modules.char2Word()
        self.resnet = modules.ResNet(modules.BasicBlock, [2, 2, 2, 2])

    def forward(self, inputs):
        wordvec = self.char2word(inputs).unsqueeze(1)
        out = self.resnet(wordvec)

        return out


class cnnNlpEj(nn.Module):
    def __init__(self):
        super(cnnNlpEj, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, 64)
        self.resnet = modules.ResNetEj(modules.Bottleneck, [3, 4, 6, 3, 3])

    def forward(self, inputs):
        wordvec = self.embedding(inputs).view(inputs.size(0), 1, -1, 64)
        out = self.resnet(wordvec)

        return out

