import modules
from torch import nn


class cnnNlp(nn.Module):
    def __init__(self):
        super(cnnNlp, self).__init__()
        self.char2word = modules.char2Word()
        self.resnet = modules.ResNet(modules.BasicBlock, [2, 2, 2, 2])

    def forward(self, inputs):
        wordvec = self.char2word(inputs).unsqueeze(1)
        out = self.resnet(wordvec)

        return out
