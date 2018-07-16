import utils
import models
import pickle
from torch import nn, optim
from torch.utils.data import DataLoader
from configs import config
from tqdm import tqdm

trainLoader = DataLoader(utils.Dsets(config.mode), batch_size=128, shuffle=True, collate_fn=utils.collate_fn)
validLoader = DataLoader(utils.Dsets(config.mode), batch_size=128, shuffle=True, collate_fn=utils.collate_fn)

inv_dict = pickle.load(open(config.id2char_path, 'rb'))
config.vocab_size = len(inv_dict)

model = models.cnnNlp().to(config.device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for i in range(config.n_epoch):
    model.train()
    total_loss = 0
    for idx, (inputs, target) in enumerate(tqdm(trainLoader)):
        optimizer.zero_grad()
        out = model(inputs)
        loss = criterion(out, target)
        loss.backward()

        total_loss += loss
        optimizer.step()
    print('epoch', i, 'loss:', total_loss.item() / idx)

    model.eval()
    loss = 0
    for idx, (inputs, target) in enumerate(validLoader):
        out = model(input)
        loss += criterion(out, target)
    print('valid loss: ', loss.itme() / idx)
