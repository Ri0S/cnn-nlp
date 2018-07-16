import torch
import models
import utils
import pickle
from torch import nn, optim
from torch.utils.data import DataLoader
from configs import config
from tqdm import tqdm

inv_dict = pickle.load(open(config.id2char_path, 'rb'))
config.vocab_size = len(inv_dict)

trainLoader = DataLoader(utils.Dsets(config.mode), batch_size=config.batch_size, shuffle=True, collate_fn=utils.collate_fn)
validLoader = DataLoader(utils.Dsets(config.mode), batch_size=config.batch_size, shuffle=True, collate_fn=utils.collate_fn)


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
    torch.save(model.state_dict(), './model/epoch' + str(i))
    model.eval()
    loss = 0
    for idx, (inputs, target) in enumerate(validLoader):
        out = model(inputs)
        loss += criterion(out, target)
    print('valid loss: ', loss.itme() / idx)

