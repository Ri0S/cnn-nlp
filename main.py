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

if config.char2word:
    model = models.cnnNlp().to(config.device)
else:
    model = models.cnnNlpEj().to(config.device)

if config.succeed:
    start = int(config.model[[a.isdigit() for a in config.saved_model[10:]].index(True) + 10:])
    save_state = torch.load('./model/' + config.saved_model)
    model.load_state_dict(save_state)
else:
    start = 0

if config.mode == 'train':
    trainLoader = DataLoader(utils.Dsets(config.mode), batch_size=config.batch_size, shuffle=True, collate_fn=utils.collate_fn)
    validLoader = DataLoader(utils.Dsets('valid'), batch_size=config.batch_size, shuffle=True, collate_fn=utils.collate_fn)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for i in range(start, start + config.n_epoch):
        model.train()
        total_loss = 0
        for idx, (inputs, target) in enumerate(tqdm(trainLoader, ncols=80)):
            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, target)
            loss.backward()

            total_loss += loss.item()
            optimizer.step()
        print('epoch', i, 'loss:', total_loss / idx)
        torch.save(model.state_dict(), './model/' + config.model + '_epoch_' + str(i))
        model.eval()
        loss = 0
        for idx, (inputs, target) in enumerate(tqdm(validLoader, ncols=80)):
            out = model(inputs)
            loss += criterion(out, target).item()
        print('valid loss: ', loss / idx)
else:
    testLoader = DataLoader(utils.Dsets(config.mode), batch_size=config.batch_size, shuffle=True, collate_fn=utils.collate_fn)
    saved_state = torch.load('./model/' + config.saved_model)
    model.load_state_dict(saved_state)
    model.eval()

    correct = 0
    total = 0
    for inputs, target in tqdm(testLoader, ncols=80):
        out = model(inputs)
        correct += (torch.max(out, 1)[1] == target).sum().item()
        total += inputs.size(0)
    print('Total:', total)
    print('Correct:', correct)
    print('Accuracy:', correct / total)
