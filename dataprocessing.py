import csv
import os
import pickle

content = []

# if not os.path.isfile('./data/u1000.csv'):
#     with open('./data/content.csv', encoding='utf-8') as f:
#         reader = csv.reader(f)
#
#         for a in reader:
#             break
#         for t in reader:
#             if 900 < len(t[2]) < 1200:
#                 content.append(t)
#
#     with open('./data/u1000.csv', 'w', newline='', encoding='utf-8') as f:
#         wr = csv.writer(f)
#         wr.writerow(['id', 'title', 'content', 'sid'])
#         for t in content:
#             wr.writerow(t)
#
# else:
#     with open('./data/u1000.csv', encoding='utf-8') as f:
#         reader = csv.reader(f)
#         for c in reader:
#             break
#         for c in reader:
#             content.append(c)

with open('./data/content.csv', encoding='utf-8') as f:
    reader = csv.reader(f)

    for a in reader:
        break
    for t in reader:
        content.append(t)

train = content[0:int(len(content) * 0.8)]
valid = content[int(len(content) * 0.8):int(len(content) * 0.9)]
test = content[int(len(content) * 0.9):]

if not os.path.isfile('./data/i2c') or not os.path.isfile('./data/c2i'):
    i2c = ['<pad>', '<unk>', '<sos>', '<eos>']
    c2i = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
    i = 4

    for _, title, sentence, _ in train:
        line = title + sentence
        for w in line:
            try:
                c2i[w]
            except KeyError:
                c2i[w] = i
                i2c.append(w)
                i += 1
    with open('./data/i2c.pkl', 'wb') as f:
        pickle.dump(i2c, f)
    with open('./data/c2i.pkl', 'wb') as f:
        pickle.dump(c2i, f)

else:
    with open('./data/i2c.pkl', 'rb') as f:
        i2c = pickle.load(f)
    with open('./data/i2c.pkl', 'rb') as f:
        c2i = pickle.load(f)

if not os.path.isfile('./data/train.pkl') or not os.path.isfile('./data/train.pkl') or \
        not os.path.isfile('./data/train.pkl'):
    trainpkl = []
    for t in train:
        tt = []
        tttt = (t[2] + t[3]).split()
        for w in tttt:
            if len(w) <= 10:
                ttt = []
                for c in w:
                    ttt.append(c2i[c])
                tt.append(ttt)
        if 260 <= len(tt) <= 300:
            trainpkl.append([tt, t[3]])

    testpkl = []
    for t in test:
        tt = []
        tttt = (t[2] + t[3]).split()
        for w in tttt:
            if len(w) <= 10:
                ttt = []
                for c in w:
                    try:
                        ttt.append(c2i[c])
                    except KeyError:
                        ttt.append(c2i['<unk>'])
                tt.append(ttt)
        if 260 <= len(tt) <= 300:
            testpkl.append([tt, t[3]])

    validpkl = []
    for t in valid:
        tt = []
        tttt = (t[2] + t[3]).split()
        for w in tttt:
            if len(w) <= 10:
                ttt = []
                for c in w:
                    try:
                        ttt.append(c2i[c])
                    except KeyError:
                        ttt.append(c2i['<unk>'])
                tt.append(ttt)
        if 260 <= len(tt) <= 300:
            validpkl.append([tt, t[3]])

    print(len(trainpkl), len(testpkl), len(validpkl))
    with open('./data/train.pkl', 'wb') as f:
        pickle.dump(trainpkl, f)
    with open('./data/test.pkl', 'wb') as f:
        pickle.dump(testpkl, f)
    with open('./data/valid.pkl', 'wb') as f:
        pickle.dump(validpkl, f)
