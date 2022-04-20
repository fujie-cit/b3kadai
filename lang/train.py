from model import Model
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.optim as optimizers
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import copy

# データの準備
df = pd.read_pickle('data/extracted_features_32k.df.pkl')

# トピックを Xperia (10000) に限定する
df = df[df['topic'] == 10000]

# posのみ，negのみとそれ以外に分類する．
# それ以外にはいくつかパターンがあるが，pos, neg, pos_neg のいずれかが立ってる
# ものは対象外にする
# neu が立ってるもの，dont_care が立ってるものからバランスよくランダムに負例を取ってくる

def extract_tweets_with_labels(condition):
     subdf = df[np.all(df[['pos_neg', 'pos', 'neg', 'neu', 'dont_care']].to_numpy() == condition, axis=1)]
     return subdf
     
df_pos = extract_tweets_with_labels([0, 1, 0, 0, 0])
df_neg = extract_tweets_with_labels([0, 0, 1, 0, 0])
df_neu = df[df['neu'] == 1]
res_idx = sorted(list(set(df.index) - set(list(df_pos.index) + list(df_neg.index) + list(df_neu.index))))
df_dc = df.loc[res_idx]

np.random.seed(123)
torch.manual_seed(123)

num_pos = df_pos.shape[0]
num_neg = df_neg.shape[0]
num_neu = (num_pos + num_neg) // 4
num_dc = num_neu

idx_neu = list(range(df_neu.shape[0]))
np.random.shuffle(idx_neu)
df_neu = df_neu.iloc[idx_neu[:num_neu]]

idx_dc = list(range(df_dc.shape[0]))
np.random.shuffle(idx_dc)
df_dc = df_dc.iloc[idx_dc[:num_dc]]

df_others = pd.concat([df_neu, df_dc], axis=0)

def extract_train_vali_test(df, train_ratio=0.8, vali_ratio=0.1):
    num = df.shape[0]
    train_end = int(num * train_ratio)
    vali_end = int(num * (train_ratio + vali_ratio))
    return df.iloc[:train_end, :], df.iloc[train_end:vali_end, :], df.iloc[vali_end:, :]

df_pos_train, df_pos_vali, df_pos_test = extract_train_vali_test(df_pos)
df_neg_train, df_neg_vali, df_neg_test = extract_train_vali_test(df_neg)
df_oth_train, df_oth_vali, df_oth_test = extract_train_vali_test(df_others)

def combine_pos_neg_oth(df_pos, df_neg, df_oth):
    num_pos = df_pos.shape[0]
    num_neg = df_neg.shape[0]
    num_oth = df_oth.shape[0]
    label = ([0] * num_pos) + ([1] * num_neg) + ([2] * num_oth)
    df_all = pd.concat([df_pos, df_neg, df_oth], axis=0)
    df_label = pd.DataFrame(dict(label=label), index=df_all.index)
    return pd.concat([df_label, df_all], axis=1)

df_train = combine_pos_neg_oth(df_pos_train, df_neg_train, df_oth_train)
df_vali = combine_pos_neg_oth(df_pos_vali, df_neg_vali, df_oth_vali)
df_test = combine_pos_neg_oth(df_pos_test, df_neg_test, df_oth_test)

# df_train.to_csv('data_train.csv')
# sys.exit(0)

def extract_x_y(df: pd.DataFrame):
    x = df.iloc[:, 10:]
    y = df.iloc[:, 0]
    return x, y

x_train, y_train = extract_x_y(df_train)
x_vali, y_vali = extract_x_y(df_vali)
x_test, y_test = extract_x_y(df_test)

def make_data(x, y):
    x = x.astype(np.float32)
    y = y.astype(np.int64)
    data = []
    for xx, yy in zip(x, y):
        data.append((xx, yy))
    return data

data_train = make_data(x_train.to_numpy(), y_train.to_numpy())
data_vali = make_data(x_vali.to_numpy(), y_vali.to_numpy())
data_test = make_data(x_test.to_numpy(), y_test.to_numpy())

train_dataloader = DataLoader(data_train, batch_size=10, shuffle=True)
vali_dataloader = DataLoader(data_vali, batch_size=10, shuffle=False)

# メイン
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_dim = x_train.shape[1]
y_dim = 3
model = Model(x_dim, y_dim).to(device)

criterion = nn.CrossEntropyLoss()
# optimizer = optimizers.SGD(model.parameters(), lr=0.1)
optimizer = optimizers.Adam(model.parameters())

def compute_loss(t, y):
    return criterion(y, t)

def train_step(x, t):
    model.train()
    preds = model(x)
    loss = compute_loss(t, preds)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, preds

def vali_step(x, t):
    model.eval()
    preds = model(x)
    loss = criterion(preds, t)

    return loss, preds

epochs = 100

log = dict(epoch=[], train_loss=[], train_acc=[], vali_loss=[], vali_acc=[])

best_loss = 1e+10
best_model_params = None

for epoch in range(epochs):
    train_loss = 0.
    train_acc = 0.

    for (x, t) in train_dataloader:
        x, t = x.to(device), t.to(device)
        loss, preds = train_step(x, t)
        train_loss += loss.item()
        train_acc += \
            accuracy_score(t.tolist(),
            preds.argmax(dim=-1).tolist())


    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)

    vali_loss = 0.
    vali_acc = 0.

    for (x, t) in vali_dataloader:
        x, t = x.to(device), t.to(device)
        loss, preds = vali_step(x, t)
        vali_loss += loss.item()
        vali_acc += \
            accuracy_score(t.tolist(),
                           preds.argmax(dim=-1).tolist())

    vali_loss /= len(vali_dataloader)
    vali_acc /= len(vali_dataloader)
    # print('vali_loss: {:.3f}, vali_acc: {:.3f}'.format(
    #     test_loss,
    #     test_acc
    # )

    print('epoch: {}, train_loss: {:.3}, train_acc: {:.3f}, vali_loss: {:.3f}, vali_acc: {:.3f}'.format(
        epoch+1,
        train_loss,
        train_acc,
        vali_loss,
        vali_acc
    ))

    if vali_loss < best_loss:
        print("best loss updated")
        # preserve the best parameters
        best_model_params = copy.deepcopy(model.state_dict())
        best_loss = vali_loss

    log['epoch'].append(epoch+1)
    log['train_loss'].append(train_loss)
    log['train_acc'].append(train_acc)
    log['vali_loss'].append(vali_loss)
    log['vali_acc'].append(vali_acc)

pd.DataFrame(log).to_csv('data/train_log.csv')

model.load_state_dict(best_model_params)

# モデルパラメータをファイルに保存
torch.save(model.state_dict(), "data/model_params.pth")
