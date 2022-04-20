from model import Model
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.optim as optimizers
import torch.nn as nn
from sklearn.metrics import accuracy_score
import copy

# データの準備
df = pd.read_pickle('data/data.df.pkl')

# データ数の少ない軽蔑，恐怖，悲しみに該当する行を取り除く
df = df[(df['emotion'] != 2) & (df['emotion'] != 4) & (df['emotion'] != 6)]

# 被験者リストを作成する
subjs = df['subject'].unique().tolist()

# 学習用，検証用，評価用の被験者リストを作成する
subjs_train = subjs[:-50]
subjs_vali  = subjs[-50:-30]
subjs_test  = subjs[-30:]

def get_subdf(df, subj_list):
    """dfから被験者リストに含まれる被験者に対応する行を取り出す"""
    condition = df['subject'].apply(lambda x: x in subj_list)
    return df[condition]

# 学習用，検証用，評価用のデータフレームを抽出する
df_train = get_subdf(df, subjs_train)
df_vali = get_subdf(df, subjs_vali)
df_test = get_subdf(df, subjs_test)

# 特徴量の規格化を行うための平均と分散を求める
x_train = df_train.iloc[:, 4:].to_numpy().astype(np.float32)
x_mean = x_train.mean(axis=0)
x_std = x_train.std(axis=0)

# 特徴量と正解ラベルを取り出す．特徴量は規格化も行う．
x_train = df_train.iloc[:, 4:].to_numpy().astype(np.float32)
x_train -= x_mean
x_train /= x_std
y_train = df_train['emotion'].to_numpy().astype(np.int64)
x_vali = df_vali.iloc[:, 4:].to_numpy().astype(np.float32)
x_vali -= x_mean
x_vali /= x_std
y_vali = df_vali['emotion'].to_numpy().astype(np.int64)
x_test = df_test.iloc[:, 4:].to_numpy().astype(np.float32) 
x_test -= x_mean
x_test /= x_std
y_test = df_test['emotion'].to_numpy().astype(np.int64)

def make_data(x, y):
    """特徴量とラベルのarrayから，DataLoaderに与えるためのタプルのリストを作成する"""
    data = []
    for xx, yy in zip(x, y):
        data.append((xx, yy))
    return data

# DataLoaderに与えるための(特徴量, ラベル)のリストを作成する
data_train = make_data(x_train, y_train)
data_vali = make_data(x_vali, y_vali)
data_test = make_data(x_test, y_test)

# 学習用，検証用のDataLoader
train_dataloader = DataLoader(data_train, batch_size=10, shuffle=True)
vali_dataloader = DataLoader(data_vali, batch_size=10, shuffle=False)

# 学習のメイン部分（『詳細ディープラーニング』のコードを参考に作成）

# 初期化．乱数のシードを設定とGPUデバイスの確認
np.random.seed(123)
torch.manual_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# モデルの作成
model = Model(136, 8).to(device)

# 損失関数
criterion = nn.CrossEntropyLoss()
# オプティマイザ
# optimizer = optimizers.SGD(model.parameters(), lr=0.01)
optimizer = optimizers.Adam(model.parameters())

# 損失の計算をする関数
def compute_loss(t, y):
    return criterion(y, t)

# 予測と損失計算とパラメータの更新を行う関数
def train_step(x, t):
    model.train()
    preds = model(x)
    loss = compute_loss(t, preds)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, preds

# 検証のための予測と損失の計算をする関数
def vali_step(x, t):
    model.eval()
    preds = model(x)
    loss = criterion(preds, t)

    return loss, preds

# エポックは最大50とする
epochs = 50

# 学習時にエポックごとの損失と正解率の記録するための辞書
log = dict(epoch=[], train_loss=[], train_acc=[], vali_loss=[], vali_acc=[])

# 最良の検証ロスの値とその時のパラメータを保存するための変数
best_loss = 1e+10
best_model_params = None

# 学習の繰り返し部分
for epoch in range(epochs):
    train_loss = 0.
    train_acc = 0.

    # 学習データのミニバッチごとに繰り返す
    for (x, t) in train_dataloader:
        # 学習データに対する予測とパラメータ更新
        x, t = x.to(device), t.to(device)
        loss, preds = train_step(x, t)
        train_loss += loss.item()
        train_acc += \
            accuracy_score(t.tolist(),
            preds.argmax(dim=-1).tolist())

    # 学習データに対する平均損失と平均正解率を計算
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)

    vali_loss = 0.
    vali_acc = 0.

    # 検証データのミニバッチごとに繰り返す
    for (x, t) in vali_dataloader:
        # 検証データに対する予測
        x, t = x.to(device), t.to(device)
        loss, preds = vali_step(x, t)
        vali_loss += loss.item()
        vali_acc += \
            accuracy_score(t.tolist(),
                           preds.argmax(dim=-1).tolist())

    # 検証データに対する平均損失と平均正解率を計算
    vali_loss /= len(vali_dataloader)
    vali_acc /= len(vali_dataloader)

    # ログを表示
    print('epoch: {}, train_loss: {:.3}, train_acc: {:.3f}, vali_loss: {:.3f}, vali_acc: {:.3f}'.format(
        epoch+1,
        train_loss,
        train_acc,
        vali_loss,
        vali_acc
    ))

    # 検証ロスがベストのものだったら記録を更新
    if vali_loss < best_loss:
        print("best loss updated")
        # preserve the best parameters
        best_model_params = copy.deepcopy(model.state_dict())
        best_loss = vali_loss

    # 記録用の辞書を更新
    log['epoch'].append(epoch+1)
    log['train_loss'].append(train_loss)
    log['train_acc'].append(train_acc)
    log['vali_loss'].append(vali_loss)
    log['vali_acc'].append(vali_acc)

# 学習の記録をCSVファイルに出力
pd.DataFrame(log).to_csv('data/train_log.csv')

# 最良の検証結果だったモデルパラメータを復元
model.load_state_dict(best_model_params)

# モデルパラメータをファイルに保存
torch.save(model.state_dict(), "data/model_params.pth")

