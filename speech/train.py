from model import Model
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.optim as optimizers
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import copy
import sys

# データの準備
df = pd.read_pickle('data/data.df.pkl')

# 被験者ごとに取り出して正規化（平均をひき，標準偏差で割る）を実行
for subject in df['subject'].unique().tolist():
    idx = np.where(df['subject'] == subject)[0]
    values = df.iloc[idx, 4:]
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-6] = 1.0   # 標準偏差が小さい場合はそのままにする
    df.iloc[idx, 4:] = (values - mean) / std

# 正解ラベル（'NEU'は無し）
emotion_list = df['emotion'].unique().tolist()

# 訓練，検証，評価のDataFrameリスト（感情ごと）を作成する
df_trains = []
df_valis = []
df_tests = []
# 感情ごとに繰り返す
for emo in emotion_list:
    # 感情に対応する発話番号のリスト
    utter_list = df[df['emotion'] == emo]['utterance'].unique().tolist()
    # 全体の発話数を取り出し，学習データの発話数(75%)，検証データの発話数（10%）に按分
    num_utters = len(utter_list)
    num_train_utters = (num_utters * 15) // 20
    num_vali_utters = (num_utters * 2) // 20
    # 学習，検証，評価の発話番号リストに分解
    train_utter_list = utter_list[:num_train_utters]
    vali_utter_list = utter_list[num_train_utters:(num_train_utters + num_vali_utters)]
    test_utter_list = utter_list[(num_train_utters + num_vali_utters):]
    #　該当する行を取り出し学習，検証，評価のDataFrameリストに追加
    df_trains.append(df[
        df['utterance'].apply(lambda x: x in train_utter_list)
    ])
    df_valis.append(df[
        df['utterance'].apply(lambda x: x in vali_utter_list)
    ])
    df_tests.append(df[
        df['utterance'].apply(lambda x: x in test_utter_list)
    ])
# DataFrameリストを結合して学習，検証，評価用のDataFrameを生成
df_train = pd.concat(df_trains, axis=0)
df_vali = pd.concat(df_valis, axis=0)
df_test = pd.concat(df_tests, axis=0)

# NEUを含む感情名のリスト
emotion_list_with_neu = ['NEU'] + emotion_list
# 感情名をキー，その番号を値とする辞書の作成
emo2no = dict([(emo , no) for (no, emo) in enumerate(emotion_list_with_neu)])

def extract_x_y(df: pd.DataFrame):
    """DataFrameから特徴量とラベルのarrayを生成"""
    # 強度が0の行を抽出して，ニュートラルのXとYとする
    idx_neu = np.where(df['strength'] == 0)[0]
    x_neu = df.iloc[idx_neu, 4:]
    y_neu = df['emotion'].iloc[idx_neu].apply(lambda x: 0)
    # 強度が3以上の行を抽出して，感情ありのXとYとする
    idx_emo = np.where(df['strength'] >= 3)[0]
    x_emo = df.iloc[idx_emo, 4:]
    y_emo = df['emotion'].iloc[idx_emo].apply(lambda x: emo2no[x])
    # ニュートラルと感情ありのXとYを結合して全体のXとYとする
    x = pd.concat([x_neu, x_emo], axis=0)
    y = pd.concat([y_neu, y_emo], axis=0)
    return x, y

# 学習，検証，評価データのXとYを抽出
x_train, y_train = extract_x_y(df_train)
x_vali, y_vali = extract_x_y(df_vali)
x_test, y_test = extract_x_y(df_test)

def make_data(x, y):
    """特徴量とラベルのarrayから，DataLoaderに与えるためのタプルのリストを作成する"""
    x = x.astype(np.float32)
    y = y.astype(np.int64)
    data = []
    for xx, yy in zip(x, y):
        data.append((xx, yy))
    return data

# DataLoaderに与えるための(特徴量, ラベル)のリストを作成する
data_train = make_data(x_train.to_numpy(), y_train.to_numpy())
data_vali = make_data(x_vali.to_numpy(), y_vali.to_numpy())
data_test = make_data(x_test.to_numpy(), y_test.to_numpy())

# 学習用，検証用のDataLoader
train_dataloader = DataLoader(data_train, batch_size=200, shuffle=True)
vali_dataloader = DataLoader(data_vali, batch_size=10, shuffle=False)

# 学習のメイン部分（『詳細ディープラーニング』のコードを参考に作成）

# 初期化．乱数のシードを設定とGPUデバイスの確認
np.random.seed(123)
torch.manual_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# モデルの作成
x_dim = x_train.shape[1]
y_dim = len(emotion_list_with_neu)
model = Model(x_dim, y_dim).to(device)

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

# エポックは最大100とする
epochs = 100

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
