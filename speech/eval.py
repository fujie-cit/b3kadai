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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# モデルの作成
x_dim = x_train.shape[1]
y_dim = len(emotion_list_with_neu)
model = Model(x_dim, y_dim).to(device)

# 損失関数
criterion = nn.CrossEntropyLoss()

# 検証のための予測と損失の計算をする関数
def vali_step(x, t):
    model.eval()
    preds = model(x)
    loss = criterion(preds, t)

    return loss, preds

# モデルパラメタの読み込み
model_params = torch.load("data/model_params.pth")
model.load_state_dict(model_params)

model.eval()

# 評価データ用のDataLoader
test_dataloader = DataLoader(data_test, batch_size=1000, shuffle=False)

# 評価を行う
for x, t in test_dataloader:
    x, t = x.to(device), t.to(device)
    loss, preds = vali_step(x, t)
    tgts = t.tolist()
    preds = preds.argmax(axis=1).tolist()

# フルラベルリスト（識別対象でないものも含む）と，有効ラベルリスト
# （カテゴリを絞り込んだ画像課題の名残り）
labels = emotion_list_with_neu
labels_ef = emotion_list_with_neu

# 正解ラベルと予測結果のカテゴリ名リスト（この両者を比較．同じ位置で一致していれば正解）
tgts_lbl = np.array(labels)[tgts]
preds_lbl = np.array(labels_ef)[preds]

# データ数を表示
print("data count {}".format(len(tgts_lbl)))

# 適合率，再現率，F-1値などを表示
print(classification_report(tgts_lbl, preds_lbl, labels=labels_ef))

# 混同行列を作成
cf_mat = confusion_matrix(tgts_lbl, preds_lbl, labels=labels_ef)
# 混同行列の行，列の合計値を求めて追加する
cf_mat = np.concatenate([cf_mat, cf_mat.sum(axis=1, keepdims=True)], axis=1)
cf_mat = np.concatenate([cf_mat, cf_mat.sum(axis=0, keepdims=True)], axis=0)
# target（正解）, pred（予測）が分かるようにマルチラベルをつけてDataFrame化する
target_index = pd.MultiIndex.from_tuples(list(zip(['target'] * 10, labels_ef + ['total'])))
pred_index = pd.MultiIndex.from_tuples(list(zip(['pred'] * 10, labels_ef + ['total'])))
df_conf = pd.DataFrame(cf_mat, index=target_index, columns=pred_index)
# 混同行列の表示
print(df_conf)
