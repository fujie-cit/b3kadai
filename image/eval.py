from model import Model
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

# データの準備
df = pd.read_pickle('data.df.pkl')

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


# 損失関数
criterion = nn.CrossEntropyLoss()

# 検証のための予測と損失の計算をする関数
def vali_step(x, t):
    model.eval()
    preds = model(x)
    loss = criterion(preds, t)

    return loss, preds

# モデルの作成
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(136, 8).to(device)

# モデルパラメタの読み込み
model_params = torch.load("model_params.pth")
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
labels = ['Neu', 'Ang', 'Con', 'Dis', 'Fea', 'Hap', 'Sad', 'Sur']
labels_ef = ['Neu', 'Ang', 'Dis', 'Hap', 'Sur']

# 正解ラベルと予測結果のカテゴリ名リスト（この両者を比較．同じ位置で一致していれば正解）
tgts_lbl = np.array(labels)[tgts]
preds_lbl = np.array(labels)[preds]

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
target_index = pd.MultiIndex.from_tuples(list(zip(['target'] * 6, labels_ef + ['total'])))
pred_index = pd.MultiIndex.from_tuples(list(zip(['pred'] * 6, labels_ef + ['total'])))
df_conf = pd.DataFrame(cf_mat, index=target_index, columns=pred_index)
# 混同行列の表示
print(df_conf)
