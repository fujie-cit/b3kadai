import numpy as np
import pandas as pd
import opensmile
import glob
import os
import tqdm

###  STEP 1. openSMILE で特徴抽出を行う
# openSMILEの初期化
smile = opensmile.Smile(feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals)

# 音声ファイルのフルパスのリスト
wavfile_list = glob.glob(
    '/autofs/diamond/share/corpus/OGVC/Vol2/Acted/wav/*/*/*.wav')

name_list = []      # ファイル名（の拡張子を除いたもの）のリスト
features_list = []  # 特徴抽出結果（numpy array）のリスト

# 音声ファイルで繰り返し
for wavfile_path in tqdm.tqdm(wavfile_list):
    # ファイル名から拡張子を除いたものを抽出してname_listに追加
    wavfile_basename = os.path.basename(wavfile_path)
    wavfile_body, _ = os.path.splitext(wavfile_basename)
    name_list.append(wavfile_body)
    # 音声ファイルからopenSMILEで特徴抽出してfeatures_listに追加    
    features = smile.process_file(wavfile_path)
    features_list.append(features.to_numpy())

# 特徴量のDataFrameとファイル名のDataFrameを作成，結合して1つにまとめる
column_names = features.columns
df_features = pd.DataFrame(np.concatenate(features_list), columns=column_names)
df_names = pd.DataFrame(name_list, columns=["name"])
df = pd.concat([df_names, df_features], axis=1)

### STEP 2. ファイル名から必要な情報を抜き出して整理し直す
# name（ファイル名の拡張子を抜いた部分相当）のリストを取得
names = df['name'].tolist()
# 特徴量の部分を取得
values = df.iloc[:, 1:]

# nameを，被験者，発話番号，感情，強度に分解してリストにする
subjs = []
utts = []
emos = []
strs = []
for name in names:
    subjs.append(name[:3])
    utts.append(int(name[3:7]))
    emos.append(name[7:10])
    strs.append(int(name[10]))

# 出力用DataFrameを生成
df_out = pd.concat([
    pd.DataFrame(dict(subject=subjs,utterance=utts,emotion=emos,strength=strs)),
    values
], axis=1)

# CSVとpickleに出力
df_out.to_csv('data/data.csv')
df_out.to_pickle('data/data.df.pkl')

