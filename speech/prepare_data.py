import numpy as np
import pandas as pd
import opensmile
import glob
import os
import tqdm

smile = opensmile.Smile(feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals)

wavfile_list = glob.glob('/autofs/diamond/share/corpus/OGVC/Vol2/Acted/wav/*/*/*.wav')

name_list = []
features_list = []
for wavfile_path in tqdm.tqdm(wavfile_list):
    wavfile_basename = os.path.basename(wavfile_path)
    wavfile_body, _ = os.path.splitext(wavfile_basename)
    name_list.append(wavfile_body)
    features = smile.process_file(wavfile_path)
    features_list.append(features.to_numpy())
column_names = features.columns
df_features = pd.DataFrame(np.concatenate(features_list), columns=column_names)
df_names = pd.DataFrame(name_list, columns=["name"])
df = pd.concat([df_names, df_features], axis=1)

# openSMILEで書き出されたデータをDataFrame化
# df = pd.read_csv('extracted_features.csv', sep=';')

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
df_out.to_csv('data.csv')
df_out.to_pickle('data.df.pkl')

