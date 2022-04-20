import json
import pandas as pd
from os import environ
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel, BertForQuestionAnswering
from mojimoji import han_to_zen
import tqdm
import numpy as np
import MeCab
import sys

# 元のJSONファイルを読み込む
# データの詳細については http://www.db.info.gifu-u.ac.jp/sentiment_analysis/
json_path='/autofs/diamond2/share/users/fujie/share/twitterJSA_data.json'
data_raw = json.load(open(json_path))

# DataFrameを生成（まず列ごとの値を持つリストを作成する）
ids = []
topics = []
statuss = []
label_pos_negs = []
label_poss = []
label_negs = []
label_neus = []
label_dont_cares = []
texts = []
for entry in data_raw:
    if len(entry['text']) == 0:
        continue
    ids.append(entry['id'])
    topics.append(entry['topic'])
    statuss.append(entry['status'])
    # フラグについてはひとつのリストにまとまってるので分解する
    pos_neg, pos, neg, neu, dont_care = entry['label']
    label_pos_negs.append(pos_neg)
    label_poss.append(pos)
    label_negs.append(neg)
    label_neus.append(neu)
    label_dont_cares.append(dont_care)
    texts.append(entry['text'])

# 実際のデータフレームの生成
df = pd.DataFrame(dict(id=ids, topic=topics, status=statuss, 
    pos_neg=label_pos_negs, pos=label_poss, neg=label_negs,
    neu=label_neus, dont_care=label_dont_cares, text=texts))

# CSVファイルとPickleの出力
df.to_csv('data/data.csv')
df.to_pickle('data/data.df.pickle')

# Mecab Tagger の準備
jumandic_dir = "/autofs/diamond2/share/users/fujie/share/mecab-jumandic-7.0-20130310"
environ["MECABRC"] = "/etc/mecabrc"
tagger_jumandic = MeCab.Tagger(f"-Owakati -d{jumandic_dir}")
def tokenize_func(text):
    return tagger_jumandic.parse(han_to_zen(text).replace("\u3000", " ")).rstrip("\n")

# BERTトークナイザとBERTモデルの準備
bert_model_dir = "/autofs/diamond2/share/users/fujie/share/NICT_BERT-base_JapaneseWikipedia_32K_BPE" 
tokenizer = BertTokenizer.from_pretrained(bert_model_dir)
model = BertModel.from_pretrained(bert_model_dir)
model.to('cuda')
model.eval()

# 文（のリスト）を特徴ベクトル（を並べた行列）に変換する関数
def convert(batch_sentences):
    processed_batch_sentences = [tokenize_func(text) for text in batch_sentences]
    batch = tokenizer(processed_batch_sentences, padding=True, return_tensors="pt")
    input_ids = batch['input_ids'].to('cuda')
    with torch.no_grad():
        outputs = model(input_ids)
    pooler_output = outputs.pooler_output
    return pooler_output.detach().cpu().numpy()

# 30文ずつ特徴量に変換していく
batch_size = 100
feature_list = []
for i in tqdm.tqdm(range((df.shape[0] - 1) // batch_size + 1)):
    idx_start = i * batch_size
    idx_end = (i + 1) * batch_size
    if idx_end > df.shape[0]:
        idx_end = df.shape[0]
    batch_sentences = df.iloc[idx_start:idx_end, :]['text'].tolist()
    feature = convert(batch_sentences)
    feature_list.append(feature)

# 変換した特徴量を一つの array にまとめる
feature_array = np.concatenate(feature_list, axis=0)

# 特徴量次元
num_feature_dims = feature_array.shape[1]
# カラム名を生成（x0, x1, x2 など）
columns = ['x{}'.format(i) for i in range(num_feature_dims)]

# 特徴量の DataFrame を生成
feature_df = pd.DataFrame(feature_array, columns=columns)

# 元の DataFrame と特徴量のDataFrameを結合
df_all = pd.concat([df, feature_df], axis=1)

# 書き出し（CSVは時間がかかりすぎるので廃止）
# df_all.to_csv('extracted_features_32k.csv')
df_all.to_pickle('data/extracted_features_32k.df.pkl')
