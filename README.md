# 藤江研 B3 課題

## 環境設定（anaconda)

silver2～silver8で実行してください．

```
$ conda env create -n b3kadai -f=b3kadai.yaml
$ conda acivate b3kadai
$ pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu113
```

## 実行

### 音声

[参考資料](https://docs.google.com/document/d/1FCUTbIILlcthU7VeDnKPg1t6RL87BxZUEAnUrZnj5Pc/edit?usp=sharing)

作業フォルダに移動．

```
$ cd speech
```

[OGVC](http://research.nii.ac.jp/src/OGVC.html)の演技音声から，[openSMILE](https://audeering.github.io/opensmile-python/)で特徴量を抽出して```data.csv```および```data.df.pkl```に保存．

```
$ python prepare_data.py
```

用意したデータを用いてニューラルネットワークを学習し，パラメータを```model_params.pth```に保存．ログを```train_log.csv```に保存．

```
$ python train.py
```

学習されたニューラルネットワークを用いて，テストデータで評価する．
適合率，再現率，F1値の表示と，混同行列の表示を行う．

```
$ python eval.py
```

### 言語

[参考資料](https://docs.google.com/document/d/1QfsIuCT8P03wCcLlRonOhhHxy8hKkRxwHBw4m0zcb6s/edit?usp=sharing)

作業フォルダに移動．

```
$ cd lang
```

[Twitter日本語評判分析データセット](http://www.db.info.gifu-u.ac.jp/sentiment_analysis/)のツイートデータに対して，[NICT BERT 日本語 Pre-trained モデル](https://alaginrc.nict.go.jp/nict-bert/index.html)の「BPEあり、32,000語」のモデルで特徴抽出を行い，```extracted_features_32k.df.pkl```に保存．

```
$ python prepare_data.py
```

用意したデータを用いてニューラルネットワークを学習し，パラメータを```model_params.pth```に保存．ログを```train_log.csv```に保存．

```
$ python train.py
```

学習されたニューラルネットワークを用いて，テストデータで評価する．
適合率，再現率，F1値の表示と，混同行列の表示を行う．

```
$ python eval.py
```

### 画像

[参考資料](https://docs.google.com/document/d/1UMtkyr-D404xT4n1bG0V74JCoSAMk8IxQZddSe6U1zs/edit?usp=sharing)

作業フォルダに移動．

```
$ cd image
```

[Cohn-Kanadeデータセット](https://paperswithcode.com/dataset/ck)の顔画像に対して，[dlib](http://dlib.net/)でランドマーク検出を行い，```data.csv```および```data.df.pkl```に保存．

```
$ python prepare_data.py
```

用意したデータを用いてニューラルネットワークを学習し，パラメータを```model_params.pth```に保存．ログを```train_log.csv```に保存．

```
$ python train.py
```

学習されたニューラルネットワークを用いて，テストデータで評価する．
適合率，再現率，F1値の表示と，混同行列の表示を行う．

```
$ python eval.py
```






