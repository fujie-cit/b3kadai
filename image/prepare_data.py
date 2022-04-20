from imutils.face_utils.helpers import FACIAL_LANDMARKS_IDXS
from numpy.core.fromnumeric import shape
from numpy.lib.type_check import imag
import glob
from os import path
import numpy as np
import dlib
from imutils import face_utils
import cv2
import numpy as np
import pandas as pd
import tqdm

import os, sys

def make_subj_seq_emotion_list(topdir='/autofs/diamond/share/corpus/Cohn-Kanade/CK+/Emotion'):
    """被験者名をキーとして，系列名と感情のタプルのリストを値とする辞書を構築して返す
    """
    filelist = glob.glob(path.join(topdir, '*/*/*_emotion.txt'))
    r = dict()
    for fullpath in filelist:
        # subpath = fullpath.removeprefix(topdir + '/')
        subpath = fullpath[len(topdir)+1:]
        subj, seq, _ = subpath.split('/')
        emo = int(np.loadtxt(fullpath))
        if subj in r:
            r[subj].append((seq, emo))
        else:
            r[subj] = [(seq, emo)]
    return r

def extract_shape(filename, shape_predictor=None, 
    shape_predictor_path='/autofs/diamond2/share/users/fujie/share/shape_predictor_68_face_landmarks.dat'):
    """顔画像中の形状（68点のランドマーク）を検出して返す

    顔検出された数が1でない場合はNoneを返す．
        
    """
    if shape_predictor is None:    
        shape_predictor = dlib.shape_predictor(shape_predictor_path)

    img = cv2.imread(filename)
    img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_detector = dlib.get_frontal_face_detector()
    faces = face_detector(img_gry, 1)

    if len(faces) != 1:
        return None, shape_predictor

    for face in faces:
        landmark = shape_predictor(img_gry, face)
        landmark = face_utils.shape_to_np(landmark)
        break

    return landmark, shape_predictor

def normalize_shape(landmarks):
    """ランドマークのX座標，Y座標の範囲がそれぞれ -1.0 〜 1.0 になるように正規化する
    """
    pmax = landmarks.max(axis=0)
    pmin = landmarks.min(axis=0)
    landmarks_s = (landmarks - pmin) / (pmax - pmin) * 2.0 - 1.0
    return landmarks_s

def get_image_filenamelist(subj, seq, topdir='/autofs/diamond/share/corpus/Cohn-Kanade/CK+/cohn-kanade-images'):
    """画像ファイル名のリストを作成する
    """
    filelist = sorted(glob.glob(path.join(topdir, subj, seq, '*.png')))
    return filelist

def run():
    # 被験者名をキーとして，系列名と感情のタプルのリストを値とする辞書を構築
    subj_seq_emo_list = make_subj_seq_emotion_list()

    db_subj = []
    db_seq = []
    db_emo = []
    db_index = []
    db_shape = []
    shape_predictor = None

    # 被験者ごとに繰り返す
    for subj, seq_emo_list in tqdm.tqdm(subj_seq_emo_list.items()):
        # subj ... 被験者名
        # seq_emo_list ... 系列名と感情のタプル

        # 系列番号と感情ごとに繰り返す
        for seq, emo in tqdm.tqdm(seq_emo_list):
            # seq ... 系列名
            # emo ... 感情番号

            # 系列に対応したファイル名のリスト（ソート済みのフルパス）を取得する
            image_filenamelist = get_image_filenamelist(subj, seq)

            # 系列の最初の画像（無表情）のランドマーク座標を取得する
            shape0, shape_predictor = extract_shape(image_filenamelist[0], shape_predictor)
            # ランドマーク座標が取得できなかったら次の系列に移す
            if shape0 is None:
                continue

            # 系列の最後の画像（表情顔）のランドマーク座標を取得する
            index_final = len(image_filenamelist) - 1
            shape_final, shape_predictor = extract_shape(image_filenamelist[index_final], shape_predictor)
            # ランドマーク座標が取得できなかったら次の系列に移す
            if shape_final is None:
                continue

            # 無表情の情報を保存
            db_subj.append(subj)
            db_seq.append(seq)
            db_emo.append(0)
            db_index.append(0)
            db_shape.append(normalize_shape(shape0).flatten())

            # 感情顔の情報を保存
            db_subj.append(subj)
            db_seq.append(seq)
            db_emo.append(emo)
            db_index.append(index_final)
            db_shape.append(normalize_shape(shape_final).flatten())

    # 画像情報用のDataFrameを作成（ランドマーク情報を除く）
    df_info = pd.DataFrame(dict(subject=db_subj, seq=db_seq, emotion=db_emo, image_index=db_index))

    # ランドマーク情報用のDataFrameを作成
    shape_column_names = []
    for i in range(68):
        shape_column_names.extend(['x{}'.format(i), 'y{}'.format(i)])
    df_shapes = pd.DataFrame(np.stack(db_shape), columns=shape_column_names)

    # 画像情報用とランドマーク情報用のDataFrameを結合する
    df = pd.concat([df_info, df_shapes], axis=1)

    # 作成したDataFrameをCSVファイルとPickleで書き出す
    df.to_csv('data/data.csv')
    df.to_pickle('data/data.df.pkl')

if __name__ == '__main__':
    run()
