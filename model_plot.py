# -*- coding: utf-8 -*-
"""
Created on Fri May  4 21:07:51 2018

@author: shen1994
"""

import pickle

from keras.utils import plot_model

from bilstm_cnn_crf import bilstm_cnn_crf

import argparse

from paths import TrainPath

import logging

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dir", help="train directory", default="/home/jovyan/shared/", type=str
    )
    parser.add_argument(
        "--to_file",
        default="/home/jovyan/shared/bilstm_cnn_crf_model.png",
        help="output file paht",
    )

    args = parser.parse_args()
    trainPath = TrainPath(args.train_dir)
    sequence_max_length, embedding_size, useful_word_length, label_2_index_length = pickle.load(
        open(trainPath.model_params_path, "rb")
    )

    model = bilstm_cnn_crf(
        sequence_max_length,
        useful_word_length,
        label_2_index_length,
        embedding_size,
        is_train=False,
    )

    model.load_weights(trainPath.weights_path)

    plot_model(model, to_file=args.to_file, show_shapes=True, show_layer_names=True)

    log.info(u"模型绘制完成" + "--------------OK")
