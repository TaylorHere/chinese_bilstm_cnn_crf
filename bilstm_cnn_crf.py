# -*- coding: utf-8 -*-
"""
Created on Fri May  4 10:18:27 2018

@author: shen1994
"""

from keras.layers import Input
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Dropout
from keras.layers import ZeroPadding1D
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Concatenate

from keras_contrib.layers import CRF
from keras_contrib.losses.crf_losses import crf_loss
from keras_contrib.metrics.crf_accuracies import crf_viterbi_accuracy, crf_accuracy
from keras.models import Model


def BiLSTM_CNN_CRF(
    input_length,
    input_dim,
    class_label_count,
    embedding_size,
    embedding_weights=None,
    is_train=True,
):
    word_input = Input(shape=(input_length,), dtype="int32", name="word_input")

    if is_train:
        word_emb = Embedding(
            input_dim=input_dim,
            output_dim=embedding_size,
            input_length=input_length,
            weights=[embedding_weights],
            name="word_emb",
        )(word_input)
    else:
        word_emb = Embedding(
            input_dim=input_dim,
            output_dim=embedding_size,
            input_length=input_length,
            name="word_emb",
        )(word_input)

    # bilstm
    bilstm = Bidirectional(LSTM(64, return_sequences=True))(word_emb)
    bilstm_drop = Dropout(0.1)(bilstm)
    bilstm_dense = TimeDistributed(Dense(embedding_size))(bilstm_drop)

    # cnn
    half_window_size = 2
    filter_kernel_number = 64
    padding_layer = ZeroPadding1D(padding=half_window_size)(word_emb)
    conv = Conv1D(
        nb_filter=filter_kernel_number,
        filter_length=2 * half_window_size + 1,
        padding="valid",
    )(padding_layer)
    conv_drop = Dropout(0.1)(conv)
    conv_dense = TimeDistributed(Dense(filter_kernel_number))(conv_drop)

    # merge
    rnn_cnn_merge = Concatenate(axis=2)([bilstm_dense, conv_dense])
    dense = TimeDistributed(Dense(class_label_count))(rnn_cnn_merge)

    # crf
    crf = CRF(class_label_count, sparse_target=False)
    crf_output = crf(dense)

    # mdoel
    model = Model(input=[word_input], output=crf_output)
    model.compile(loss=crf_loss, optimizer="adam", metrics=[crf_accuracy, crf_viterbi_accuracy])

    return model
