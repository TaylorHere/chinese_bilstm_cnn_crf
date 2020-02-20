# -*- coding: utf-8 -*-
"""
Created on Fri May  4 21:21:56 2018

@author: shen1994
"""
import os
import logging
import pickle

logger = logging.getLogger(__name__)
import argparse
import gensim
from data_preprocess import DataPreprocess
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, EarlyStopping, TensorBoard
from data_generate import generate_batch

from paths import TrainPath

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_dir", help="train directory", default="/home/jovyan/shared/", type=str
)
parser.add_argument("--max_len", default=306, type=int)
parser.add_argument("--documents_length", default=2531574, type=int)
parser.add_argument("--label_2_index_length", default=2531574, type=int)
parser.add_argument("--batch_size", help="batch size", default=256, type=int)
parser.add_argument("--epochs", help="epochs", default=3, type=int)
args = parser.parse_args()

trainPath = TrainPath(args.train_dir)
dataPreprocess = DataPreprocess(trainPath)


from log import setUpLogger

setUpLogger(trainPath)

embedding_model = gensim.models.Word2Vec.load(trainPath.model_vector_path)
word_dict = dataPreprocess.create_useful_words(embedding_model)
logger.info("step-4--->" + u"对语料中的词统计排序生成索引" + "--->START")
lexicon, lexicon_reverse = dataPreprocess.create_lexicon(word_dict)

logger.info("step-9--->" + u"模型创建" + "--->START")
from bilstm_cnn_crf import BiLSTM_CNN_CRF

embedding_model = gensim.models.Word2Vec.load(trainPath.model_vector_path)
embedding_size = embedding_model.vector_size
useful_word_length, embedding_weights = dataPreprocess.create_embedding(
    embedding_model, embedding_size, lexicon_reverse
)

model = BiLSTM_CNN_CRF(
    args.max_len,
    useful_word_length + 2,
    args.label_2_index_length,
    embedding_size,
    embedding_weights,
)
logger.info("setp-9.1--->" + "加载模型" + "--->START")
if os.path.exists(trainPath.checkpoints_path):
    model.load_weights(trainPath.checkpoints_path)
logger.info("step-10--->" + u"模型训练" + "--->START")


checkpoint = ModelCheckpoint(
    trainPath.checkpoints_path,
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    mode="max",
)

earlyStopping = EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=0,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)
tensorBoard = TensorBoard(
    log_dir=trainPath.traning_log_basepath,
    histogram_freq=0,
    batch_size=args.batch_size,
    write_graph=True,
    write_grads=False,
    write_images=False,
    embeddings_freq=0,
    embeddings_layer_names=None,
    embeddings_metadata=None,
    embeddings_data=None,
    update_freq="epoch",
)

_ = model.fit_generator(
    generator=generate_batch(
        trainPath=trainPath,
        batch_size=args.batch_size,
        label_class=args.label_2_index_length,
    ),
    steps_per_epoch=int(args.documents_length / args.batch_size),
    epochs=args.epochs,
    verbose=1,
    workers=1,
    callbacks=[checkpoint, tensorBoard, earlyStopping],
)

logger.info("step-11--->" + u"模型和字典保存" + "--->START")

model.save_weights(trainPath.weights_path)

index_2_label = dataPreprocess.create_index_label()

pickle.dump([lexicon, index_2_label], open(trainPath.lexicon_path, "wb"))

pickle.dump(
    [args.max_len, embedding_size, useful_word_length + 2, args.label_2_index_length],
    open(trainPath.model_params_path, "wb"),
)

logger.info("step-12--->" + u"打印恢复模型的重要参数" + "--->START")

logger.info("sequence_max_length: " + str(args.max_len))

logger.info("embedding size: " + str(embedding_size))

logger.info("useful_word_length: " + str(useful_word_length + 2))

logger.info("label_2_index_length: " + str(args.label_2_index_length))

logger.info(u"训练完成" + "--->OK")
