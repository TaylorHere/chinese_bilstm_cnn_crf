# -*- coding: utf-8 -*-
"""
Created on Fri May  4 21:21:56 2018

@author: shen1994
"""

import os
import argparse
import gensim
import pickle
import logging

logger = logging.getLogger(__name__)

from log import setUpLogger

from data_create import create_label_data, path_flatten

from data_preprocess import DataPreprocess

from data_generate import generate_batch

from bilstm_cnn_crf import bilstm_cnn_crf

from keras.callbacks import ModelCheckpoint

from paths import TrainPath

parser = argparse.ArgumentParser()
parser.add_argument("--corpus_path", help="corpus path", default="/home/jovyan/shared/corpus/2014/", type=str)
parser.add_argument("--batch_size", help="batch size", default=256, type=int)
parser.add_argument("--epochs", help="epochs", default=3, type=int)
parser.add_argument("--use_cache_train_data", default=false, type=bool)
parser.add_argument("--max_len", default=306, type=int)
parser.add_argument("--documents_length", default=2531574, type=int)
parser.add_argument(
    "--train_dir", help="train directory", default="/home/jovyan/shared/", type=str
)
args = parser.parse_args()

corpus_path = args.corpus_path
batch_size = args.batch_size
epochs = args.epochs

trainPath = TrainPath(args.train_dir)

setUpLogger(trainPath)
dataPreprocess = DataPreprocess(trainPath)


def run():

    logger.info("step-1--->" + u"加载词向量模型" + "--->START")
    embedding_model = gensim.models.Word2Vec.load(trainPath.model_vector_path)

    word_dict = dataPreprocess.create_useful_words(embedding_model)

    embedding_size = embedding_model.vector_size



    if not args.use_cache_train_data:

        logger.info("step-2--->" + u"语料格式转换,加标注生成标准文件" + "--->START")
        
        raw_train_file = path_flatten(corpus_path)
        create_label_data(trainPath, word_dict, raw_train_file)

        logger.info("step-3--->" + u"按标点符号或是空格存储文件" + "--->START")
        
        documents_length = dataPreprocess.create_documents()
    documents_length = args.documents_length
    logger.info("step-4--->" + u"对语料中的词统计排序生成索引" + "--->START")

    lexicon, lexicon_reverse = dataPreprocess.create_lexicon(word_dict)

    logger.info("step-5--->" + u"对所有的词创建词向量" + "--->START")

    useful_word_length, embedding_weights = dataPreprocess.create_embedding(
        embedding_model, embedding_size, lexicon_reverse
    )

    logger.info("step-6--->" + u"生成标注以及索引" + "--->START")

    label_2_index = dataPreprocess.create_label_index()

    label_2_index_length = len(label_2_index)

    logger.info("step-7--->" + u"将语料中每一句和label进行索引编码" + "--->START")
    if not args.use_cache_train_data:
        dataPreprocess.create_matrix(lexicon, label_2_index)

    logger.info("step-8--->" + u"将语料中每一句和label以最大长度统一长度,不足补零" + "--->START")
    if not args.max_len:
        max_len = dataPreprocess.maxlen_2d_list()

    if not args.use_cache_train_data:
        dataPreprocess.padding_sentences(max_len)

    logger.info("step-9--->" + u"模型创建" + "--->START")

    model = bilstm_cnn_crf(
        max_len,
        useful_word_length + 2,
        label_2_index_length,
        embedding_size,
        embedding_weights,
    )
    logger.info("setp-9.1--->" + "加载模型" + "--->START")
    if os.path.exists(trainPath.checkpoints_path):
        model.load_weights(trainPath.checkpoints_path)
    logger.info("step-10--->" + u"模型训练" + "--->START")

    if batch_size > documents_length:

        logger.info("ERROR--->" + u"语料数据量过少，请再添加一些")

        return None

    checkpoint = ModelCheckpoint(
        trainPath.checkpoints_path,
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        mode="max",
    )

    _ = model.fit_generator(
        generator=generate_batch(
            trainPath=trainPath,
            batch_size=batch_size, label_class=label_2_index_length
        ),
        steps_per_epoch=int(documents_length / batch_size),
        epochs=epochs,
        verbose=1,
        workers=1,
        callbacks=[checkpoint],
    )

    logger.info("step-11--->" + u"模型和字典保存" + "--->START")

    model.save_weights(trainPath.weights_path)

    index_2_label = dataPreprocess.create_index_label()

    pickle.dump([lexicon, index_2_label], open(trainPath.lexicon_path, "wb"))

    pickle.dump(
        [max_len, embedding_size, useful_word_length + 2, label_2_index_length],
        open(trainPath.model_params_path, "wb"),
    )

    logger.info("step-12--->" + u"打印恢复模型的重要参数" + "--->START")

    logger.info("sequence_max_length: " + str(max_len))

    logger.info("embedding size: " + str(embedding_size))

    logger.info("useful_word_length: " + str(useful_word_length + 2))

    logger.info("label_2_index_length: " + str(label_2_index_length))

    logger.info(u"训练完成" + "--->OK")


if __name__ == "__main__":
    run()
