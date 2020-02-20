import os
import argparse
import logging
import gensim

logger = logging.getLogger(__name__)

from log import setUpLogger
from data_create import create_label_data, path_flatten
from data_preprocess import DataPreprocess
from paths import TrainPath

parser = argparse.ArgumentParser()
parser.add_argument(
    "--corpus_path",
    help="corpus path",
    default="/home/jovyan/shared/corpus/2014/",
    type=str,
)
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

logger.info("step-1--->" + u"加载词向量模型" + "--->START")
embedding_model = gensim.models.Word2Vec.load(trainPath.model_vector_path)
word_dict = dataPreprocess.create_useful_words(embedding_model)

logger.info("step-2--->" + u"语料格式转换,加标注生成标准文件" + "--->START")
raw_train_file = path_flatten(corpus_path)
create_label_data(trainPath, word_dict, raw_train_file)

logger.info("step-3--->" + u"按标点符号或是空格存储文件" + "--->START")
documents_length = dataPreprocess.create_documents()

logger.info("step-4--->" + u"对语料中的词统计排序生成索引" + "--->START")
lexicon, lexicon_reverse = dataPreprocess.create_lexicon(word_dict)

logger.info("step-6--->" + u"生成标注以及索引" + "--->START")
label_2_index = dataPreprocess.create_label_index()
label_2_index_length = len(label_2_index)

logger.info("step-7--->" + u"将语料中每一句和label进行索引编码" + "--->START")
dataPreprocess.create_matrix(lexicon, label_2_index)

logger.info("step-8--->" + u"将语料中每一句和label以最大长度统一长度,不足补零" + "--->START")
max_len = dataPreprocess.maxlen_2d_list()
dataPreprocess.padding_sentences(max_len)
