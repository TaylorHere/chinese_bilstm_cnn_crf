import os


class TrainPath(object):
    def __init__(self, train_dir="./"):
        data_basepath = os.path.join(train_dir, "data")
        if not os.path.exists(data_basepath):
            os.makedirs(data_basepath)
        data_path = os.path.join(data_basepath, "data.data")
        label_path = os.path.join(data_basepath, "label.data")
        train_data_path = os.path.join(data_basepath, "train.data")
        data_index_path = os.path.join(data_basepath, "data_index.data")
        label_index_path = os.path.join(data_basepath, "label_index.data")
        data_index_padding_path = os.path.join(data_basepath, "data_index_padding.data")
        label_index_padding_path = os.path.join(
            data_basepath, "label_index_padding.data"
        )

        model_basepath = os.path.join(train_dir, "model")
        if not os.path.exists():
            os.makedirs(model_basepath)
        model_vector_path = os.path.join(model_basepath, "model_vector_people.m")
        model_vector_text_path = os.path.join(model_basepath, "model_vector_people.txt")
        checkpoints_basepath = os.path.join(model_basepath, "checkpoints")
        checkpoints_path = os.path.join(checkpoints_basepath, "weights.best.hdf5")
        weights_path = os.path.join(model_basepath, "train_model.hdf5")
        lexicon_path = os.path.join(model_basepath, "lexicon.pkl")
        model_params_path = os.path.join(trainPath.model_basepath, "model_params.pkl")
