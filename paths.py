import os


class TrainPath(object):
    def __init__(self, train_dir="./"):
        self.data_basepath = os.path.join(train_dir, "data")
        if not os.path.exists(self.data_basepath):
            os.makedirs(self.data_basepath)
        self.data_path = os.path.join(self.data_basepath, "data.data")
        self.label_path = os.path.join(self.data_basepath, "label.data")
        self.train_data_path = os.path.join(self.data_basepath, "train.data")
        self.data_index_path = os.path.join(self.data_basepath, "data_index.data")
        self.label_index_path = os.path.join(self.data_basepath, "label_index.data")
        self.data_index_padding_path = os.path.join(self.data_basepath, "data_index_padding.data")
        self.label_index_padding_path = os.path.join(
            self.data_basepath, "label_index_padding.data"
        )

        self.model_basepath = os.path.join(train_dir, "model")
        if not os.path.exists(self.model_basepath):
            os.makedirs(self.model_basepath)
        self.model_vector_path = os.path.join(self.model_basepath, "model_vector_people.m")
        self.model_vector_text_path = os.path.join(self.model_basepath, "model_vector_people.txt")
        self.checkpoints_basepath = os.path.join(self.model_basepath, "checkpoints")
        self.checkpoints_path = os.path.join(self.checkpoints_basepath, "weights.best.hdf5")
        self.weights_path = os.path.join(self.model_basepath, "train_model.hdf5")
        self.lexicon_path = os.path.join(self.model_basepath, "lexicon.pkl")
        self.model_params_path = os.path.join(self.model_basepath, "model_params.pkl")
