from configparser import SafeConfigParser
import os


class Configurable(object):
    def __init__(self, config_file, extra_args):
        config = SafeConfigParser()
        config.read(config_file)
        if extra_args:
            extra_args = dict([(k[2:], v) for k, v in zip(
                extra_args[0::2], extra_args[1::2])])
        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)
        self._config = config
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        print('Loaded config file sucessfully.')

    # Data
    @property
    def data_dir(self):
        return self._config.get('Data', 'data_dir')

    @property
    def train_file(self):
        return self._config.get('Data', 'train_file')

    @property
    def dev_file(self):
        return self._config.get('Data', 'dev_file')

    @property
    def test_file(self):
        return self._config.get('Data', 'test_file')

    # Save
    @property
    def save_dir(self):
        return self._config.get('Save', 'save_dir')

    @property
    def save_model_path(self):
        return self._config.get('Save', 'save_model_path')

    @property
    def load_dir(self):
        return self._config.get('Save', 'load_dir')

    @property
    def load_model_path(self):
        return self._config.get('Save', 'load_model_path')

    @property
    def evalb_dir(self):
        return self._config.get('Save', 'evalb_dir')

    # Network
    @property
    def word_embedding_dim(self):
        return self._config.getint('Network', 'word_embedding_dim')

    @property
    def tag_embedding_dim(self):
        return self._config.getint('Network', 'tag_embedding_dim')

    @property
    def char_embedding_dim(self):
        return self._config.getint('Network', 'char_embedding_dim')

    @property
    def label_embedding_dim(self):
        return self._config.getint('Network', 'label_embedding_dim')

    @property
    def pos_embedding_dim(self):
        return self._config.getint('Network', 'pos_embedding_dim')

    @property
    def char_lstm_layers(self):
        return self._config.getint('Network', 'char_lstm_layers')

    @property
    def char_lstm_dim(self):
        return self._config.getint('Network', 'char_lstm_dim')

    @property
    def lstm_layers(self):
        return self._config.getint('Network', 'lstm_layers')

    @property
    def lstm_dim(self):
        return self._config.getint('Network', 'lstm_dim')

    @property
    def fc_hidden_dim(self):
        return self._config.getint('Network', 'fc_hidden_dim')

    @property
    def dropout(self):
        return self._config.getfloat('Network', 'dropout')

    @property
    def unk_param(self):
        return self._config.getfloat('Network', 'unk_param')

    # Run
    @property
    def numpy_seed(self):
        return self._config.getint('Run', 'numpy_seed')

    @property
    def batch_size(self):
        return self._config.getint('Run', 'batch_size')

    @property
    def epochs(self):
        return self._config.getint('Run', 'epochs')

    @property
    def checks_per_epoch(self):
        return self._config.getint('Run', 'checks_per_epoch')
