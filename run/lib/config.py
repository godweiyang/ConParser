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

    @property
    def train_bert_file(self):
        return self._config.get('Data', 'train_bert_file')

    @property
    def dev_bert_file(self):
        return self._config.get('Data', 'dev_bert_file')

    @property
    def test_bert_file(self):
        return self._config.get('Data', 'test_bert_file')

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
    

    # Optimizer
    @property
    def clip_grad_norm(self):
        return self._config.getint('Optimizer', 'clip_grad_norm')

    @property
    def learning_rate(self):
        return self._config.getfloat('Optimizer', 'learning_rate')

    @property
    def learning_rate_warmup_steps(self):
        return self._config.getint('Optimizer', 'learning_rate_warmup_steps')

    @property
    def step_decay(self):
        return self._config.getboolean('Optimizer', 'step_decay')

    @property
    def step_decay_factor(self):
        return self._config.getfloat('Optimizer', 'step_decay_factor')

    @property
    def step_decay_patience(self):
        return self._config.getint('Optimizer', 'step_decay_patience')

    # Run
    @property
    def random_seed(self):
        return self._config.getint('Run', 'random_seed')

    @property
    def numpy_seed(self):
        return self._config.getint('Run', 'numpy_seed')

    @property
    def torch_seed(self):
        return self._config.getint('Run', 'torch_seed')

    @property
    def batch_size(self):
        return self._config.getint('Run', 'batch_size')

    @property
    def epochs(self):
        return self._config.getint('Run', 'epochs')

    @property
    def checks_per_epoch(self):
        return self._config.getint('Run', 'checks_per_epoch')

    @property
    def subbatch_max_tokens(self):
        return self._config.getint('Run', 'subbatch_max_tokens')
