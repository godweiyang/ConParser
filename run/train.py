import argparse
import sys
sys.setrecursionlimit(10000)
import time
import os
import itertools
import random
random.seed(666)

import numpy as np
np.random.seed(666)
import _dynet as dy
dyparams = dy.DynetParams()
# dyparams.from_args()
dyparams.set_autobatch(True)
dyparams.set_random_seed(666)
dyparams.set_mem(2000)
dyparams.init()

import models
from lib import *
from test import *


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--config_file', default='configs/InOrderParser.cfg')
    argparser.add_argument('--model', default='InOrderParser')
    args, extra_args = argparser.parse_known_args()
    args.config_file = "configs/{}.cfg".format(args.model)
    config = Configurable(args.config_file, extra_args)

    vocab = Vocabulary(config.train_file)
    training_data = vocab.gold_data_from_file(config.train_file)
    print('Loaded {} training sentences!'.format(len(training_data)))
    deving_trees = PhraseTree.load_treefile(config.dev_file)
    print('Loaded {} validation trees!'.format(len(deving_trees)))

    model = dy.ParameterCollection()
    trainer = dy.AdadeltaTrainer(model, eps=1e-7, rho=0.99)
    # trainer = dy.AdamTrainer(model)
    # trainer.set_sparse_updates(False)

    Parser = getattr(models, args.model)
    parser = Parser(
        model,
        vocab,
        config.word_embedding_dim,
        config.tag_embedding_dim,
        config.char_embedding_dim,
        config.label_embedding_dim,
        config.pos_embedding_dim,
        config.char_lstm_layers,
        config.char_lstm_dim,
        config.lstm_layers,
        config.lstm_dim,
        config.fc_hidden_dim,
        config.dropout,
        config.unk_param,
    )

    current_processed = 0
    check_every = len(training_data) / config.checks_per_epoch
    best_dev_fscore = FScore(-np.inf, -np.inf, -np.inf)
    best_dev_epoch = 0
    best_dev_model_path = None

    start_time = time.time()

    for epoch in itertools.count(start=1):
        if config.epochs is not None and epoch > config.epochs:
            model_path_finished = "{}{}_{}epoch".format(
                config.save_model_path, args.model, config.epochs)
            print("    [Saving {} epochs model to {}]".format(
                config.epochs, model_path_finished))
            dy.save(model_path_finished, [parser])
            break

        print('........... epoch {} ...........'.format(epoch))
        np.random.shuffle(training_data)
        total_loss_count = 0
        total_loss_value = 0.0

        for start_index in range(0, len(training_data), config.batch_size):
            batch_losses = []
            dy.renew_cg()

            for data in training_data[start_index:start_index + config.batch_size]:
                loss, _, loss_counts = parser.parse(data)
                batch_losses.append(loss)
                current_processed += 1
                total_loss_count += loss_counts

            batch_loss = dy.esum(batch_losses)
            total_loss_value += batch_loss.scalar_value()
            batch_loss.backward()
            trainer.update()

            print(
                "\r"
                "batch {:,}/{:,}  "
                "mean-loss {:.8f}  "
                "total-elapsed {}  ".format(
                    start_index // config.batch_size + 1,
                    int(np.ceil(len(training_data) / config.batch_size)),
                    total_loss_value / total_loss_count,
                    format_elapsed(start_time)
                ),
                end=""
            )
            sys.stdout.flush()

            if current_processed >= check_every:
                current_processed -= check_every
                dev_fscore = test(parser, deving_trees, config.evalb_dir)
                print("[Dev: {}]".format(dev_fscore))
                if dev_fscore.fscore >= best_dev_fscore.fscore:
                    if best_dev_model_path is not None:
                        for ext in [".data", ".meta"]:
                            path = best_dev_model_path + ext
                            if os.path.exists(path):
                                os.remove(path)

                    best_dev_fscore = dev_fscore
                    best_dev_epoch = epoch
                    best_dev_model_path = "{}{}_dev={:.2f}".format(
                        config.save_model_path, args.model, dev_fscore.fscore)
                    print("    [Saving new best model to {}]".format(
                        best_dev_model_path))
                    dy.save(best_dev_model_path, [parser])

        print("[Best dev: {}, best epoch {}]".format(
            best_dev_fscore, best_dev_epoch))
