import argparse
import sys
sys.setrecursionlimit(10000)
import time
import os
import itertools
import random

import numpy as np
import torch
import torch.optim.lr_scheduler

import models
from lib import *
from test import *

if __name__ == "__main__":
    # load the config parameters
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file',
                           default='configs/ChartParser.cfg')
    argparser.add_argument('--model', default='ChartParser')
    argparser.add_argument('--unsupervised', action="store_true")
    argparser.add_argument('--train_more', action="store_true")
    argparser.add_argument('--use_elmo', action="store_true")
    argparser.add_argument('--use_bert', action="store_true")
    argparser.add_argument('--more_epochs', type=int, default=20)
    args, extra_args = argparser.parse_known_args()
    args.config_file = "configs/{}.cfg".format(
        args.model[:args.model.find('Parser') + 6])
    config = Configurable(args.config_file, extra_args)

    # set the seeds
    random.seed(config.random_seed)
    np.random.seed(config.numpy_seed)
    torch.manual_seed(config.torch_seed)

    vocab = Vocabulary(config.train_file)
    training_data = vocab.gold_data_from_file(config.train_file)
    print('Loaded {} training data!'.format(len(training_data)))
    deving_data = vocab.gold_data_from_file(config.dev_file)
    print('Loaded {} validation data!'.format(len(deving_data)))

    if args.use_bert:
        train_bert_embeddings = vocab.load_bert_embeddings(
            config.train_bert_file)
        dev_bert_embeddings = vocab.load_bert_embeddings(config.dev_bert_file)
        print('Loaded bert embeddings!')

    if args.train_more:
        model_path = config.load_model_path + \
            args.model + "_{}epoch".format(config.epochs)
        [parser] = dy.load(model_path, model)
        print("Loaded model from {}".format(model_path))
    else:
        parameters = [
            vocab, config.word_embedding_dim, config.tag_embedding_dim,
            config.char_embedding_dim, config.label_embedding_dim,
            config.pos_embedding_dim, config.char_lstm_layers,
            config.char_lstm_dim, config.lstm_layers, config.lstm_dim,
            config.fc_hidden_dim, config.dropout, config.unk_param
        ]
        Parser = getattr(models, args.model)
        parser = Parser(parameters)

    print("Initializing optimizer...")
    trainable_parameters = [
        param for param in parser.parameters() if param.requires_grad
    ]
    trainer = torch.optim.Adam(trainable_parameters,
                               lr=1.,
                               betas=(0.9, 0.98),
                               eps=1e-9)

    def set_lr(new_lr):
        for param_group in trainer.param_groups:
            param_group['lr'] = new_lr

    assert config.step_decay, "Only step_decay schedule is supported"

    warmup_coeff = config.learning_rate / config.learning_rate_warmup_steps
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer,
        'max',
        factor=config.step_decay_factor,
        patience=config.step_decay_patience,
        verbose=True,
    )

    def schedule_lr(iteration):
        iteration = iteration + 1
        if iteration <= config.learning_rate_warmup_steps:
            set_lr(iteration * warmup_coeff)

    clippable_parameters = trainable_parameters
    grad_clip_threshold = np.inf if config.clip_grad_norm == 0 else config.clip_grad_norm

    current_processed = 0
    check_every = len(training_data) / config.checks_per_epoch
    best_dev_fscore = FScore(0, 0, 0)
    best_dev_epoch = 0
    best_dev_model_path = None

    start_time = time.time()

    if args.train_more:
        start_epoch = config.epochs + 1
        end_epoch = config.epochs + args.more_epochs
    else:
        start_epoch = 1
        end_epoch = config.epochs

    for epoch in itertools.count(start=start_epoch):
        if config.epochs is not None and epoch > end_epoch:
            model_path_finished = "{}{}_{}epoch".format(
                config.save_model_path, args.model, end_epoch)
            print("    [Saving {} epochs model to {}]".format(
                end_epoch, model_path_finished))
            dy.save(model_path_finished, [parser])
            break

        print('........... epoch {} ...........'.format(epoch))
        np.random.shuffle(training_data)
        total_loss_count = 0
        total_loss_value = 0.0

        for start_index in range(0, len(training_data), config.batch_size):
            batch_losses = []
            trainer.zero_grad()
            schedule_lr(current_processed // config.batch_size)
            parser.train()

            batch_data = training_data[start_index:start_index +
                                       config.batch_size]

            for data in split_batch(batch_data, config.subbatch_max_tokens):
                if args.use_bert:
                    loss, _, loss_counts = parser.parse_batch(
                        data, True, train_bert_embeddings[data['idx']])
                else:
                    loss, _, loss_counts = parser.parse_batch(data, True)
                batch_losses.append(loss)
                current_processed += 1
                total_loss_count += loss_counts

            batch_loss = dy.esum(batch_losses)
            total_loss_value += batch_loss.scalar_value()
            batch_loss.backward()
            trainer.update()

            print("\r"
                  "batch {:,}/{:,}  "
                  "mean-loss {:.8f}  "
                  "total-elapsed {}  ".format(
                      start_index // config.batch_size + 1,
                      int(np.ceil(len(training_data) / config.batch_size)),
                      total_loss_value / total_loss_count,
                      format_elapsed(start_time)),
                  end="")
            sys.stdout.flush()

            if current_processed >= check_every:
                current_processed -= check_every
                if args.use_bert:
                    dev_fscore = test(parser, deving_data, config.evalb_dir,
                                      args.unsupervised, dev_bert_embeddings)
                else:
                    dev_fscore = test(parser, deving_data, config.evalb_dir,
                                      args.unsupervised)
                print("[Dev: {}]".format(dev_fscore))
                if dev_fscore.fscore() >= best_dev_fscore.fscore():
                    if best_dev_model_path is not None:
                        for ext in [".data", ".meta"]:
                            path = best_dev_model_path + ext
                            if os.path.exists(path):
                                os.remove(path)

                    best_dev_fscore = dev_fscore
                    best_dev_epoch = epoch
                    best_dev_model_path = "{}{}_dev={:.2f}".format(
                        config.save_model_path, args.model,
                        dev_fscore.fscore())
                    print("    [Saving new best model to {}]".format(
                        best_dev_model_path))
                    dy.save(best_dev_model_path, [parser])

        print("[Best dev: {}, best epoch {}]".format(best_dev_fscore,
                                                     best_dev_epoch))
