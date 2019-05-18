import time
import argparse
import sys
sys.setrecursionlimit(10000)
import random
random.seed(666)

import numpy as np
np.random.seed(666)
import _dynet as dy

import models
from lib import *


def test(parser, testing_data, evalb_dir, unsupervised=False):
    test_predicted = []
    for data in testing_data:
        dy.renew_cg()
        predicted = parser.parse(data, False)
        test_predicted.append(predicted)

    if unsupervised:
        test_predicted_errs = [x[0] for x in test_predicted]
        test_predicted_span_sets = [x[1] for x in test_predicted]
        test_fscore = evaluate.evalb_US(testing_data, test_predicted_span_sets)
        test_ppl = evaluate.evalb_ppl(testing_data, test_predicted_errs)
        return test_fscore, test_ppl
    else:
        test_fscore = evaluate.evalb(evalb_dir, testing_data, test_predicted)
        # test_fscore = evaluate.evalb_tag(testing_data, test_predicted)
        return test_fscore


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--config_file', default='configs/InOrderParser.cfg')
    argparser.add_argument('--model', default='InOrderParser')
    argparser.add_argument('--dev_fscore', required=True)
    argparser.add_argument('--unsupervised', action="store_true")
    args, extra_args = argparser.parse_known_args()
    args.config_file = "configs/{}.cfg".format(
        args.model[:args.model.find('Parser') + 6])
    config = Configurable(args.config_file, extra_args)

    dyparams = dy.DynetParams()
    # dyparams.from_args()
    # dyparams.set_autobatch(True)
    dyparams.set_random_seed(666)
    dyparams.set_mem(10240)
    dyparams.init()

    model = dy.ParameterCollection()
    model_path = config.load_model_path + \
        args.model + "_dev={}".format(args.dev_fscore)

    [parser] = dy.load(model_path, model)
    print("Loaded model from {}".format(model_path))

    testing_data = parser.vocab.gold_data_from_file(config.test_file)
    print("Loaded testing data from {}".format(config.test_file))

    start_time = time.time()
    test_fscore = test(parser, testing_data,
                       config.evalb_dir, args.unsupervised)
    print(
        "test-fscore {} "
        "test-elapsed {}".format(
            test_fscore,
            format_elapsed(start_time),
        )
    )
