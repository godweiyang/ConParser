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


def test(parser, test_trees, evalb_dir):
    test_predicted = []
    for tree in test_trees:
        dy.renew_cg()
        predicted = parser.predict(tree)
        test_predicted.append(predicted)

    test_fscore = evaluate.evalb(evalb_dir, test_trees, test_predicted)
    return test_fscore


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--config_file', default='configs/ShiftReduceParser.cfg')
    argparser.add_argument('--model', default='ShiftReduceParser')
    argparser.add_argument('--dev_fscore', required=True)
    args, extra_args = argparser.parse_known_args()
    args.config_file = "configs/{}.cfg".format(args.model)
    config = Configurable(args.config_file, extra_args)

    dyparams = dy.DynetParams()
    # dyparams.from_args()
    dyparams.set_autobatch(True)
    dyparams.set_random_seed(666)
    dyparams.set_mem(2000)
    dyparams.init()

    testing_trees = PhraseTree.load_treefile(config.test_file)
    print("Loaded testing trees from {}".format(config.test_file))

    model = dy.ParameterCollection()
    model_path = config.load_model_path + \
        args.model + "_dev={}".format(args.dev_fscore)

    [parser] = dy.load(model_path, model)
    print("Loaded model from {}".format(model_path))

    start_time = time.time()
    test_fscore = test(parser, testing_trees, config.evalb_dir)

    print(
        "test-fscore {} "
        "test-elapsed {}".format(
            test_fscore,
            format_elapsed(start_time),
        )
    )
