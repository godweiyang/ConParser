import math
import os.path
import re
import subprocess
import tempfile

import numpy as np


class FScore(object):

    def __init__(self, correct=0, predcount=0, goldcount=0):
        self.correct = correct        # correct brackets
        self.predcount = predcount    # total predicted brackets
        self.goldcount = goldcount    # total gold brackets

    def precision(self):
        if self.predcount > 0:
            return (100.0 * self.correct) / self.predcount
        else:
            return 0.0

    def recall(self):
        if self.goldcount > 0:
            return (100.0 * self.correct) / self.goldcount
        else:
            return 0.0

    def fscore(self):
        precision = self.precision()
        recall = self.recall()
        if (precision + recall) > 0:
            return (2 * precision * recall) / (precision + recall)
        else:
            return 0.0

    def __str__(self):
        return "(R= {:.2f}, P= {:.2f}, F= {:.2f})".format(
            self.recall(), self.precision(), self.fscore())

    def __iadd__(self, other):
        self.correct += other.correct
        self.predcount += other.predcount
        self.goldcount += other.goldcount
        return self

    def __add__(self, other):
        return Fmeasure(self.correct + other.correct,
                        self.predcount + other.predcount,
                        self.goldcount + other.goldcount)

    def __cmp__(self, other):
        return cmp(self.fscore(), other.fscore())


def evalb(gold_data, predicted_trees):
    gold_trees = [data['tree'] for data in gold_data]
    result = FScore()
    assert len(gold_trees) == len(predicted_trees)
    for gold_tree, predicted_tree in zip(gold_trees, predicted_trees):
        result += predicted_tree.compare(gold_tree)
    return result


def evalb_US(gold_data, predicted_graphs):
    gold_trees = [data['tree'] for data in gold_data]
    assert len(gold_trees) == len(predicted_graphs)

    predicted_right_count = 0
    gold_span_count = 0
    predicted_span_count = 0
    for gold_tree, predicted_graph in zip(gold_trees, predicted_graphs):
        assert len(gold_tree.sentence) == len(predicted_graph[0])
        length = len(gold_tree.sentence)
        for left in range(length):
            for right in range(left, length):
                if left == right:
                    predicted_right_count += 1
                    gold_span_count += 1
                    predicted_span_count += 1
                else:
                    label, crossing = gold_tree.span_labels(left, right)
                    label = label[::-1]
                    if (len(label) > 0):
                        gold_span_count += 1
                        if predicted_graph[left][right] > 0.8:
                            predicted_right_count += 1
                    if predicted_graph[left][right] > 0.8:
                        predicted_span_count += 1

    recall = float(100.0 * predicted_right_count / gold_span_count)
    precision = float(100.0 * predicted_right_count / predicted_span_count)
    fscore = float(2.0 * recall * precision / (recall + precision))

    return FScore(recall, precision, fscore)


def evalb_tag(gold_data, predicted_tags):
    gold_tags = [data['t'][1:-1] for data in gold_data]
    assert len(gold_tags) == len(predicted_tags)

    predicted_right_count = 0
    gold_tag_count = 0
    predicted_tag_count = 0
    for gold_tag, predicted_tag in zip(gold_tags, predicted_tags):
        assert len(gold_tag) == len(predicted_tag)
        for i in range(len(gold_tag)):
            if gold_tag[i] == predicted_tag[i]:
                predicted_right_count += 1
            gold_tag_count += 1
            predicted_tag_count += 1

    recall = float(100.0 * predicted_right_count / gold_tag_count)
    precision = float(100.0 * predicted_right_count / predicted_tag_count)
    fscore = float(2.0 * recall * precision / (recall + precision))

    return FScore(recall, precision, fscore)


def evalb_ppl(gold_data, neg_log_probs):
    total_length = 0
    for data in gold_data:
        total_length += len(data['w']) - 2

    sum_neg_log_probs = np.sum(neg_log_probs)

    ppl = np.exp(sum_neg_log_probs / total_length)

    return ppl
