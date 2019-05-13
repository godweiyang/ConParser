import math
import os.path
import re
import subprocess
import tempfile


class FScore(object):
    def __init__(self, recall, precision, fscore):
        self.recall = recall
        self.precision = precision
        self.fscore = fscore

    def __str__(self):
        return "(R= {:0.2f}, P= {:0.2f}, F= {:0.2f})".format(
            self.recall, self.precision, self.fscore)


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


def evalb(evalb_dir, gold_data, predicted_trees):
    assert os.path.exists(evalb_dir)
    evalb_program_path = os.path.join(evalb_dir, "evalb")
    evalb_param_path = os.path.join(evalb_dir, "COLLINS.prm")
    assert os.path.exists(evalb_program_path)
    assert os.path.exists(evalb_param_path)

    gold_trees = [data['tree'] for data in gold_data]

    assert len(gold_trees) == len(predicted_trees)

    temp_dir = tempfile.TemporaryDirectory(prefix="evalb-")
    gold_path = os.path.join(temp_dir.name, "gold.txt")
    predicted_path = os.path.join(temp_dir.name, "predicted.txt")
    output_path = os.path.join(temp_dir.name, "output.txt")

    with open(gold_path, "w") as outfile:
        for tree in gold_trees:
            outfile.write("{}\n".format(str(tree)))

    with open(predicted_path, "w") as outfile:
        for tree in predicted_trees:
            outfile.write("{}\n".format(str(tree)))

    command = "{} -p {} {} {} > {}".format(
        evalb_program_path,
        evalb_param_path,
        gold_path,
        predicted_path,
        output_path,
    )
    subprocess.run(command, shell=True)

    fscore = FScore(math.nan, math.nan, math.nan)
    with open(output_path) as infile:
        for line in infile:
            match = re.match(r"Bracketing Recall\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.recall = float(match.group(1))
            match = re.match(r"Bracketing Precision\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.precision = float(match.group(1))
            match = re.match(r"Bracketing FMeasure\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.fscore = float(match.group(1))
                break

    success = (
        not math.isnan(fscore.fscore) or
        fscore.recall == 0.0 or
        fscore.precision == 0.0)

    if success:
        temp_dir.cleanup()
    else:
        print("Error reading EVALB results.")
        print("Gold path: {}".format(gold_path))
        print("Predicted path: {}".format(predicted_path))
        print("Output path: {}".format(output_path))

    return fscore
