# Constituent Parser
This is a project of the following constituent parsers:
* Chart Parser (from [A Minimal Span-Based Neural Constituency Parser](http://aclweb.org/anthology/P17-1076)).
* Top-Down Parser (from [A Minimal Span-Based Neural Constituency Parser](http://aclweb.org/anthology/P17-1076)).
* Shift-Reduce Parser (from [Span-Based Constituency Parsing with a Structure-Label System and Provably Optimal Dynamic Oracles](http://aclweb.org/anthology/D16-1001)).
* In-Order Parser (Ours).
* GNN Parser (Future work).

*The shift-reduce parser code still has not been reconstructed.*


# Train 
**Full dataset:**

`python3 run/train.py --model InOrderParser`

**Full dataset (for more training):**

`python3 run/train.py --model InOrderParser --train_more --more_epoch 20`

**Small dataset (for code correctness test):**

`python3 run/train.py --model InOrderParser --train_file data/train_small.trees --dev_file data/dev_small.trees`

# Test
`python3 run/test.py --model InOrderParser --dev_fscore xx.xx`

