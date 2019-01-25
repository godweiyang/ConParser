# Train 
python3 run/train.py --model MyTopDownParser
python3 run/train.py --model MyTopDownParser --train_file data/train_small.trees --dev_file data/dev_small.trees
python3 run/train_more.py --model InOrderParser --more_epoch 20

# Test
python3 run/test.py --model MyTopDownParser --dev_fscore xx.xx

