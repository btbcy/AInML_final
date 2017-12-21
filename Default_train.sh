#Default_train.sh
#!/bin/bash
#
# USAGE
# bash Default_train.sh <training data>
# ex. bash Default_train.sh Train.csv
# input file: Train.csv
# output file: model.m

python2.7 train.py $1
