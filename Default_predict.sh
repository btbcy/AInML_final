#Default_predict.sh
#!/bin/bash
#
# USAGE
# bash Default_predict.sh <testing public data> <testing private data>
# ex. bash Default_predict.sh Test_Public.csv Test_Private.csv
# input file: Test_Public.csv, Test_Private.csv
# output file: public.csv, private.csv

python2.7 predict.py model.m $1 $2
