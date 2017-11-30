#Default_predict.sh
#!/bin/bash

# sh Default_predict.sh <model> <testing public data> <testing private data>
python 2.7 predict.py My_model $1 $2 $3
