#!/bin/bash

MY_DIR="`python -c "import os; print(os.path.split(os.path.realpath('$0'))[0])"`"
cd $MY_DIR

cd oke
./evaluate_all.sh 10
./evaluate_all.sh 5
# ./evaluate_all.sh 3
cd ..
