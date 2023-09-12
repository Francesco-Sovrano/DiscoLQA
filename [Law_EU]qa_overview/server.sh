#!/bin/bash

MY_DIR="`python -c "import os; print(os.path.split(os.path.realpath('$0'))[0])"`"
cd $MY_DIR

# PyClean
(find ./ -name __pycache__ -type d | xargs rm -r) && (find ./ -name *.pyc -type f | xargs rm -r)

# Run YAI Server
cd yai
echo 'Running YAI server..'
source .env/bin/activate
python3 server.py $1 &> server.log &
disown
cd ..

# Run OKE Server
cd oke
echo 'Running OKE server..'
source .env/bin/activate
# python3 server.py $1 &> server.log &
# disown
# cd ..
python3 server.py $1 $2 $3 &> server.log &
disown
cd ..