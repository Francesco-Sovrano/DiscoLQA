#!/bin/bash

MY_DIR="`python -c "import os; print(os.path.realpath('$1'))"`"
cd $MY_DIR

# Run YAI server
cd yai
echo 'Setting up YAI server..'
rm -r .env
virtualenv .env -p python3.9
source .env/bin/activate
pip install -U pip
pip install -U setuptools wheel twine
pip install  -r requirements.txt
deactivate
cd ..

# Run OKE Server
cd oke
echo 'Setting up OKE server..'
rm -r .env
virtualenv .env -p python3.9
source .env/bin/activate
pip install -U pip
pip install -U setuptools wheel twine
echo 'Install QuAnsX'
cd /local/scratch/francesco_sovrano/DiscoLQA/packages/quansx
pip install .
echo 'Install KnowPy'
cd /local/scratch/francesco_sovrano/DiscoLQA/packages/knowpy
pip install .
# cd .env/lib
# git clone https://github.com/huggingface/neuralcoref.git
# cd neuralcoref
# pip install  -r requirements.txt
# pip install -e .
# cd ..
# cd ../..
pip install  -r requirements.txt
# python3 -m spacy download en_core_web_trf
python3 -m spacy download en_core_web_md
python3 -m nltk.downloader stopwords punkt averaged_perceptron_tagger framenet_v17 wordnet brown
cd ..
