#!/bin/bash

# MY_DIR="`python -c "import os; print(os.path.split(os.path.realpath('$0'))[0])"`"
# cd $MY_DIR

# deactivate
source .env/bin/activate

LOG="top$1_log"
mkdir results

################################################################################################
mkdir results/syntagm_tuner

########################
########################
mkdir results/syntagm_tuner/all_regulations_search
mkdir results/syntagm_tuner/all_regulations_search/$LOG

LOG_DIR="results/syntagm_tuner/all_regulations_search/$LOG/minilm"
mkdir $LOG_DIR
python3 evaluate.py $1 0.5 minilm false edu_amr $LOG_DIR &> $LOG_DIR/edu_amr.log
python3 evaluate.py $1 0.5 minilm false edu_amr_clause $LOG_DIR &> $LOG_DIR/edu_amr_clause.log
python3 evaluate.py $1 0.5 minilm false clause $LOG_DIR &> $LOG_DIR/clause.log
python3 evaluate.py $1 0.5 minilm false edu_clause $LOG_DIR &> $LOG_DIR/edu_clause.log
python3 evaluate.py $1 0.5 minilm false amr_clause $LOG_DIR &> $LOG_DIR/amr_clause.log
python3 evaluate.py $1 0.5 minilm false edu $LOG_DIR &> $LOG_DIR/edu.log
python3 evaluate.py $1 0.5 minilm false amr $LOG_DIR &> $LOG_DIR/amr.log

python3 statistical_tests.py $LOG_DIR &> $LOG_DIR/stats.txt
####

LOG_DIR="results/syntagm_tuner/all_regulations_search/$LOG/mpnet"
mkdir $LOG_DIR
python3 evaluate.py $1 0.5 mpnet false clause $LOG_DIR &> $LOG_DIR/clause.log
python3 evaluate.py $1 0.5 mpnet false edu_amr_clause $LOG_DIR &> $LOG_DIR/edu_amr_clause.log
python3 evaluate.py $1 0.5 mpnet false edu_clause $LOG_DIR &> $LOG_DIR/edu_clause.log
python3 evaluate.py $1 0.5 mpnet false amr_clause $LOG_DIR &> $LOG_DIR/amr_clause.log
python3 evaluate.py $1 0.5 mpnet false edu_amr $LOG_DIR &> $LOG_DIR/edu_amr.log
python3 evaluate.py $1 0.5 mpnet false edu $LOG_DIR &> $LOG_DIR/edu.log
python3 evaluate.py $1 0.5 mpnet false amr $LOG_DIR &> $LOG_DIR/amr.log

python3 statistical_tests.py $LOG_DIR &> $LOG_DIR/stats.txt
####

LOG_DIR="results/syntagm_tuner/all_regulations_search/$LOG/tf"
mkdir $LOG_DIR
python3 evaluate.py $1 0.5 tf false clause $LOG_DIR &> $LOG_DIR/clause.log
python3 evaluate.py $1 0.5 tf false edu_amr_clause $LOG_DIR &> $LOG_DIR/edu_amr_clause.log
python3 evaluate.py $1 0.5 tf false edu_clause $LOG_DIR &> $LOG_DIR/edu_clause.log
python3 evaluate.py $1 0.5 tf false amr_clause $LOG_DIR &> $LOG_DIR/amr_clause.log
python3 evaluate.py $1 0.5 tf false edu_amr $LOG_DIR &> $LOG_DIR/edu_amr.log
python3 evaluate.py $1 0.5 tf false edu $LOG_DIR &> $LOG_DIR/edu.log
python3 evaluate.py $1 0.5 tf false amr $LOG_DIR &> $LOG_DIR/amr.log

python3 statistical_tests.py $LOG_DIR &> $LOG_DIR/stats.txt
####

########################
########################
mkdir results/syntagm_tuner/target_regulations_only
mkdir results/syntagm_tuner/target_regulations_only/$LOG

LOG_DIR="results/syntagm_tuner/target_regulations_only/$LOG/tf"
mkdir $LOG_DIR
python3 evaluate.py $1 0.5 tf true clause $LOG_DIR &> $LOG_DIR/clause.log
python3 evaluate.py $1 0.5 tf true edu_amr_clause $LOG_DIR &> $LOG_DIR/edu_amr_clause.log
python3 evaluate.py $1 0.5 tf true edu_clause $LOG_DIR &> $LOG_DIR/edu_clause.log
python3 evaluate.py $1 0.5 tf true amr_clause $LOG_DIR &> $LOG_DIR/amr_clause.log
python3 evaluate.py $1 0.5 tf true edu_amr $LOG_DIR &> $LOG_DIR/edu_amr.log
python3 evaluate.py $1 0.5 tf true edu $LOG_DIR &> $LOG_DIR/edu.log
python3 evaluate.py $1 0.5 tf true amr $LOG_DIR &> $LOG_DIR/amr.log

python3 statistical_tests.py $LOG_DIR &> $LOG_DIR/stats.txt
####

LOG_DIR="results/syntagm_tuner/target_regulations_only/$LOG/minilm"
mkdir $LOG_DIR
python3 evaluate.py $1 0.5 minilm true clause $LOG_DIR &> $LOG_DIR/clause.log
python3 evaluate.py $1 0.5 minilm true edu_amr_clause $LOG_DIR &> $LOG_DIR/edu_amr_clause.log
python3 evaluate.py $1 0.5 minilm true edu_clause $LOG_DIR &> $LOG_DIR/edu_clause.log
python3 evaluate.py $1 0.5 minilm true amr_clause $LOG_DIR &> $LOG_DIR/amr_clause.log
python3 evaluate.py $1 0.5 minilm true edu_amr $LOG_DIR &> $LOG_DIR/edu_amr.log
python3 evaluate.py $1 0.5 minilm true edu $LOG_DIR &> $LOG_DIR/edu.log
python3 evaluate.py $1 0.5 minilm true amr $LOG_DIR &> $LOG_DIR/amr.log

python3 statistical_tests.py $LOG_DIR &> $LOG_DIR/stats.txt
####

LOG_DIR="results/syntagm_tuner/target_regulations_only/$LOG/mpnet"
mkdir $LOG_DIR
python3 evaluate.py $1 0.5 mpnet true clause $LOG_DIR &> $LOG_DIR/clause.log
python3 evaluate.py $1 0.5 mpnet true edu_amr_clause $LOG_DIR &> $LOG_DIR/edu_amr_clause.log
python3 evaluate.py $1 0.5 mpnet true edu_clause $LOG_DIR &> $LOG_DIR/edu_clause.log
python3 evaluate.py $1 0.5 mpnet true amr_clause $LOG_DIR &> $LOG_DIR/amr_clause.log
python3 evaluate.py $1 0.5 mpnet true edu_amr $LOG_DIR &> $LOG_DIR/edu_amr.log
python3 evaluate.py $1 0.5 mpnet true edu $LOG_DIR &> $LOG_DIR/edu.log
python3 evaluate.py $1 0.5 mpnet true amr $LOG_DIR &> $LOG_DIR/amr.log

python3 statistical_tests.py $LOG_DIR &> $LOG_DIR/stats.txt
####

################################################################################################
mkdir results/no_tfidf

########################
########################
mkdir results/no_tfidf/all_regulations_search
mkdir results/no_tfidf/all_regulations_search/$LOG

LOG_DIR="results/no_tfidf/all_regulations_search/$LOG/tf"
mkdir $LOG_DIR
python3 evaluate.py $1 0 tf false clause $LOG_DIR &> $LOG_DIR/clause.log
python3 evaluate.py $1 0 tf false edu_amr_clause $LOG_DIR &> $LOG_DIR/edu_amr_clause.log
python3 evaluate.py $1 0 tf false edu_clause $LOG_DIR &> $LOG_DIR/edu_clause.log
python3 evaluate.py $1 0 tf false amr_clause $LOG_DIR &> $LOG_DIR/amr_clause.log
python3 evaluate.py $1 0 tf false edu_amr $LOG_DIR &> $LOG_DIR/edu_amr.log
python3 evaluate.py $1 0 tf false edu $LOG_DIR &> $LOG_DIR/edu.log
python3 evaluate.py $1 0 tf false amr $LOG_DIR &> $LOG_DIR/amr.log

python3 statistical_tests.py $LOG_DIR &> $LOG_DIR/stats.txt
####

LOG_DIR="results/no_tfidf/all_regulations_search/$LOG/minilm"
mkdir $LOG_DIR
python3 evaluate.py $1 0 minilm false clause $LOG_DIR &> $LOG_DIR/clause.log
python3 evaluate.py $1 0 minilm false edu_amr_clause $LOG_DIR &> $LOG_DIR/edu_amr_clause.log
python3 evaluate.py $1 0 minilm false edu_clause $LOG_DIR &> $LOG_DIR/edu_clause.log
python3 evaluate.py $1 0 minilm false amr_clause $LOG_DIR &> $LOG_DIR/amr_clause.log
python3 evaluate.py $1 0 minilm false edu_amr $LOG_DIR &> $LOG_DIR/edu_amr.log
python3 evaluate.py $1 0 minilm false edu $LOG_DIR &> $LOG_DIR/edu.log
python3 evaluate.py $1 0 minilm false amr $LOG_DIR &> $LOG_DIR/amr.log

python3 statistical_tests.py $LOG_DIR &> $LOG_DIR/stats.txt
####

LOG_DIR="results/no_tfidf/all_regulations_search/$LOG/mpnet"
mkdir $LOG_DIR
python3 evaluate.py $1 0 mpnet false clause $LOG_DIR &> $LOG_DIR/clause.log
python3 evaluate.py $1 0 mpnet false edu_amr_clause $LOG_DIR &> $LOG_DIR/edu_amr_clause.log
python3 evaluate.py $1 0 mpnet false edu_clause $LOG_DIR &> $LOG_DIR/edu_clause.log
python3 evaluate.py $1 0 mpnet false amr_clause $LOG_DIR &> $LOG_DIR/amr_clause.log
python3 evaluate.py $1 0 mpnet false edu_amr $LOG_DIR &> $LOG_DIR/edu_amr.log
python3 evaluate.py $1 0 mpnet false edu $LOG_DIR &> $LOG_DIR/edu.log
python3 evaluate.py $1 0 mpnet false amr $LOG_DIR &> $LOG_DIR/amr.log

python3 statistical_tests.py $LOG_DIR &> $LOG_DIR/stats.txt
####

########################
########################
mkdir results/no_tfidf/target_regulations_only
mkdir results/no_tfidf/target_regulations_only/$LOG

LOG_DIR="results/no_tfidf/target_regulations_only/$LOG/tf"
mkdir $LOG_DIR
python3 evaluate.py $1 0 tf true clause $LOG_DIR &> $LOG_DIR/clause.log
python3 evaluate.py $1 0 tf true edu_amr_clause $LOG_DIR &> $LOG_DIR/edu_amr_clause.log
python3 evaluate.py $1 0 tf true edu_clause $LOG_DIR &> $LOG_DIR/edu_clause.log
python3 evaluate.py $1 0 tf true amr_clause $LOG_DIR &> $LOG_DIR/amr_clause.log
python3 evaluate.py $1 0 tf true edu_amr $LOG_DIR &> $LOG_DIR/edu_amr.log
python3 evaluate.py $1 0 tf true edu $LOG_DIR &> $LOG_DIR/edu.log
python3 evaluate.py $1 0 tf true amr $LOG_DIR &> $LOG_DIR/amr.log

python3 statistical_tests.py $LOG_DIR &> $LOG_DIR/stats.txt
####

LOG_DIR="results/no_tfidf/target_regulations_only/$LOG/minilm"
mkdir $LOG_DIR
python3 evaluate.py $1 0 minilm true clause $LOG_DIR &> $LOG_DIR/clause.log
python3 evaluate.py $1 0 minilm true edu_amr_clause $LOG_DIR &> $LOG_DIR/edu_amr_clause.log
python3 evaluate.py $1 0 minilm true edu_clause $LOG_DIR &> $LOG_DIR/edu_clause.log
python3 evaluate.py $1 0 minilm true amr_clause $LOG_DIR &> $LOG_DIR/amr_clause.log
python3 evaluate.py $1 0 minilm true edu_amr $LOG_DIR &> $LOG_DIR/edu_amr.log
python3 evaluate.py $1 0 minilm true edu $LOG_DIR &> $LOG_DIR/edu.log
python3 evaluate.py $1 0 minilm true amr $LOG_DIR &> $LOG_DIR/amr.log

python3 statistical_tests.py $LOG_DIR &> $LOG_DIR/stats.txt
####

LOG_DIR="results/no_tfidf/target_regulations_only/$LOG/mpnet"
mkdir $LOG_DIR
python3 evaluate.py $1 0 mpnet true clause $LOG_DIR &> $LOG_DIR/clause.log
python3 evaluate.py $1 0 mpnet true edu_amr_clause $LOG_DIR &> $LOG_DIR/edu_amr_clause.log
python3 evaluate.py $1 0 mpnet true edu_clause $LOG_DIR &> $LOG_DIR/edu_clause.log
python3 evaluate.py $1 0 mpnet true amr_clause $LOG_DIR &> $LOG_DIR/amr_clause.log
python3 evaluate.py $1 0 mpnet true edu_amr $LOG_DIR &> $LOG_DIR/edu_amr.log
python3 evaluate.py $1 0 mpnet true edu $LOG_DIR &> $LOG_DIR/edu.log
python3 evaluate.py $1 0 mpnet true amr $LOG_DIR &> $LOG_DIR/amr.log

python3 statistical_tests.py $LOG_DIR &> $LOG_DIR/stats.txt
####