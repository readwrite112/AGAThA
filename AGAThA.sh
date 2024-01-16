#!/bin/bash
PROG="/agatha_ae/AGAThA/test_prog/"

ITER=1 #number of iteration of each program
IDLE=5 #sleep between iterations
RAW_FILE="/agatha_ae/output/raw.log"
FINAL_FILE="/agatha_ae/output/time.json"
SCORE_FILE="/agatha_ae/output/score.log"
DATASET_DIR="/agatha_ae/dataset/"
DATASET_NAME="test"
FINAL_DIR=$PWD
PROCESS="AGAThA"

while getopts "i:" opt
do
    case "$opt" in
    i ) ITER="$OPTARG" ;;
    esac
done

echo ">>> Running $PROCESS"


if [ -f $RAW_FILE ]; then
    rm $RAW_FILE
fi

if [ -f $SCORE_FILE ]; then
    rm $SCORE_FILE
fi

if [ -f $FINAL_FILE ]; then
    rm $FINAL_FILE
fi

iter=0
while [ "$iter" -lt $ITER ] 
do
    ${PROG}manual -p -y local -a 1 -b 4 -q 6 -r 2 -z 400 -w 751 ${DATASET_DIR}ref.fasta ${DATASET_DIR}query.fasta > ${SCORE_FILE}
    ((iter++))
    sleep ${IDLE}s
done

python3 /agatha_ae/misc/avg_time.py $PROCESS $DATASET_NAME ${RAW_FILE} ${FINAL_FILE} $ITER  #get average exec. time across multiple iterations


