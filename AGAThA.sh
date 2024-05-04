#!/bin/bash
MAIN_DIR="/agatha_ae/"                  #the main directory
PROG_DIR=$MAIN_DIR"AGAThA/test_prog/"   #the directory where the test program is
OUTPUT_DIR=$MAIN_DIR"output/"           #the directory for the RAW_, FINAL_, SCORE_FILE
DATASET_DIR=$MAIN_DIR"dataset/"         #the directory where the input dataset is located in
FINAL_DIR=$PWD

RAW_FILE=$OUTPUT_DIR"raw.log"           #stores all kernel exec. time of all iterations        
FINAL_FILE=$OUTPUT_DIR"time.json"       #stores the average (total kernel exec. time of a single iteration)
SCORE_FILE=$OUTPUT_DIR"score.log"       #stores the scores after alignment

ITER=1                                  #number of iteration of each program
IDLE=5                                  #sleep between iterations
DATASET_NAME="test"                     #the name for the current dataset (will be shown in FINAL_FILE)
PROCESS="AGAThA"                        #the process name (will be shown in FINAL_FILE)

while getopts "i:" opt
do
    case "$opt" in
    i ) ITER="$OPTARG" ;;
    esac
done

mkdir -p $OUTPUT_DIR                    #creating the output directory

echo ">>> Running $PROCESS for $ITER iterations."

if [ -f $RAW_FILE ]; then               #remove the output files before running the program
    rm $RAW_FILE
fi

if [ -f $SCORE_FILE ]; then
    rm $SCORE_FILE
fi

if [ -f $FINAL_FILE ]; then
    rm $FINAL_FILE
fi

iter=0                                  #start the main program
while [ "$iter" -lt $ITER ] 
do  
    echo ">> Iteration $(($iter+1))"
    ${PROG_DIR}manual -p -m 1 -x 4 -q 6 -r 2 -s 3 -z 400 -w 751 ${DATASET_DIR}ref.fasta ${DATASET_DIR}query.fasta ${RAW_FILE} > ${SCORE_FILE}
    ((iter++))
    sleep ${IDLE}s
done

echo "$PROCESS complete."
echo "Creating output files..."         #creating additional output files

python3 /agatha_ae/misc/avg_time.py $PROCESS $DATASET_NAME ${RAW_FILE} ${FINAL_FILE} $ITER 

echo "Complete."