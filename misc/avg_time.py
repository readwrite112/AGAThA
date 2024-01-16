import sys
import os
import json

# Input: process name | dataset id | raw_file | output_file | number of iterations

process = sys.argv[1]
dataset_id = sys.argv[2]
raw_file = sys.argv[3]
output_file = sys.argv[4]
iter = int(sys.argv[5])

# Get average execution time
if os.path.exists(raw_file):
    raw = open(raw_file, "r").read().splitlines()

    total_time = 0.0
    
    if len(raw) != 0:
        for r in raw:
            total_time += float(r)
        avg_time = total_time/float(iter)
    else:
        avg_time = "NaN"
    
else:
    avg_time = "NaN"


# Store result to json file

if os.path.exists(output_file):
    with open(output_file, "r") as json_file:
        output = json.load(json_file)
else: 
    output = {}

if process not in output:
    output[process] = {}

output[process][dataset_id] = avg_time

with open(output_file, "w") as json_file:
    json.dump(output, json_file)
