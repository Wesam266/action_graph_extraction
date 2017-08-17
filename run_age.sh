#!/bin/bash

model="$1"
if [[ $model == '' ]]; then
    echo "Must specify model to run from: seq_model/uwkiddon_model as first param to script"
    exit 1
fi

# Set-up command line args to py script.
log_dir='/iesl/canvas/smysore/material_science_ag/pipeline/all_logs'
data_dir='/iesl/canvas/smysore/material_science_ag/papers_data_json'
db_name='predsynth'
collection_name='annotated_papers'
tar_task='age'
# This needs to come in a cl_arg to the sh to prevent mistakes but putting it
# here for now.
parsed_file_suffix='deps_heu_parsed'

# Create directory to write learnt models to.
model_dir="./models/$model"
mkdir -p "$model_dir"
echo "Created model directory: $model_dir"

# Create results directory.
results_dir="$data_dir/$db_name-$collection_name-$tar_task-results"
mkdir -p "$results_dir"
echo "Created results directory: $results_dir"

doi_file="$data_dir/$db_name-$collection_name-$tar_task-train-dois.txt"

if [[ "$model" == "seq_model" ]]; then
	cmd="python2 -u main.py seq_model \
	--db_name $db_name \
	--collection_name $collection_name \
	--tar_task $tar_task \
	--parsed_file_suffix $parsed_file_suffix \
	--doi_file $doi_file \
	--model_dir $model_dir"
	echo ${cmd} | tee "$log_dir/action_graph_extraction_logs/${tar_task}_${model}_logs.txt"
	eval ${cmd} 2>&1 | tee -a "$log_dir/action_graph_extraction_logs/${tar_task}_${model}_logs.txt"
elif [[ "$model" == "uwkiddon_model" ]]; then
	cmd="python2 -u main.py uwkiddon_model \
	--db_name $db_name \
	--collection_name $collection_name \
	--tar_task $tar_task \
	--parsed_file_suffix $parsed_file_suffix \
	--doi_file $doi_file \
	--model_dir $model_dir \
	--em_iters 5 \
	--term_min_swaps 40"
	echo ${cmd} | tee "$log_dir/action_graph_extraction_logs/${tar_task}_${model}_logs.txt"
	eval ${cmd} 2>&1 | tee -a "$log_dir/action_graph_extraction_logs/${tar_task}_${model}_logs.txt"
fi