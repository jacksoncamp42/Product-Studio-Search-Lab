#!/bin/bash

# Set paths
INPUT_CSV="data.csv"
OUTPUT_CSV="data_fluency.csv"

# Run Python script
python run_experiment.py --input_csv $INPUT_CSV --output_csv $OUTPUT_CSV --func_name=fluent_optimization_gpt

echo "Evaluation complete. Results saved to $OUTPUT_CSV."