#!/bin/bash
#SBATCH -A kcis
#SBATCH -n 1
#SBATCH --qos=kl4
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=4-00:00:00
#SBATCH --job-name=openai_topic_evaluation_rest
#SBATCH --output=/home2/tathagato/summarization/MACSUM/naacl/logs/topic_output.log
#SBATCH --partition=lovelace
#SBATCH --mail-user=tathagato.roy@research.iiit.ac.in
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH -N 1
#SBATCH -w gnode121
export NCCL_P2P_DISABLE=1

# python run_all_dpo_single_attribute.py
# echo "done first part"
# python run_all_dpo_joint_multi_attribute.py
# echo "done second part"
# python run_all_weighted_adapter_fusion_dpo.py

## add your huggingface token here 


# Perform the login using the token
# TEMP_FILE=$(mktemp)

# # Write the token to the temporary file
# echo "$HUGGINGFACE_TOKEN" > "$TEMP_FILE"

# # Perform the login using the token from the temporary file
# huggingface-cli login < "$TEMP_FILE"

cd /home2/tathagato/summarization/MACSUM/naacl
python test_openai.py




