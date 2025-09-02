import subprocess
import time
import yaml
import os

os.chdir("/home2/tathagato/summarization/MACSUM/naacl")
# Load the config file
config_path = "/home2/tathagato/summarization/MACSUM/naacl/configs/hlora_sft.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

DEBUG = False
experiment_names = list(config["experiments"].keys())

# Set up GPU and experiment configurations
gpu_capacity = 1  # Each GPU can run 5 experiments at a time
num_gpus = 4
experiments_per_gpu = {gpu_id: [] for gpu_id in range(num_gpus)}
processes_per_gpu = {gpu_id: [] for gpu_id in range(num_gpus)}

# Function to run a single experiment
def run_experiment(experiment, gpu_id):
    os.makedirs("./logs", exist_ok=True)
    log_file = f"./logs/{experiment}_gpu{gpu_id}.log"
    print("Starting experiment", experiment, "on GPU", gpu_id)
    
    if DEBUG:
        command = f"CUDA_VISIBLE_DEVICES={gpu_id} python multi_attribute_hlora_sft.py --experiment_name {experiment} --debug > {log_file} 2>&1"
    else:
        command = f"CUDA_VISIBLE_DEVICES={gpu_id} python multi_attribute_hlora_sft.py --experiment_name {experiment} > {log_file} 2>&1"
    
    process = subprocess.Popen(command, shell=True)
    print("Started experiment", experiment, "on GPU", gpu_id, "with PID", process.pid)
    return process

# Function to monitor the running processes
def monitor_processes():
    for gpu_id in range(num_gpus):
        finished_processes = []
        finished_experiments = []
        
        for i, process in enumerate(processes_per_gpu[gpu_id]):
            if process.poll() is not None:  # Check if the process has finished
                finished_processes.append(process)
                finished_experiments.append(i)  # Track the index of the finished experiment
                
                # Print if finished successfully or not
                if process.returncode == 0:
                    print(f"Experiment on GPU {gpu_id} with PID {process.pid} finished successfully")
                else:
                    print(f"Experiment on GPU {gpu_id} with PID {process.pid} finished with error")
        
        # Remove finished processes and experiments from the list
        for i in sorted(finished_experiments, reverse=True):
            del processes_per_gpu[gpu_id][i]
            del experiments_per_gpu[gpu_id][i]

# Function to distribute experiments across GPUs
def distribute_experiments():
    remaining_experiments = experiment_names.copy()
    
    # Start 5 experiments per GPU
    for gpu_id in range(num_gpus):
        while len(experiments_per_gpu[gpu_id]) < gpu_capacity and remaining_experiments:
            experiment = remaining_experiments.pop(0)
            process = run_experiment(experiment, gpu_id)
            experiments_per_gpu[gpu_id].append(experiment)
            processes_per_gpu[gpu_id].append(process)

    # Continue monitoring the experiments
    while remaining_experiments or any(processes_per_gpu[gpu_id] for gpu_id in range(num_gpus)):
        monitor_processes()

        # Launch new experiments on GPUs where capacity is available
        for gpu_id in range(num_gpus):
            while len(experiments_per_gpu[gpu_id]) < gpu_capacity and remaining_experiments:
                experiment = remaining_experiments.pop(0)
                process = run_experiment(experiment, gpu_id)
                experiments_per_gpu[gpu_id].append(experiment)
                processes_per_gpu[gpu_id].append(process)

        time.sleep(5)  # Wait for a few seconds before checking again

if __name__ == "__main__":
    distribute_experiments()
