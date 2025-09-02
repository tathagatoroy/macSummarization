import torch
from transformers import LlamaForCausalLM, BitsAndBytesConfig, AutoModelForCausalLM
import os 
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import sys
import pickle as pkl
from dataset import MACSUM
import tqdm
import time
from utils import generate_text
from dataset import MACSUM
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/zero_shot.yaml")
    parser.add_argument("--experiment_name", type=str, default="llama_length")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    
    experiment_config = config["experiments"][args.experiment_name]
    config = config["global"]

    #setup the bnb config
    bnb_compute_dtype = torch.bfloat16 if config["bnb_4bit_compute_dtype"] == "bfloat16" else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['load_in_4bit'],
        bnb_4bit_quant_type=config['bnb_4bit_quant_type'],
        bnb_4bit_use_double_quant=config['bnb_4bit_use_double_quant'],
        bnb_4bit_compute_dtype=bnb_compute_dtype

    )

    #load the model
    model = AutoModelForCausalLM.from_pretrained(
        experiment_config["model_id"],
        use_cache=False,
        quantization_config=bnb_config,
        device_map="cuda:0"
    )

    #load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(experiment_config["model_id"])

    #load the dataset
    if args.debug:
        dataset = MACSUM(
            dataset_path = config["dataset_path"],
            attributes = experiment_config["attributes"],
            tokenizer = tokenizer,
            mode = "inference",
            size = 4,
            max_seq_len = config["max_seq_len"],
            model_type = experiment_config["model_type"]
        )  
    else:
        dataset = MACSUM(
            dataset_path = config["dataset_path"],
            attributes = experiment_config["attributes"],
            tokenizer = tokenizer,
            mode = "inference",
            size = -1,
            max_seq_len = config["max_seq_len"],
            model_type = experiment_config["model_type"]
        )
    
    print(f"Length of dataset: {len(dataset)}")


    #start the generation
    result_dict = generate_text(model, dataset, tokenizer = tokenizer, config = config)

    #save the results
    print("Saving the results")
    os.makedirs(config["output_dir"], exist_ok=True)
    save_path = os.path.join(config["output_dir"], args.experiment_name + ".pkl")
    with open(save_path, "wb") as f:
        pkl.dump(result_dict, f) 
    print(f"Results saved at {save_path}")

