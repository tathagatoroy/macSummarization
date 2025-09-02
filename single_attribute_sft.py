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
from utils import generate_text, train, evaluate, get_single_adapter_lora_model, merge_configs, load_peft_checkpoint, get_adapter_status, print_layerwise_details
from dataset import MACSUM
import yaml
from peft import LoraConfig, TaskType
import wandb
#python single_attribute_sft.py --debug > logs/sft_single.txt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="/home2/tathagato/summarization/MACSUM/naacl/configs/single_attribute_sft.yaml")
    parser.add_argument("--experiment_name", type=str, default="test_llama_length")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    
    experiment_config = config["experiments"][args.experiment_name]
    config = config["global"]
    config = merge_configs(config, experiment_config)

    config['output_dir'] = os.path.join(config['output_dir'], args.experiment_name)
    os.makedirs(config['output_dir'], exist_ok=True)

    if args.debug:
        config['do_wandb'] = False
        config['logging_steps'] = 1
        config['eval_interval'] = 1

    #setup wandb 
    if config['do_wandb'] :
        wandb.init(project = config["wandb_project"], name = args.experiment_name)
        wandb.config.update(config)

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

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r = config["rank"],
        lora_alpha = config["lora_alpha"],
        target_modules= config["target_modules"],
        lora_dropout= config["lora_dropout"],
        inference_mode = False 
        
    )
    assert len(experiment_config["attributes"]) == 1, "Only single attribute supported"
    model = get_single_adapter_lora_model(model, lora_config, experiment_config["attributes"][0])



    #load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(experiment_config["model_id"])

    #load the dataset
    if args.debug:
        train_dataset = MACSUM(
            dataset_path = config["train_dataset_path"],
            attributes = experiment_config["attributes"],
            tokenizer = tokenizer,
            mode = "train",
            size = 16,
            max_seq_len = config["max_seq_len"],
            model_type = experiment_config["model_type"]
        )  
        val_dataset = MACSUM(
            dataset_path = config["val_dataset_path"],
            attributes = experiment_config["attributes"],
            tokenizer = tokenizer,
            mode = "train",
            size = 16,
            max_seq_len = config["max_seq_len"],
            model_type = experiment_config["model_type"]
        )
        test_dataset = MACSUM(
            dataset_path = config["test_dataset_path"],
            attributes = experiment_config["attributes"],
            tokenizer = tokenizer,
            mode = "inference",
            size = 4,
            max_seq_len = config["max_seq_len"],
            model_type = experiment_config["model_type"]
        )
    else:
        train_dataset = MACSUM(
            dataset_path = config["train_dataset_path"],
            attributes = experiment_config["attributes"],
            tokenizer = tokenizer,
            mode = "train",
            size = -1,
            max_seq_len = config["max_seq_len"],
            model_type = experiment_config["model_type"]
        )
        val_dataset = MACSUM(
            dataset_path = config["val_dataset_path"],
            attributes = experiment_config["attributes"],
            tokenizer = tokenizer,
            mode = "train",
            size = -1,
            max_seq_len = config["max_seq_len"],
            model_type = experiment_config["model_type"]
        )
        test_dataset = MACSUM(
            dataset_path = config["test_dataset_path"],
            attributes = experiment_config["attributes"],
            tokenizer = tokenizer,
            mode = "inference",
            size = -1,
            max_seq_len = config["max_seq_len"],
            model_type = experiment_config["model_type"]
        )
    
    print(f"Length of train dataset: {len(train_dataset)}")
    print(f"Length of val dataset: {len(val_dataset)}")
    print(f"Length of test dataset: {len(test_dataset)}")

    print(model)
    get_adapter_status(model)
    model.print_trainable_parameters()
    #print_layerwise_details(model)
    train(model, tokenizer, train_dataset, val_dataset, config, device=0, save_pretrained= True, do_wandb=config['do_wandb'])

    print("training done")

    #loading the best checkpoint
    #delete the model from memory and free up the memory 
    del model
    #load the best model from the checkpoint
    best_path = os.path.join(config["output_dir"], f"best_model_{config['attributes'][0]}", config["attributes"][0])
    print(f"Loading the best model from {best_path}")
    model = load_peft_checkpoint(config, bnb_config, best_path)

    #print the active adapter
    get_adapter_status(model)

    result_dict = generate_text(model, test_dataset, tokenizer = tokenizer, config = config)
    print("done generating text")

    #save the results
    with open(os.path.join(config["output_dir"], "results.pkl"), "wb") as f:
        pkl.dump(result_dict, f)
    print("done saving results at ", os.path.join(config["output_dir"], "results.pkl"))

    #setup wandb 




