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
from utils import generate_text, train, evaluate, get_single_adapter_lora_model, merge_configs, load_peft_checkpoint, get_adapter_status, print_layerwise_details, load_fused_adapter_model, load_fused_dpo_models
from dataset import MACSUM
import yaml
from peft import LoraConfig, TaskType
import wandb
#CUDA_VISIBLE_DEVICES=1 python adapter_fusion.py --debug > logs/adapter_fusion.txt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="/home2/tathagato/summarization/MACSUM/naacl/configs/adapter_fusion_dpo.yaml")
    parser.add_argument("--experiment_name", type=str, default="llama_length_and_extractiveness_test")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    
    experiment_config = config["experiments"][args.experiment_name]
    config = config["global"]
    config = merge_configs(config, experiment_config)
    args.experiment_name = args.experiment_name + "_" + str(experiment_config["weights"][0]) + "_" + str(experiment_config["weights"][1]) + "_" + str(experiment_config["combination_type"])
    config['output_dir'] = os.path.join(config['output_dir'], args.experiment_name)
    os.makedirs(config['output_dir'], exist_ok=True)
    if "test" in args.experiment_name:
        args.debug = True
        

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



    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r = config["rank"],
        lora_alpha = config["lora_alpha"],
        target_modules= config["target_modules"],
        lora_dropout= config["lora_dropout"],
        inference_mode = True 
        
    )
    assert len(experiment_config["attributes"]) > 1, "there needs to be more than one attribute, one is the existing checkpoint attribute the other is the new attribute"
    old_attribute = experiment_config["attributes"][0]
    new_attribute = experiment_config["attributes"][1]
    w1, w2 = config["weights"][0], config["weights"][1]
    adapter_name = f"{old_attribute}_and_{new_attribute}_fused"
    #load the model
    config['checkpoint_paths'] = config['checkpoint_dirs']
    print("loading from the checkpoints:\n", config['checkpoint_paths'][0],"\n", config['checkpoint_paths'][1])
    model = load_fused_dpo_models(config, bnb_config, is_trainable = False, adapter_name = adapter_name, combination_type = config["combination_type"])
    get_adapter_status(model)
    print_layerwise_details(model)






    #load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(experiment_config["model_id"])

    #load the dataset
    if args.debug:

        test_datasets = [
            MACSUM(
                dataset_path = config["test_dataset_path"],
                attributes = experiment_config["attributes"],
                tokenizer = tokenizer,
                mode = "inference",
                size = 4,
                max_seq_len = config["max_seq_len"],
                model_type = experiment_config["model_type"]
            ),
            MACSUM(
                dataset_path= config["test_dataset_path"],
                attributes = experiment_config["attributes"][:1],
                tokenizer = tokenizer,
                mode = "inference",
                size = 4,
                max_seq_len = config["max_seq_len"],
                model_type = experiment_config["model_type"]
            ),
            MACSUM(
                dataset_path= config["test_dataset_path"],
                attributes = experiment_config["attributes"][1:],
                tokenizer = tokenizer,
                mode = "inference",
                size = 4,
                max_seq_len = config["max_seq_len"],
                model_type = experiment_config["model_type"]
            )
        ]

    else:

        test_datasets = [
            MACSUM(
            dataset_path = config["test_dataset_path"],
            attributes = experiment_config["attributes"],
            tokenizer = tokenizer,
            mode = "inference",
            size = -1,
            max_seq_len = config["max_seq_len"],
            model_type = experiment_config["model_type"]
            ),
            MACSUM(
                dataset_path= config["test_dataset_path"],
                attributes = experiment_config["attributes"][:1],
                tokenizer = tokenizer,
                mode = "inference",
                size = -1,
                max_seq_len = config["max_seq_len"],
                model_type = experiment_config["model_type"]
                ),
            MACSUM(
                dataset_path= config["test_dataset_path"],
                attributes = experiment_config["attributes"][1:],
                tokenizer = tokenizer,
                mode = "inference",
                size = -1,
                max_seq_len = config["max_seq_len"],
                model_type = experiment_config["model_type"]
            )
        ]
    

    for i, test_dataset in enumerate(test_datasets):
        print(f"Length of test dataset with attributes {test_dataset.attributes} : {len(test_dataset)}")
    print(model)

    #print the active adapter
    get_adapter_status(model)
    for i, test_dataset in tqdm.tqdm(enumerate(test_datasets)):
        print(f"generating on test dataset with attributes {test_dataset.attributes}")
        result_dict = generate_text(model, test_dataset, tokenizer = tokenizer, config = config)
        print("done generating text for test dataset with attributes ", test_dataset.attributes)
        dataset_attributes = "_and_".join(test_dataset.attributes)
        with open(os.path.join(config["output_dir"], f"model_{adapter_name}_results_{dataset_attributes}.pkl"), "wb") as f:
            pkl.dump(result_dict, f)

        print("done saving results at ", os.path.join(config["output_dir"], f"model_{adapter_name}_results_{dataset_attributes}.pkl"))





