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
from utils import generate_text, compare_two_models, train, evaluate, get_single_adapter_lora_model, merge_configs, load_peft_checkpoint, get_adapter_status, print_layerwise_details, set_adapter_requires_grad
from dataset import MACSUM
import yaml
from peft import LoraConfig, TaskType
import wandb
import adapters.composition as ac

#python multi_attribute_single_adapter_continued_sft.py --debug > logs/sft_continued.txt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="/home2/tathagato/summarization/MACSUM/naacl/configs/multi_attribute_multi_adapter_sft.yaml")
    parser.add_argument("--experiment_name", type=str, default="llama_length_then_extractiveness_test")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    
    experiment_config = config["experiments"][args.experiment_name]
    config = config["global"]
    config = merge_configs(config, experiment_config)

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
        inference_mode = False
        
    )
    assert len(experiment_config["attributes"]) > 1, "there needs to be more than one attribute, one is the existing checkpoint attribute the other is the new attribute"
    old_attribute = experiment_config["attributes"][0]
    new_attribute = experiment_config["attributes"][1]
    old_adapter_name = old_attribute
    new_adapter_name = new_attribute

    #load the model
    config['checkpoint_path'] = os.path.join(config['checkpoint_dir'], f"best_model_{old_attribute}", old_attribute)
    print("loading from the checkpoint", config['checkpoint_path'])
    model = load_peft_checkpoint(config, bnb_config, config['checkpoint_path'], is_trainable= True, adapter_name= old_adapter_name)
    model.merge_and_unload() # not ideal but this is what works till adapter hub supports loading llama3 and mistral 
    #print(model)
    model.add_adapter(new_adapter_name, lora_config)
    #get_adapter_status(model)
    model.set_adapter(new_adapter_name)
    #get_adapter_status(model)
    #print_layerwise_details(model)


    #print_layerwise_details(model)
    #import code; code.interact(local=locals())








    #load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(experiment_config["model_id"])

    #load the dataset
    if args.debug:
        train_dataset = MACSUM(
            dataset_path = config["train_dataset_path"],
            attributes = experiment_config["attributes"][1:],
            tokenizer = tokenizer,
            mode = "train",
            size = 16,
            max_seq_len = config["max_seq_len"],
            model_type = experiment_config["model_type"]
        )  
        val_dataset = MACSUM(
            dataset_path = config["val_dataset_path"],
            attributes = experiment_config["attributes"][1:],
            tokenizer = tokenizer,
            mode = "train",
            size = 16,
            max_seq_len = config["max_seq_len"],
            model_type = experiment_config["model_type"]
        )
        #assumes one two attributes for now 
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
        train_dataset = MACSUM(
            dataset_path = config["train_dataset_path"],
            attributes = experiment_config["attributes"][1:],
            tokenizer = tokenizer,
            mode = "train",
            size = -1,
            max_seq_len = config["max_seq_len"],
            model_type = experiment_config["model_type"]
        )
        val_dataset = MACSUM(
            dataset_path = config["val_dataset_path"],
            attributes = experiment_config["attributes"][1:],
            tokenizer = tokenizer,
            mode = "train",
            size = -1,
            max_seq_len = config["max_seq_len"],
            model_type = experiment_config["model_type"]
        )
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
    
    print(f"Length of train dataset: {len(train_dataset)}")
    print(f"train attributes: {train_dataset.attributes}")
    print(f"Length of val dataset: {len(val_dataset)}")
    print(f"val attributes: {val_dataset.attributes}")
    for i, test_dataset in enumerate(test_datasets):
        print(f"Length of test dataset with attributes {test_dataset.attributes} : {len(test_dataset)}")
    print(model)
    model.print_trainable_parameters()
    print("start training")
    train(model, tokenizer, train_dataset, val_dataset, config, device=0, save_pretrained= True, do_wandb=config['do_wandb'], adapter_name= new_adapter_name)

    print("training done")

    #loading the best checkpoint
    #delete the model from memory and free up the memory 
    del model
    #load the best model from the checkpoint
    best_path = os.path.join(config["output_dir"], f"best_model_{new_adapter_name}", new_adapter_name)
    print(f"Loading the best model from {best_path}")
    model = load_peft_checkpoint(config, bnb_config, config['checkpoint_path'], is_trainable= True, adapter_name= old_adapter_name)
    model.merge_and_unload() # not ideal but this is what works till adapter hub supports loading llama3 and mistral 
    model.load_adapter(best_path, adapter_name= new_adapter_name, peft_config = lora_config)
    #get_adapter_status(model)
    model.set_adapter(new_adapter_name)
    #compare_two_models(model, loaded_model)
    #exit()

    #print the active adapter
    get_adapter_status(model)
    train_adapter_name = "_then_".join(experiment_config["attributes"])
    for i, test_dataset in tqdm.tqdm(enumerate(test_datasets)):
        print(f"generating on test dataset with attributes {test_dataset.attributes}")
        result_dict = generate_text(model, test_dataset, tokenizer = tokenizer, config = config)
        print("done generating text for test dataset with attributes ", test_dataset.attributes)
        adapter_name = "_and_".join(test_dataset.attributes)
        with open(os.path.join(config["output_dir"], f"model_{train_adapter_name}_results_{adapter_name}.pkl"), "wb") as f:
            pkl.dump(result_dict, f)

        print("done saving results at ", os.path.join(config["output_dir"], f"model_{train_adapter_name}_results_{adapter_name}.pkl"))





