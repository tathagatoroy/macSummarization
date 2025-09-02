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
from utils import generate_text, train, evaluate, get_single_adapter_lora_model, merge_configs, load_peft_checkpoint, get_adapter_status, print_layerwise_details, get_detached_state_dict, print_layers_with_requires_grad, return_changed_layers
from dataset import MACSUM
import yaml
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model
import wandb
from hlora import HLORAConfig, replace_linear4bit_with_hlora, set_train_adapters, set_inference_adapters, set_gradients_on_the_model, print_model_layer_info, replace_lora_with_hlora, setup_hlora_model

#CUDA_VISIBLE_DEVICES=0, python multi_attribute_hlora_sft.py --debug > logs/sft_single.txt

def output_generation(model, test_datasets, prefix = "final", config = None):
    for index, test_dataset in enumerate(test_datasets):
        test_attributes = "_".join(test_dataset.attributes)
        result_dict = generate_text(model, test_dataset, tokenizer = tokenizer, config = config)
        save_path = os.path.join(config["output_dir"], f"{prefix}_results_{test_attributes}_{index}.pkl")
        with open(save_path, "wb") as f:
            pkl.dump(result_dict, f)
        print("done saving results at ", save_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="/home2/tathagato/summarization/MACSUM/naacl/configs/hlora_sft.yaml")
    parser.add_argument("--experiment_name", type=str, default="llama_length_and_extractiveness_test")
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
        config['num_epochs'] = 1

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
    # model = AutoModelForCausalLM.from_pretrained(
    #     experiment_config["model_id"],
    #     use_cache=False,
    #     quantization_config=bnb_config,
    #     device_map="cuda:0"
    # )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r = config["rank_1"],
        lora_alpha = config["lora_alpha_1"],
        target_modules= config["target_modules"],
        lora_dropout= config["lora_dropout"],
        inference_mode = False 
        
    )

    hlora_config = HLORAConfig(
        lora_rank_1=config["rank_1"],
        lora_rank_2=config["rank_2"],
        lora_alpha_1=config["lora_alpha_1"],
        lora_alpha_2=config["lora_alpha_2"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"]
    )    
    assert len(experiment_config["attributes"]) >  1, "Atleast 2 attributes are needed for multi attribute training"


    model, tokenizer = setup_hlora_model(config['model_id'], lora_config, hlora_config, bnb_config)
    model = set_train_adapters(model, level_1=True, level_2=False)
    model = set_inference_adapters(model, level_1=True, level_2=False)
    model = set_gradients_on_the_model(model)
    


    #load the dataset
    if args.debug:
        train_datasets = [
            MACSUM(
                dataset_path = config["train_dataset_path"],
                attributes = experiment_config["attributes"][:1],
                tokenizer = tokenizer,
                mode = "train",
                size = 16,
                max_seq_len = config["max_seq_len"],
                model_type = experiment_config["model_type"]
                ),
            MACSUM(
                dataset_path = config["train_dataset_path"],
                attributes = experiment_config["attributes"][1:],
                tokenizer = tokenizer,
                mode = "train",
                size = 16,
                max_seq_len = config["max_seq_len"],
                model_type = experiment_config["model_type"]
                )
        ]
        val_datasets = [
            MACSUM(
                dataset_path = config["val_dataset_path"],
                attributes = experiment_config["attributes"][:1],
                tokenizer = tokenizer,
                mode = "train",
                size = 4,
                max_seq_len = config["max_seq_len"],
                model_type = experiment_config["model_type"]
                ),
            MACSUM(
                dataset_path = config["val_dataset_path"],
                attributes = experiment_config["attributes"][1:],
                tokenizer = tokenizer,
                mode = "train",
                size = 4,
                max_seq_len = config["max_seq_len"],
                model_type = experiment_config["model_type"]
                ),
            MACSUM(
                dataset_path= config["val_dataset_path"],
                attributes = experiment_config["attributes"],
                tokenizer = tokenizer,
                mode = "inference",
                size = 4,
                max_seq_len = config["max_seq_len"],
                model_type = experiment_config["model_type"]
            )
        ]
        test_datasets = [
            # MACSUM(
            #     dataset_path = config["test_dataset_path"],
            #     attributes = experiment_config["attributes"][:1],
            #     tokenizer = tokenizer,
            #     mode = "inference",
            #     size = 4,
            #     max_seq_len = config["max_seq_len"],
            #     model_type = experiment_config["model_type"]
            # ),
            # MACSUM(
            #     dataset_path = config["test_dataset_path"],
            #     attributes = experiment_config["attributes"][1:],
            #     tokenizer = tokenizer,
            #     mode = "inference",
            #     size = 4,
            #     max_seq_len = config["max_seq_len"],
            #     model_type = experiment_config["model_type"]
            # ),
            MACSUM(
                dataset_path = config["test_dataset_path"],
                attributes = experiment_config["attributes"],
                tokenizer = tokenizer,
                mode = "inference",
                size = 4,
                max_seq_len = config["max_seq_len"],
                model_type = experiment_config["model_type"]
            )

        ]
    else:
        train_datasets = [
            MACSUM(
                dataset_path = config["train_dataset_path"],
                attributes = experiment_config["attributes"][:1],
                tokenizer = tokenizer,
                mode = "train",
                size = -1,
                max_seq_len = config["max_seq_len"],
                model_type = experiment_config["model_type"]
                ),
            MACSUM(
                dataset_path = config["train_dataset_path"],
                attributes = experiment_config["attributes"][1:],
                tokenizer = tokenizer,
                mode = "train",
                size = -1,
                max_seq_len = config["max_seq_len"],
                model_type = experiment_config["model_type"]
                )
        ]
        val_datasets = [
            MACSUM(
                dataset_path = config["val_dataset_path"],
                attributes = experiment_config["attributes"][:1],
                tokenizer = tokenizer,
                mode = "train",
                size = -1,
                max_seq_len = config["max_seq_len"],
                model_type = experiment_config["model_type"]
                ),
            MACSUM(
                dataset_path = config["val_dataset_path"],
                attributes = experiment_config["attributes"][1:],
                tokenizer = tokenizer,
                mode = "train",
                size = -1,
                max_seq_len = config["max_seq_len"],
                model_type = experiment_config["model_type"]
                ),
            MACSUM(
                dataset_path= config["val_dataset_path"],
                attributes = experiment_config["attributes"],
                tokenizer = tokenizer,
                mode = "inference",
                size = -1,
                max_seq_len = config["max_seq_len"],
                model_type = experiment_config["model_type"]
            )
        ]
        test_datasets = [
            # MACSUM(
            #     dataset_path = config["test_dataset_path"],
            #     attributes = experiment_config["attributes"][:1],
            #     tokenizer = tokenizer,
            #     mode = "inference",
            #     size = -1,
            #     max_seq_len = config["max_seq_len"],
            #     model_type = experiment_config["model_type"]
            # ),
            # MACSUM(
            #     dataset_path = config["test_dataset_path"],
            #     attributes = experiment_config["attributes"][1:],
            #     tokenizer = tokenizer,
            #     mode = "inference",
            #     size = -1,
            #     max_seq_len = config["max_seq_len"],
            #     model_type = experiment_config["model_type"]
            # ),
            MACSUM(
                dataset_path = config["test_dataset_path"],
                attributes = experiment_config["attributes"],
                tokenizer = tokenizer,
                mode = "inference",
                size = -1,
                max_seq_len = config["max_seq_len"],
                model_type = experiment_config["model_type"]
            )

        ]
    
    # train dataset size 
    for dataset in train_datasets:
        print(f"Train dataset {dataset.attributes} size: {len(dataset)}")
    # val dataset size
    for dataset in val_datasets:
        print(f"Val dataset {dataset.attributes} size: {len(dataset)}")
    # test dataset size
    for dataset in test_datasets:
        print(f"Test dataset {dataset.attributes} size: {len(dataset)}")


    initial_state_dict = get_detached_state_dict(model)
    if args.debug:
        print_layerwise_details(model)
        print_layers_with_requires_grad(model)
        model.print_trainable_parameters()

    #first training 
    train(model, tokenizer, train_datasets[0], val_datasets[0], config, device=0, save_pretrained= False, do_wandb=config['do_wandb'])

    #output_generation(model, test_datasets, prefix = "intermediate", config = config)

    print("first training done")
    intermediate_state_dict = get_detached_state_dict(model)

    model = set_train_adapters(model, level_1=False, level_2=True)
    model = set_inference_adapters(model, level_1=True, level_2=True)
    model = set_gradients_on_the_model(model)

    if args.debug:
        print_layers_with_requires_grad(model)
        model.print_trainable_parameters()

    #second training
    train(model, tokenizer, train_datasets[1], val_datasets[1], config, device=0, save_pretrained= False, do_wandb=config['do_wandb'])

    if args.debug:
        final_state_dict = get_detached_state_dict(model)

        print("Checking the changed layers between the initial and intermediate state dict")
        return_changed_layers(initial_state_dict, intermediate_state_dict)

        print("---------------------------------------------------------------------------")
        print("Checking the changed layers between the intermediate and final state dict")
        return_changed_layers(intermediate_state_dict, final_state_dict)

    # import code; code.interact(local=locals());
    # exit()

    output_generation(model, test_datasets, prefix = "final", config = config)

    print("done generating text")







