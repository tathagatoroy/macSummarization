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
from dataset import MACSUM, dpo_dataset, get_huggingface_dataset
import yaml
from peft import LoraConfig, TaskType
import wandb
from trl import DPOConfig, DPOTrainer
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#python multi_attribute_joint_training_dpo.py --debug > logs/sft_multi.txt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="/home2/tathagato/summarization/MACSUM/naacl/configs/joint_multi_attribute_dpo.yaml")
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
    assert len(experiment_config["attributes"]) > 1, "there needs to be more than one attribute"
    adapter_name = "_and_".join(experiment_config["attributes"])
    model = get_single_adapter_lora_model(model, lora_config, adapter_name)



    #load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(experiment_config["model_id"])

    #load the dataset
    if args.debug:
        train_dataset = dpo_dataset(
            dataset_path= config["train_dataset_path"],
            attributes = experiment_config["attributes"],
            tokenizer = tokenizer,
            mode = "train",
            model_type= experiment_config["model_type"],
            size = 16        
            )
        val_dataset = dpo_dataset(
            dataset_path= config["val_dataset_path"],
            attributes = experiment_config["attributes"],
            tokenizer = tokenizer,
            mode = "train",
            model_type= experiment_config["model_type"],
            size = 16
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
        train_dataset = dpo_dataset(
            dataset_path= config["train_dataset_path"],
            attributes = experiment_config["attributes"],
            tokenizer = tokenizer,
            mode = "train",
            model_type= experiment_config["model_type"],
            size = -1        
            )
        val_dataset = dpo_dataset(
            dataset_path= config["val_dataset_path"],
            attributes = experiment_config["attributes"],
            tokenizer = tokenizer,
            mode = "train",
            model_type= experiment_config["model_type"],
            size = -1
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
    print(f"Length of val dataset: {len(val_dataset)}")
    for i, test_dataset in enumerate(test_datasets):
        print(f"Length of test dataset with attributes {test_dataset.attributes} : {len(test_dataset)}")
    print(model)
    get_adapter_status(model)
    model.print_trainable_parameters()

    train_dataset = get_huggingface_dataset(train_dataset)
    val_dataset = get_huggingface_dataset(val_dataset)
    tokenizer.pad_token = tokenizer.eos_token
    training_args = DPOConfig(
        output_dir= config['output_dir'], 
        learning_rate= config['learning_rate'],
        num_train_epochs= config['num_epochs'],
        per_device_train_batch_size= config['batch_size'],
        per_device_eval_batch_size= config['batch_size'],
        gradient_accumulation_steps= config['gradient_accumulation_steps'],
        max_grad_norm= config['max_grad_norm'],
        eval_steps= config['eval_interval'],
        save_steps= config['eval_interval'],
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        logging_steps= config['logging_steps'],
        optim = config['optim'],
        warmup_ratio= config['warmup_ratio'],
        lr_scheduler_type = config['lr_scheduler_type'],
        ddp_find_unused_parameters=config['ddp_find_unused_parameters'],
        evaluation_strategy = "steps",
        save_strategy= "steps",
        eval_on_start=True ,
        report_to = "wandb" if config['do_wandb'] else "none",
        max_prompt_length= config['max_prompt_length'],
        max_length = config['max_seq_len']
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args, 
        tokenizer=tokenizer, 
        max_length=config['max_seq_len'],
        is_encoder_decoder=False,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
        )
    trainer.train()

    trainer.evaluate()

    #save the current checkpoint as the best


    print("training done")

    #loading the best checkpoint
    #delete the model from memory and free up the memory 


    #print the active adapter
    get_adapter_status(model)

    for i, test_dataset in tqdm.tqdm(enumerate(test_datasets)):
        print(f"generating on test dataset with attributes {test_dataset.attributes}")
        result_dict = generate_text(model, test_dataset, tokenizer = tokenizer, config = config)
        print("done generating text for test dataset with attributes ", test_dataset.attributes)
        result_adapter_name = "_and_".join(test_dataset.attributes)
        with open(os.path.join(config["output_dir"], f"model_{adapter_name}_results_{result_adapter_name}.pkl"), "wb") as f:
            pkl.dump(result_dict, f)

        print("done saving results at ", os.path.join(config["output_dir"], f"model_{adapter_name}_results_{result_adapter_name}.pkl"))





