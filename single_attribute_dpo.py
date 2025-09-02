import torch
from transformers import LlamaForCausalLM, BitsAndBytesConfig, AutoModelForCausalLM
import os 
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import sys
import pickle as pkl
import tqdm
import time
from utils import generate_text, get_single_adapter_lora_model, merge_configs, load_peft_checkpoint, get_adapter_status, print_layerwise_details, load_dpo_adapter_model
from dataset import dpo_dataset, get_huggingface_dataset, MACSUM
import yaml
from peft import LoraConfig, TaskType
import wandb
from trl import DPOConfig, DPOTrainer

#do not parallelize the tokenizer 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#CUDA_VISIBLE_DEVICES=0, python single_attribute_dpo.py --debug > logs/sft_single.txt
#CUDA_VISIBLE_DEVICES=0, python single_attribute_dpo.py | tee output.txt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="/home2/tathagato/summarization/MACSUM/naacl/configs/single_attribute_dpo.yaml")
    parser.add_argument("--experiment_name", type=str, default="llama_length")
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
    report_to = "None"
    if config['do_wandb'] :
        wandb.init(project = config["wandb_project"], name = args.experiment_name)
        wandb.config.update(config)
        report_to = "wandb"

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

    
    print(model)
    get_adapter_status(model)
    model.print_trainable_parameters()



    #load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(experiment_config["model_id"])
    tokenizer.pad_token = tokenizer.eos_token

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
        #for testing we will use MACSUM dataset
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

    train_dataset = get_huggingface_dataset(train_dataset)
    val_dataset = get_huggingface_dataset(val_dataset)


    # num_examples = 5

    # for i in range(num_examples):

    #     example = train_dataset[i]
    #     for key in example.keys():
    #         print(f"{key}: {example[key]}")
    #         print("\n")
    #     print("----------------------------------")

    tokenizer.pad_token = tokenizer.eos_token
    adapter_name = experiment_config["attributes"][0]
    #model = load_dpo_adapter_model(config, quantization_config=bnb_config,adapter_name=adapter_name)

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

    # dataloader = trainer.get_train_dataloader()
    # tokenizer = trainer.tokenizer

    # batch = next(iter(dataloader))
    # print(batch.keys())
    # prompt = batch['prompt'][0]
    # chosen = batch['chosen'][0]
    # rejected = batch['rejected'][0]
    # prompt_input_ids = batch['prompt_input_ids'][0]
    # chosen_input_ids = batch['chosen_input_ids'][0]
    # rejected_input_ids = batch['rejected_input_ids'][0]

    

    # import code; code.interact(local=locals())
    # exit()



    

    print("training done")

    #print the active adapter
    get_adapter_status(model)

    result_dict = generate_text(model, test_dataset, tokenizer = tokenizer, config = config)
    print("done generating text")

    #save the results
    with open(os.path.join(config["output_dir"], "results.pkl"), "wb") as f:
        pkl.dump(result_dict, f)
    print("done saving results at ", os.path.join(config["output_dir"], "results.pkl"))

    #setup wandb 




