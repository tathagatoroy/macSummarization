
## IMPLEMENTATION NOTES:
Implementation notes and details
    * certain hyperparameters will be kept constant for all experiments. Examples :
        * quant_config
        * peft config
        * use_4bit 
        * lora_r = 32
        * lora_alpha = 16
    For each experiment subtype they are stored under global and specific parameters are stored inside the experiment type

    * for now when selecting the checkpoints when training using attribute 2 on top of attribute 1 checkpoint I am using attribute 2 val loss for selecting the best attribute and not the best joint eval loss on both the attributes
    
    * I have access to 4 gpus but I am not getting access to 4 gpus in sbatch but it is working in srun. Not sure why. Will have to run 2 sbatch of 2 gpus each. Will make the folder untidy but no option

    