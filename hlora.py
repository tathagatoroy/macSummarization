import torch
import torch.nn as nn
from peft import PeftConfig, PeftModel,LoraConfig
from peft.tuners.lora import LoraLayer
from peft.utils import PeftType
from transformers.utils import PushToHubMixin
from bitsandbytes.nn import Linear4bit
from peft import PeftModel, LoraConfig, get_peft_model,prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import wraps
import sys 
sys.path.insert(0, "/home2/tathagato/summarization/MACSUM/fsdp_lora")
from dataset import MACSUM

def replace_lora_with_hlora(model, hlora_config):
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            base_layer = module.get_base_layer()
            if isinstance(base_layer, Linear4bit) and getattr(base_layer, "compute_dtype", None) is not None:
                # Replace LoraLayer with HLORA
                new_module = HLORA(base_layer, hlora_config)
                # Get the parent module
                parent_name, child_name = name.rsplit('.', 1)
                parent_module = model.get_submodule(parent_name)
                # Replace the old module with the new one
                setattr(parent_module, child_name, new_module)
    model.do_inference = [True, False]
    model.do_train = [True, False]
    model.set_inference_adapters(level_1=True, level_2=False)
    model.set_train_adapters(level_1=True, level_2=False)
    model.set_gradients()
    return model



class HLORAConfig(LoraConfig):
    def __init__(self, lora_rank_1=32, lora_rank_2=16, lora_alpha_1=16, lora_alpha_2=8, lora_dropout=0.1, target_modules = ["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"],**kwargs):
        super().__init__(**kwargs)
        self.peft_type = PeftType.LORA
        self.lora_rank_1 = lora_rank_1
        self.lora_rank_2 = lora_rank_2
        self.lora_alpha_1 = lora_alpha_1
        self.lora_alpha_2 = lora_alpha_2
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules

class HLORA(nn.Module):
    def __init__(self, base_layer, config):
        super().__init__()

        self.config = config
        self.base_layer = base_layer
        
        dtype = getattr(base_layer, "compute_dtype", next(base_layer.parameters()).dtype)
        device = next(base_layer.parameters()).device

        lora_A = nn.Linear(base_layer.in_features, config.lora_rank_1, bias=False, device=device, dtype=dtype)
        lora_B = nn.Linear(config.lora_rank_1, base_layer.out_features, bias=False, device=device, dtype=dtype)
        #lora_B.weight.data.zero_()
        #kaiming uniform for lora 
        nn.init.kaiming_uniform_(lora_A.weight, a=5**0.5)
        nn.init.kaiming_uniform_(lora_B.weight, a=5**0.5)

        lora_C = nn.Linear(base_layer.in_features, config.lora_rank_2, bias=False, device=device, dtype=dtype)
        lora_D = nn.Linear(config.lora_rank_2, config.lora_rank_1, bias=False, device=device, dtype=dtype)
        lora_E = nn.Linear(config.lora_rank_1, config.lora_rank_2, bias=False, device=device, dtype=dtype)
        lora_F = nn.Linear(config.lora_rank_2, base_layer.out_features, bias=False, device=device, dtype=dtype)
        #lora_F.weight.data.zero_()
        #kaiming uniform for lora C,D,E
        nn.init.kaiming_uniform_(lora_C.weight, a=5**0.5)
        nn.init.kaiming_uniform_(lora_D.weight, a=5**0.5)
        nn.init.kaiming_uniform_(lora_E.weight, a=5**0.5)
        nn.init.kaiming_uniform_(lora_F.weight, a=5**0.5)



        self.lora_A1B1 = nn.Sequential(
            lora_C, lora_D,lora_E, lora_F
        )
        self.lora_AB = nn.Sequential(lora_A, lora_B)

        self.scaling_1 = self.config.lora_alpha_1 / self.config.lora_rank_1
        self.scaling_2 = self.config.lora_alpha_2 / self.config.lora_rank_2
        self.lora_dropout = nn.Dropout(config.lora_dropout)

        self.do_inference = [True, False]
        self.do_train = [True, False]

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        result = self.base_layer(x, *args, **kwargs)
        result = result.clone()

        #self.lora_AB.requires_grad = self.do_train[0]
        #self.lora_A1B1.requires_grad = self.do_train[1]

        requires_conversion = not torch.is_autocast_enabled()
            #result += output

        if self.do_inference[0]:
            #print("Inference 0")
            if requires_conversion:
                expected_dtype = result.dtype
                x = x.to(next(iter(self.lora_AB)).weight.dtype)
            output = self.lora_AB(self.lora_dropout(x))
            if requires_conversion:
                output = output.to(expected_dtype)
            output = (output ) * self.scaling_1
            result += output

        if self.do_inference[1]:
            #print("Inference 1")
            if requires_conversion:
                expected_dtype = result.dtype
                x = x.to(next(iter(self.lora_A1B1)).weight.dtype)
            output = self.lora_A1B1(self.lora_dropout(x))
            if requires_conversion:
                output = output.to(expected_dtype)
            output = output * self.scaling_2 * self.scaling_1
            result += output

        return result

# a decorator to add methods to a class which accepts self. Taken from https://mgarod.medium.com/dynamically-add-a-method-to-a-class-in-python-c49204b85bd6
def add_method(cls):
    def decorator(func):
        @wraps(func) 
        def wrapper(self, *args, **kwargs): 
            return func(*args, **kwargs)
        setattr(cls, func.__name__, wrapper)
        # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
        return func # returning func means func can still be used normally
    return decorator

#TODO : decorator to add methods to a class is not working as expected. Need to fix it
# hence adding the methods to the class directly
def set_train_adapters(model, level_1=True, level_2=False):
    for module in model.modules():
        if isinstance(module, HLORA):
            module.do_train = [level_1, level_2]
    model.do_train = [level_1,level_2]
    return model

def set_inference_adapters(model, level_1=True, level_2=False):
    for module in model.modules():
        if isinstance(module, HLORA):
            module.do_inference = [level_1, level_2]
    model.do_inference = [level_1, level_2]
    return model
def set_gradients_on_the_model(model):
    for param in model.base_model.parameters():
        param.requires_grad = False

    # Set HLORA module gradients
    # for module in model.modules():
    #     if isinstance(module, HLORA):
    #         # Set lora_AB gradients
    #         for param in module.lora_AB.parameters():
    #             param.requires_grad = module.do_train[0]

    #         # Set lora_A1B1 gradients
    #         for param in module.lora_A1B1.parameters():
    #             param.requires_grad = module.do_train[1]

    for name, param in model.named_parameters():
        if "lora_AB" in name:
            param.requires_grad = model.do_train[0]
            #print(f"setting {name} to {model.do_train[0]}, {param.requires_grad}")
        elif "lora_A1B1" in name:
            param.requires_grad = model.do_train[1]
            #print(f"setting {name} to {model.do_train[1]} , {param.requires_grad}")
        else:
            param.requires_grad = False
            #print(f"setting {name} to False, {param.requires_grad}")
    return model


@add_method(PeftModel)    
def set_gradients(self):
    """ set the gradient of all modules in the model
        base model : requires grad always false
        lora_AB = nn.Sequential(self.lora_A, self.lora_B) req_grad = False if self.lora_AB.do_train[0] is False else true 
        lora_A1B1 = nn.Sequential(lora_A1B1 = nn.Sequential(nn.Sequential(self.lora_C, self.lora_D),nn.Sequential(self.lora_E, self.lora_F)) if self.lora_A1B!.do_train[False]
    """
    for param in self.base_model.parameters():
        param.requires_grad = False

    # Set HLORA module gradients
    for module in self.modules():
        if isinstance(module, HLORA):
            # Set lora_AB gradients
            for param in module.lora_AB.parameters():
                param.requires_grad = self.do_train[0]

            # Set lora_A1B1 gradients
            for param in module.lora_A1B1.parameters():
                param.requires_grad = self.do_train[1]








def replace_linear4bit_with_hlora(model, peft_config):
    for name, module in model.named_children():
        if isinstance(module, Linear4bit) and getattr(module, "compute_dtype", None) is not None:
            # Check if it's a Linear4bit
            #print("setting attribute")
            setattr(model, name, HLORA(module, peft_config))
        else:
            replace_linear4bit_with_hlora(module, peft_config)
    return model

def replace_lora_with_hlora(model, hlora_config):
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            base_layer = module.get_base_layer()
            if isinstance(base_layer, Linear4bit) and getattr(base_layer, "compute_dtype", None) is not None:
                # Replace LoraLayer with HLORA
                new_module = HLORA(base_layer, hlora_config)
                # Get the parent module
                parent_name, child_name = name.rsplit('.', 1)
                parent_module = model.get_submodule(parent_name)
                # Replace the old module with the new one
                setattr(parent_module, child_name, new_module)
    return model


def setup_hlora_model(model_id: str, lora_config: LoraConfig, hlora_config: HLORAConfig, bnb_config):
    # 1. Load the model
    model = AutoModelForCausalLM.from_pretrained(model_id,quantization_config = bnb_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 2. Get PEFT model
    peft_model = get_peft_model(model, lora_config)


    # 3. Replace lora layers with HLORA
    peft_model = replace_lora_with_hlora(peft_model, hlora_config)



    # 4. Set which adapters to use for training, inference and gradients    
    # hlora_model= set_train_adapters(hlora_model, level_1=True, level_2=False)
    # hlora_model = set_inference_adapters(hlora_model, level_1=True, level_2=False)
    # hlora_model = set_gradients_on_the_model(hlora_model)

    #print_model_layer_info(hlora_model)
    prepare_model_for_kbit_training(peft_model)

    return peft_model, tokenizer
def print_model_layer_info(model):
    for name, param in model.named_parameters():
        #print(name, param.shape)
        #print name , shape, device, dtype and requires_grad or not
        if param.requires_grad:
            print(f"Name: {name}, Shape: {param.shape}, Device: {param.device}, Dtype: {param.dtype}, Requires Grad: {param.requires_grad}")
    print("\n\n")
    for name, param in model.named_parameters():
        if param.requires_grad == False:
            print(f"Name: {name}, Shape: {param.shape}, Device: {param.device}, Dtype: {param.dtype}, Requires Grad: {param.requires_grad}")
    print("-----------------\n\n------------------------------")
# Test script
def test_hlora_replacement():
    # 1. Load the base model
    model_id = "akjindal53244/Llama-3.1-Storm-8B"
    model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 2. Initialize LORA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )


    # 3. Get PEFT model
    peft_model = get_peft_model(model, lora_config)

    # 4. Define HLORA config
    hlora_config = HLORAConfig(
        lora_rank_1=32,
        lora_rank_2=16,
        lora_alpha_1=16,
        lora_alpha_2=8,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    print(peft_model)
    lora_count = sum(1 for m in peft_model.modules() if isinstance(m, LoraLayer))


    # 5. Replace LORA with HLORA
    hlora_model = replace_lora_with_hlora(peft_model, hlora_config)

    # 6. Verify replacement
    hlora_count = sum(1 for m in hlora_model.modules() if isinstance(m, HLORA))
    
    print(f"Number of LORA layers before replacement: {lora_count}")
    print(f"Number of HLORA layers after replacement: {hlora_count}")

    # # 7. Test inference
    # input_text = "Once upon a time"
    # input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(hlora_model.device)
    
    # with torch.no_grad():
    #     output = hlora_model.generate(input_ids, max_length=50)
    
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(f"Generated text: {generated_text}")

    set_train_adapters(hlora_model, level_1=True, level_2=False)
    set_inference_adapters(hlora_model, level_1=True, level_2=False)
    print(hlora_model.do_inference)
    print(hlora_model.do_train)
    print(hlora_model)
    print_model_layer_info(hlora_model)
    set_gradients_on_the_model(hlora_model)
    print("------------------------")
    print_model_layer_info(hlora_model)
    dataset = MACSUM(attribute = "length", tokenizer = tokenizer, mode = 'train', size = 20, model_type= "llama31")
    print(dataset[0])
    input_ids = dataset[0]['input_ids']
    labels = dataset[0]['labels']
    input_ids = input_ids.unsqueeze(0)
    labels = labels.unsqueeze(0)
    res = hlora_model(input_ids, labels = labels)
    loss = res.loss
    logits = res.logits
    print(loss.shape)
    print(loss)
    print(logits.shape)



if __name__ == "__main__":
    test_hlora_replacement()