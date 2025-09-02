from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm.auto import tqdm
import json
import torch
import copy
from typing import List, Dict, Any, Tuple, Optional
import datasets


# Dataset class
class MACSUM(Dataset):
    def __init__(self, dataset_path = "/home2/tathagato/summarization/MACSUM/dataset/macdoc/val_dataset.json", attributes:Optional[List] = None, tokenizer = None, mode = 'inference', size = -1,max_seq_len = 2048, model_type = 'llama'):
        self.dataset_path = dataset_path
        self.dataset = json.load(open(dataset_path,"r"))
        self.size = size 
        self.tokenizer = tokenizer
        self.attributes = attributes
        self.filter_by_attribute()
        self.mode = mode
        self.system_prompt = "You are an honest and to the point assistant, please follow the instruction and answer to the point. Please do not provide any irrelevant information or add any extra words than that is necessary to answer the question."
        self.model_type = model_type
        self.max_seq_len = max_seq_len
        if self.size != -1:
            self.dataset = dict(list(self.dataset.items())[:self.size])
            self.index_to_keys = list(self.dataset.keys())
        else:
            self.size = len(self.dataset)



    
    def generate_attribute_specific_instruction(self,control_attributes):
        base_prompt = f"Write a summary of the source text."
        ca_aspects = ""
        for attr in self.attributes:
            control_value = control_attributes[attr]
            if attr == 'length':
                ca_aspect = f"The summary should be {control_value} in length. The length is defined in terms of number of words used in the summary."
            elif attr == 'extractiveness':
                ca_aspect = f"The summary should be {control_value} in extractiveness. Extractiveness is defined by the degree of exact copying from the source text."
            elif attr == 'specificity':
                ca_aspect = f"The summary should be {control_value} in specificity. Specificity is defined by the degree of detail in the summary."
            elif attr == 'topic':
                ca_aspect = f"The summary should be focussed on the topic {control_value}."
            elif attr == 'Speaker':
                ca_aspect = f"The summary should be written from the perspective of {control_value}."
            ca_aspects += ca_aspect
        #prompt = f"{base_prompt} {ca_aspect}. The source text is given below. "
        instruction = f"{base_prompt} {ca_aspects}. The source text is given below. "
        return instruction


    
    def filter_by_attribute(self):
        tmp_dataset = {}
        for key , value in self.dataset.items():
            #joint dataset : all the attributes have to be non empty
            valid = True 
            for attr in self.attributes:
                if value['control_attribute'][attr] == "":
                    valid = False
                    break
            if valid:
                tmp_dataset[key] = value
        self.dataset = tmp_dataset
        self.index_to_keys = list(self.dataset.keys())


    def format_data_llama31(self, instruction, input_text, output):
        # Construct the full text in Llama 3.1 format
        full_text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|> {self.system_prompt}<|eot_id|>"
        full_text += f"<|start_header_id|>user<|end_header_id|> {instruction}\n\n{input_text}<|eot_id|>"
        prompt = full_text + f"<|start_header_id|>assistant<|end_header_id|>"
        full_text += f"<|start_header_id|>assistant<|end_header_id|> {output}<|eot_id|>"

        return full_text, prompt
    def format_data_mistral(self, instruction, input_text, output):
        #mistral doesn't require system prompt
        # Construct the full text in Mistral format
        full_text = f"<s>[INST] {self.system_prompt} {instruction}"
        full_text += f" {input_text}[/INST] "
        prompt = full_text 
        answer = prompt + f"{output}</s>"
        return answer, prompt



    
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
        src = self.dataset[self.index_to_keys[index]]['source']
        reference = self.dataset[self.index_to_keys[index]]['reference']

        instruction = self.generate_attribute_specific_instruction(self.dataset[self.index_to_keys[index]]['control_attribute'])

        if self.model_type == 'mistral':
            example , prompt = self.format_data_mistral(instruction, src, reference)
        elif self.model_type == 'llama':
            example , prompt = self.format_data_llama31(instruction, src, reference)
        #example = prompt + reference
        controllability_dict = {attr : self.dataset[self.index_to_keys[index]]['control_attribute'][attr] for attr in self.attributes}
        control_values = [controllability_dict[attr] for attr in self.attributes]
        control_attributes = [attr for attr in self.attributes]

        tokenized_prompt = torch.tensor(
            self.tokenizer.encode(prompt, add_special_tokens = False), dtype=torch.int64
        )
        example = self.tokenizer.encode(example, add_special_tokens = False)
        #example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )

        labels = copy.deepcopy(example)
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        if self.mode == 'inference':
            return {
                "input_ids": tokenized_prompt.unsqueeze(0),
                "output" : reference,
                "input": src,
                "prompt": prompt,
                "control_value": control_values,
                "control_attribute": self.attributes

            }
        else:

            return {
                "input_ids": example,
                "labels": labels,
                "attention_mask":example_mask
            }


class dpo_dataset(Dataset):

    def __init__(self, dataset_path = "/home2/tathagato/summarization/MACSUM/dataset/macdoc/train.json", attributes = ['length'], tokenizer = None, mode = 'train', model_type = "llama", size = -1):
        self.dataset_path = dataset_path
        self.data = json.load(open(dataset_path,"r"))
        self.size = size 
        self.tokenizer = tokenizer
        self.attributes = attributes
        self.generate_dpo_pairs()
        self.mode = mode
        self.model_type = model_type
        self.system_prompt = "You are an honest and to the point assistant, please follow the instruction and answer to the point. Please do not provide any irrelevant information or add any extra words than that is necessary to answer the question."
        self.keys = list(self.dataset.keys())
        if self.size != -1:
            self.dataset = {key : self.dataset[key] for key in self.keys[:self.size]}
    def generate_dpo_pairs(self):
        dataset = {}
        new_idx = 0
        for idx, example in enumerate(self.data):
            #keys is source : input text
            # references : List of dicts which contain subdict control_attribute, summary
            input_text = example['source'][0]
            references = example['references']
            for idx_1, reference_1 in  enumerate(references):
                #check if all attributes are present 
                control_attributes = reference_1['control_attribute']
                is_valid = True
                for attr in self.attributes:
                    if control_attributes[attr] == "":
                        is_valid = False
                        break
                if is_valid:
                    for idx_2, reference_2 in enumerate(references):
                        if idx_1 != idx_2:
                            control_attributes_2 = reference_2['control_attribute']
                            #check values chosen attributes do not match 
                            is_valid_pair = True
                            for attr in self.attributes:
                                if control_attributes[attr] == control_attributes_2[attr]:
                                    is_valid_pair = False
                                    break
                            if is_valid_pair:
                                new_datapoint = {
                                    "source" : input_text,
                                    "chosen" : reference_1['summary'],
                                    "rejected" : reference_2['summary'],
                                    "chosen_raw" : reference_1['control_attribute'],
                                    "rejected_raw" : reference_2['control_attribute'],
                                    "control_attributes" : self.attributes,
                                    "control_values" : [control_attributes[attr] for attr in self.attributes]

                                
                                }
                                dataset[new_idx] = new_datapoint
                                new_idx += 1
        self.dataset = dataset

                        

                
    def __len__(self):
        return len(self.dataset)

    def generate_attribute_specific_instruction(self,control_attributes):
        base_prompt = f"Write a summary of the source text."
        ca_aspects = ""
        for attr in self.attributes:
            control_value = control_attributes[attr]
            if attr == 'length':
                ca_aspect = f"The summary should be {control_value} in length. The length is defined in terms of number of words used in the summary."
            elif attr == 'extractiveness':
                ca_aspect = f"The summary should be {control_value} in extractiveness. Extractiveness is defined by the degree of exact copying from the source text."
            elif attr == 'specificity':
                ca_aspect = f"The summary should be {control_value} in specificity. Specificity is defined by the degree of detail in the summary."
            elif attr == 'topic':
                ca_aspect = f"The summary should be focussed on the topic {control_value}."
            elif attr == 'Speaker':
                ca_aspect = f"The summary should be written from the perspective of {control_value}."
            ca_aspects += ca_aspect
        #prompt = f"{base_prompt} {ca_aspect}. The source text is given below. "
        instruction = f"{base_prompt} {ca_aspects}. The source text is given below. "
        return instruction

    def format_dpo_data_llama(self, instruction, input_text, chosen, rejected):
        
        # full_text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|> {self.system_prompt}<|eot_id|>"
        # full_text += f"<|start_header_id|>user<|end_header_id|> {instruction}\n\n{input_text}<|eot_id|>"
        # prompt = full_text + f"<|start_header_id|>assistant<|end_header_id|>"
        # full_text += f"<|start_header_id|>assistant<|end_header_id|> {output}<|eot_id|>"

        system_prompt = f"<|start_header_id|>system<|end_header_id|> {self.system_prompt}<|eot_id|>"
        prompt = system_prompt + f"<|start_header_id|>user<|end_header_id|> \n\n{instruction}\n{input_text}<|eot_id|>"
        chosen = f"<|start_header_id|>assistant<|end_header_id|>\n\n{chosen}"
        rejected = f"<|start_header_id|>assistant<|end_header_id|>\n\n{rejected}"
        return prompt, chosen, rejected
    
    
    def format_dpo_data_mistral(self, instruction, input_text , chosen, rejected):
        prompt = f"[INST] {instruction}.\n{input_text} [/INST]"
        chosen = f"{chosen}"
        rejected = f"{rejected}"
        return prompt, chosen, rejected
    

    def __getitem__(self, index):
        #IGNORE_INDEX = -100
        source = self.dataset[index]['source']
        prefered_control_attributes = self.dataset[index]['control_attributes'] 
        chosen = self.dataset[index]['chosen']
        rejected = self.dataset[index]['rejected']
        chosen_raw = self.dataset[index]['chosen_raw']
        rejected_raw = self.dataset[index]['rejected_raw']
        instruction = self.generate_attribute_specific_instruction(chosen_raw) 
        prompt = instruction
        if self.model_type == "llama":
            prompt, chosen, rejected = self.format_dpo_data_llama(instruction, source, chosen, rejected)
        elif self.model_type == "mistral":
            prompt, chosen, rejected = self.format_dpo_data_mistral(instruction, source, chosen, rejected)
        examples = {
            "prompt" : prompt,
            "chosen" : chosen,
            "rejected" : rejected,
            "prefered_control_attributes" : prefered_control_attributes,
            "chosen_raw" : chosen_raw,
            "rejected_raw" : rejected_raw
        }
        return examples


    
def get_huggingface_dataset(dataset):

    dataset_dict = {}
    keys = dataset[0].keys()
    size = len(dataset)
    for key in keys:
        dataset_dict[key] = []
    for idx in range(size):
        example = dataset[idx]
        for key in keys:
            dataset_dict[key].append(example[key])
    dataset = datasets.Dataset.from_dict(dataset_dict)
    return dataset

def display_inference_dataset(dataset):
    example = dataset[0]
    for key, value in example.items():
        print(f"{key}: {value}")
    print("text with special tokens")
    print(tokenizer.decode(example['input_ids'][0]))
    print("text without special tokens")
    print(tokenizer.decode(example['input_ids'][0], skip_special_tokens=True))

def display_train_dataset(dataset):
    example = dataset[0]
    for key, value in example.items():
        print(f"{key}: {value}")
    print("text with special tokens")
    print(tokenizer.decode(example['input_ids']))
    print("text without special tokens")
    print(tokenizer.decode(example['input_ids'], skip_special_tokens=True))


if __name__=='__main__':
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    #import huggingface tokenizers from transformers
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(tokenizer.special_tokens_map)
    print(tokenizer.all_special_ids)
    print(tokenizer.all_special_tokens)
    
    model_ids = ['mistral','llama']
    dataset_modes = ['train','inference']
    dataset_attributes = [['length'],['extractiveness'],['specificity'],['topic'],['length', 'extractiveness']]
    dataset_path = "/home2/tathagato/summarization/MACSUM/dataset/macdoc/val_dataset.json"
    for model_id in model_ids:
        for mode in dataset_modes:
            for attribute in dataset_attributes:
                dataset = MACSUM(dataset_path = dataset_path, attributes = attribute, tokenizer = tokenizer, mode = mode, model_type = model_id)
                print(f"Model: {model_id} | Mode: {mode} | Attribute: {attribute}")
                print(f"Length of dataset: {len(dataset)}")
                if mode == 'train':
                    display_train_dataset(dataset)
                else:
                    display_inference_dataset(dataset)
                print("------------------------------------------------------------")



