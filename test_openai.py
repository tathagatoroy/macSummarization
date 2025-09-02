import os 
from openai import OpenAI
import json
import pickle as pkl 
import tqdm
import time

not_topics = [
    'normal',
    'short',
    'long',
    'high',
    'low',
    'fully'
    'full'
]
def convert_chat_completion_to_json(chat_completion):
    # Convert the nested structure to a dictionary
    json_output = {
        "id": chat_completion.id,
        "model": chat_completion.model,
        "object": chat_completion.object,
        "created": chat_completion.created,
        "service_tier": chat_completion.service_tier,
        "system_fingerprint": chat_completion.system_fingerprint,
        "usage": {
            "completion_tokens": chat_completion.usage.completion_tokens,
            "prompt_tokens": chat_completion.usage.prompt_tokens,
            "total_tokens": chat_completion.usage.total_tokens,
            "prompt_tokens_details": chat_completion.usage.prompt_tokens_details,

        },
        "choices": [
            {
                "index": choice.index,
                "finish_reason": choice.finish_reason,
                "logprobs": choice.logprobs,
                "message": {
                    "role": choice.message.role,
                    "content": choice.message.content,
                    "function_call": choice.message.function_call,
                    "tool_calls": choice.message.tool_calls,
                    "refusal": choice.message.refusal
                }
            }
            for choice in chat_completion.choices
        ]
    }
    
     # Convert the dictionary to a JSON string
    #json_output = json.dumps(chat_completion_dict, indent=4)
    
    return json_output

def extract_answer(json_output):
    responses = [res['message']['content'] for res in json_output['choices']]
    return responses


def prompt_template(document, summary, topics):
    prompt = f"You will be given one summary written for a news article along with the news article.\n \
Your task is to rate the summary on one metric, that is topical coherence. \n \
Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.\n\
Evaluation Criteria:\n \
Topical Coherence:Ensure the summary focuses on the topics(there can more than one topic) requested by the user.The summary should reflect the content of the article and not introduce made-up information.It should draw conclusions or summarize only the details presented in the input. The summary should not include any unrelated or extraneous information that is not aligned with the topic or input article.\n\
Evaluation Steps:\n \
1. Read the Input Article : Understand the main topics and points that should be covered in the summary.\n\
2. Read the topic(s) provided by the user\n\
3. Read the Summary: Carefully go through the summary and assess whether it covers the topic in a meaningful and coherent way.\n\
4. Score the Summary:1-5 scale:\n\
     * 1: The summary somewhat reflects the topic but contains a significant amount of irrelevant or incorrect information or misses relevant information.\n\
     * 2: The summary is generally on-topic, but may include minor irrelevant details or miss some key points.\n\
     * 3: The summary is mostly on-topic, covering the requested topic well with very few irrelevant details.\n\
     * 4: The summary  reflects the requested topic with full coherence and no irrelevant or made-up content.\n\
     * 5: The summary is perfectly on-topic, coherent, and includes all the key points from the input article and is very crisp and to the point \n\
5. Just respond with the one score and nothing else.\n\
Example:\n\
Source Text:\n\
{document} \n\
Topics:\n\
{topics}  \n\
Summary: \n\
{summary} \n\
Evaluation Form (scores ONLY): \n\
Topical Coherence:\n\
"
    return prompt

def get_data_from_output_file(output_file):
    pkl_data = pkl.load(open(output_file, "rb"))
    arguments = []
    for key, value in tqdm.tqdm(pkl_data.items()):
        control_value = value['control_value']
        topics = []
        for topic in control_value:
            if topic not in not_topics and topic != '':
                topics.append(topic)
        if len(topics) > 0:
            topics = ', '.join(topics)
            document = value['input']
            summary = value['predicted_summary']
            prompt = prompt_template(document, summary, topics)
            arguments.append((prompt, document, summary, topics))
    print(f"Number of arguments for file {output_file}: {len(arguments)}")
    return arguments

def get_openai_output_for_prompt(prompt, client):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{prompt}"}
            ],
        max_tokens=20,
        temperature=0,
        top_p=1,
    )
    json_output = convert_chat_completion_to_json(completion)
    extract_answer(json_output)
    return completion, json_output, extract_answer(json_output)

def get_openai_outputs_for_file(data_for_openai, file, client):
    relevant_data = data_for_openai[file]
    #relevant_data = relevant_data[:10]
    # print(len(relevant_data[0]))
    result = {}
    for index, arguments in tqdm.tqdm(enumerate(relevant_data)):
        prompt, document, summary, topics = arguments
        completion, json_output, responses = get_openai_output_for_prompt(prompt, client)
        result[index] = {
            "prompt": prompt,
            "document": document,
            "summary": summary,
            "topics": topics,
            "json_output": json_output,
            "responses": responses
        }
    name = file.replace("/","-")
    name = name[1:]
    save_dir = "/scratch/tathagato/openai_outputs"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{name}"
    with open(save_path, "wb") as f:
        pkl.dump(result, f)
    print(f"Saved results for file {file} to {save_path}")
    return result

def get_all_results(data_for_openai, files, client):
    all_results = {}
    for file in files:
        all_results[file] = get_openai_outputs_for_file(data_for_openai, file, client)
        #one minute sleep for rate limiting
        # print("Sleeping for 60 seconds")
        # time.sleep(60)

    save_all_results = f"/scratch/tathagato/openai_outputs/all_results_rest.pkl"
    with open(save_all_results, "wb") as f:
        pkl.dump(all_results, f)
    print(f"Saved all results to {save_all_results}")
    return all_results

        
if __name__ == "__main__":


    client = OpenAI()
    # completion = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": "Hello!"}
    #         ],
    #     max_tokens=20,
    #     temperature=0,
    #     top_p=1,
    #     )
    #save completion to a file
    # save_dir = "openai_outputs"
    # os.makedirs(save_dir, exist_ok=True)
    # print(completion)
    # json_output = convert_chat_completion_to_json(completion)
    # import code; code.interact(local=locals())
    # print(json_output)
    # print(extract_answer(json_output))
    # with open(f"{save_dir}/completion.json", "w") as f:
    #     json.dump(json_output, f, indent=4)

    files = [
        "/scratch/tathagato/naacl/zero_shot/mistral_topic.pkl",
        "/scratch/tathagato/naacl/zero_shot/mistral_topic_and_extractiveness.pkl",
        "/scratch/tathagato/naacl/zero_shot/mistral_length_and_topic.pkl",

        "/scratch/tathagato/naacl/single_attribute_sft/mistral_topic/results.pkl",
        "/scratch/tathagato/naacl/single_attribute_dpo/mistral_topic/results.pkl",

        "/scratch/tathagato/naacl/joint_mult_attribute_sft/mistral_extractiveness_and_topic/model_extractiveness_and_topic_results_extractiveness_and_topic.pkl",
        "/scratch/tathagato/naacl/joint_mult_attribute_sft/mistral_length_and_topic/model_length_and_topic_results_length_and_topic.pkl",
        "/scratch/tathagato/naacl/joint_mult_attribute_dpo/mistral_extractiveness_and_topic/model_extractiveness_and_topic_results_extractiveness_and_topic.pkl",
        "/scratch/tathagato/naacl/joint_mult_attribute_dpo/mistral_length_and_topic/model_length_and_topic_results_length_and_topic.pkl",

        "/scratch/tathagato/naacl/multi_attribute_single_adapter_continued_sft/mistral_extractiveness_then_topic/model_extractiveness_and_topic_results_extractiveness_and_topic.pkl",
        "/scratch/tathagato/naacl/multi_attribute_single_adapter_continued_sft/mistral_topic_then_extractiveness/model_topic_and_extractiveness_results_topic_and_extractiveness.pkl",
        "/scratch/tathagato/naacl/multi_attribute_single_adapter_continued_sft/mistral_topic_then_length/model_topic_and_length_results_topic_and_length.pkl",
        "/scratch/tathagato/naacl/multi_attribute_single_adapter_continued_sft/mistral_length_then_topic/model_length_and_topic_results_length_and_topic.pkl",


        "/scratch/tathagato/naacl/multi_attribute_single_adapter_continued_dpo/mistral_extractiveness_then_topic/model_extractiveness_and_topic_results_extractiveness_and_topic.pkl",
        "/scratch/tathagato/naacl/multi_attribute_single_adapter_continued_dpo/mistral_topic_then_extractiveness/model_topic_and_extractiveness_results_topic_and_extractiveness.pkl",
        "/scratch/tathagato/naacl/multi_attribute_single_adapter_continued_dpo/mistral_topic_then_length/model_topic_and_length_results_topic_and_length.pkl",
        "/scratch/tathagato/naacl/multi_attribute_single_adapter_continued_dpo/mistral_length_then_topic/model_length_and_topic_results_length_and_topic.pkl",

        
        "/scratch/tathagato/naacl/multi_attribute_multi_adapter_sft/mistral_extractiveness_then_topic/model_extractiveness_then_topic_results_extractiveness_and_topic.pkl",
        "/scratch/tathagato/naacl/multi_attribute_multi_adapter_sft/mistral_topic_then_extractiveness/model_topic_then_extractiveness_results_topic_and_extractiveness.pkl",
        "/scratch/tathagato/naacl/multi_attribute_multi_adapter_sft/mistral_topic_then_length/model_topic_then_length_results_topic_and_length.pkl",
        "/scratch/tathagato/naacl/multi_attribute_multi_adapter_sft/mistral_length_then_topic/model_length_then_topic_results_length_and_topic.pkl",

        "/scratch/tathagato/naacl/multi_attribute_multi_adapter_dpo/mistral_extractiveness_then_topic/model_extractiveness_and_topic_results_extractiveness_and_topic.pkl",
        "/scratch/tathagato/naacl/multi_attribute_multi_adapter_dpo/mistral_topic_then_extractiveness/model_topic_and_extractiveness_results_topic_and_extractiveness.pkl",
        "/scratch/tathagato/naacl/multi_attribute_multi_adapter_dpo/mistral_topic_then_length/model_topic_and_length_results_topic_and_length.pkl",
        "/scratch/tathagato/naacl/multi_attribute_multi_adapter_dpo/mistral_length_then_topic/model_length_and_topic_results_length_and_topic.pkl",

        "/scratch/tathagato/naacl/adapter_fusion_sft/mistral_extractiveness_and_topic_0.67_0.67_0.33_linear/model_extractiveness_and_topic_fused_results_extractiveness_and_topic.pkl",
        "/scratch/tathagato/naacl/adapter_fusion_sft/mistral_extractiveness_and_topic_0.5_0.5_0.5_linear/model_extractiveness_and_topic_fused_results_extractiveness_and_topic.pkl",
        "/scratch/tathagato/naacl/adapter_fusion_sft/mistral_extractiveness_and_topic_0.33_0.33_0.67_linear/model_extractiveness_and_topic_fused_results_extractiveness_and_topic.pkl",

        "/scratch/tathagato/naacl/adapter_fusion_sft/mistral_length_and_topic_0.67_0.67_0.33_linear/model_length_and_topic_fused_results_length_and_topic.pkl",
        "/scratch/tathagato/naacl/adapter_fusion_sft/mistral_length_and_topic_0.5_0.5_0.5_linear/model_length_and_topic_fused_results_length_and_topic.pkl",
        "/scratch/tathagato/naacl/adapter_fusion_sft/mistral_length_and_topic_0.33_0.33_0.67_linear/model_length_and_topic_fused_results_length_and_topic.pkl",
        
        "/scratch/tathagato/naacl/adapter_fusion_dpo/mistral_extractiveness_and_topic_0.67_0.67_0.33_linear/model_extractiveness_and_topic_fused_results_extractiveness_and_topic.pkl",
        "/scratch/tathagato/naacl/adapter_fusion_dpo/mistral_extractiveness_and_topic_0.5_0.5_0.5_linear/model_extractiveness_and_topic_fused_results_extractiveness_and_topic.pkl",
        "/scratch/tathagato/naacl/adapter_fusion_dpo/mistral_extractiveness_and_topic_0.33_0.33_0.67_linear/model_extractiveness_and_topic_fused_results_extractiveness_and_topic.pkl",

        "/scratch/tathagato/naacl/adapter_fusion_dpo/mistral_length_and_topic_0.67_0.67_0.33_linear/model_length_and_topic_fused_results_length_and_topic.pkl",
        "/scratch/tathagato/naacl/adapter_fusion_dpo/mistral_length_and_topic_0.5_0.5_0.5_linear/model_length_and_topic_fused_results_length_and_topic.pkl",
        "/scratch/tathagato/naacl/adapter_fusion_dpo/mistral_length_and_topic_0.33_0.33_0.67_linear/model_length_and_topic_fused_results_length_and_topic.pkl",










    ]
    files = [file for file in files if os.path.exists(file)]
    files = files[10:]
    # keep only unique files
    files = list(set(files))

    data_for_openai = {}
    all_examples_counter = 0
    for file in tqdm.tqdm(files):
        data_for_openai[file] = get_data_from_output_file(file)
        all_examples_counter += len(data_for_openai[file])
    print(f"Total number of examples: {all_examples_counter}")

    all_results = get_all_results(data_for_openai, files, client)
    distribution = {}
    for file in files:
        result = all_results[file]
        for key in result:
            #print(result[key]['responses'])
            num = result[key]['responses'][0]
            if num not in distribution:
                distribution[num] = 1
            else:
                distribution[num] += 1
    print(distribution)
    
    #import code; code.interact(local=locals())


    
    
