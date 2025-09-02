import pickle as pkl 
import os 
import numpy as np
import matplotlib.pyplot as plt
import pandas 
from pyrouge import Rouge155
from multiprocessing import Pool
import multiprocessing
import tempfile
import shutil
import tqdm
import time
from rouge_score import rouge_scorer
import json
import sys
import nltk
nltk.download('punkt')
from nltk import word_tokenize, ngrams
import math
from collections import defaultdict
from nltk.corpus import stopwords
import glob
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
st_words = stopwords.words('english')
sys.path.insert(0,"/home2/tathagato/summarization/MACSUM/naacl/speciteller/python3_code")
#from speciteller import process_strings_parallel


def get_specificity_scores(texts):
    return process_strings_parallel(texts)

def get_specificity_results(candidates, references, articles, control_values):
    rouge_results = get_rouge_score(candidates, references)
    specificity_scores = get_specificity_scores(candidates)
    gold_specificity_scores = get_specificity_scores(references)
    rouge_1s = [result['rouge1']['f1'] for result in rouge_results['results']]
    rouge_2s = [result['rouge2']['f1'] for result in rouge_results['results']]
    rouge_3s = [result['rouge3']['f1'] for result in rouge_results['results']]
    rouge_Ls = [result['rougeL']['f1'] for result in rouge_results['results']]
    final_results = {}
    final_results['rouge_raw'] = rouge_results
    final_results['overall'] = {'specificity' : np.mean(specificity_scores), 'gold_specificity' : np.mean(gold_specificity_scores), 'rouge1' : np.mean(rouge_1s), 'rouge2' : np.mean(rouge_2s), 'rouge3' : np.mean(rouge_3s), 'rougeL' : np.mean(rouge_Ls), 'number' : len(candidates)}
    normal_indexes = [index for index, control_value in enumerate(control_values) if control_value == 'normal']
    normal_specificity_scores = [specificity_scores[index] for index in normal_indexes]
    normal_rouge1 = [rouge_1s[index] for index in normal_indexes]
    normal_rouge2 = [rouge_2s[index] for index in normal_indexes]
    normal_rouge3 = [rouge_3s[index] for index in normal_indexes]
    normal_rougeL = [rouge_Ls[index] for index in normal_indexes]
    gold_normal_specificity_scores = [gold_specificity_scores[index] for index in normal_indexes]
    final_results['normal'] = {'specificity' : np.mean(normal_specificity_scores), 'gold_specificity' : np.mean(gold_normal_specificity_scores), 'rouge1' : np.mean(normal_rouge1), 'rouge2' : np.mean(normal_rouge2), 'rouge3' : np.mean(normal_rouge3), 'rougeL' : np.mean(normal_rougeL), 'number' : len(normal_indexes)}

    high_indexes = [index for index, control_value in enumerate(control_values) if control_value == 'high']
    high_specificity_scores = [specificity_scores[index] for index in high_indexes]
    high_rouge1 = [rouge_1s[index] for index in high_indexes]
    high_rouge2 = [rouge_2s[index] for index in high_indexes]
    high_rouge3 = [rouge_3s[index] for index in high_indexes]
    high_rougeL = [rouge_Ls[index] for index in high_indexes]
    gold_high_specificity_scores = [gold_specificity_scores[index] for index in high_indexes]
    final_results['high'] = {'specificity' : np.mean(high_specificity_scores), 'gold_specificity' : np.mean(gold_high_specificity_scores), 'rouge1' : np.mean(high_rouge1), 'rouge2' : np.mean(high_rouge2), 'rouge3' : np.mean(high_rouge3), 'rougeL' : np.mean(high_rougeL), 'number' : len(high_indexes)}
    print("\n\nspecificity evaluation")
    for key in final_results.keys():
        if key == 'rouge_raw':
            continue
        print(f"--------------{key}----------------")
        res = final_results[key]
        for sub_key in res.keys():
            print(f"{sub_key} : {res[sub_key]}")
        print("-------------------------------------------------")
    print("----------------------------------------------------------------------------------------")
    return final_results

def get_rouge_score(candidates , references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rougeL'], use_stemmer=True)
    arguments = [(candidate, reference, scorer) for candidate,reference in zip(candidates, references)]
    with Pool(multiprocessing.cpu_count() - 2) as pool:
        results = pool.starmap(scores, arguments)
    # do agregation
    aggregrated_score = {}
    aggregrated_score['rouge1'] = {'precision' : 0, 'recall' : 0, 'f1' : 0}
    aggregrated_score['rouge2'] = {'precision' : 0, 'recall' : 0, 'f1' : 0}
    aggregrated_score['rouge3'] = {'precision' : 0, 'recall' : 0, 'f1' : 0}
    aggregrated_score['rougeL'] = {'precision' : 0, 'recall' : 0, 'f1' : 0}

    for result in results:
        for key in result.keys():
            aggregrated_score[key]['precision'] += result[key]['precision']
            aggregrated_score[key]['recall'] += result[key]['recall']
            aggregrated_score[key]['f1'] += result[key]['f1']
    for key in aggregrated_score.keys():
        aggregrated_score[key]['precision'] /= len(results)
        aggregrated_score[key]['recall'] /= len(results)
        aggregrated_score[key]['f1'] /= len(results)
    output = {'aggregrated_score' : aggregrated_score, 'results' : results}
    return output
    

def scores(summary, reference, scorer):
    score = scorer.score(summary, reference)
    res = {}
    res['rouge1'] = {'precision' : score['rouge1'].precision, 'recall' : score['rouge1'].recall, 'f1' : score['rouge1'].fmeasure}
    res['rouge2'] = {'precision' : score['rouge2'].precision, 'recall' : score['rouge2'].recall, 'f1' : score['rouge2'].fmeasure}
    res['rouge3'] = {'precision' : score['rouge3'].precision, 'recall' : score['rouge3'].recall, 'f1' : score['rouge3'].fmeasure}
    res['rougeL'] = {'precision' : score['rougeL'].precision, 'recall' : score['rougeL'].recall, 'f1' : score['rougeL'].fmeasure}
    return res
def get_summary_length(summary):
    return len(word_tokenize(summary.lower()))

def get_compression_ratio(article, summary):
    return float(len(word_tokenize(summary.lower()))) / float(len(word_tokenize(article.lower())))

def get_length_results(candidates, references, articles, control_values):
    rouge_results = get_rouge_score(candidates, references)
    summary_lengths = [get_summary_length(summary) for summary in candidates]
    compression_ratios = [get_compression_ratio(article, summary) for article, summary in zip(articles, candidates)]
    groundtruth_lengths = [get_summary_length(reference) for reference in references]
    groundtruth_compression_ratios = [get_compression_ratio(article, reference) for article, reference in zip(articles, references)]
    rouge_1s = [result['rouge1']['f1'] for result in rouge_results['results']]
    rouge_2s = [result['rouge2']['f1'] for result in rouge_results['results']]
    rouge_3s = [result['rouge3']['f1'] for result in rouge_results['results']]
    rouge_Ls = [result['rougeL']['f1'] for result in rouge_results['results']]

    final_results = {}
    final_results['rouge_raw'] = rouge_results
    final_results['overall'] = {'summary_length' : np.mean(summary_lengths), 'gold_summary_length' : np.mean(groundtruth_lengths), 'compression_ratio' : np.mean(compression_ratios),'gold_compression_ratio' : np.mean(groundtruth_compression_ratios), 'rouge1' : np.mean(rouge_1s), 'rouge2' : np.mean(rouge_2s), 'rouge3' : np.mean(rouge_3s), 'rougeL' : np.mean(rouge_Ls), 'number' : len(candidates)}
    short_indexes = [index for index, control_value in enumerate(control_values) if control_value == 'short']
    short_candidates = [candidates[index] for index in short_indexes]
    short_references = [references[index] for index in short_indexes]
    short_length = [summary_lengths[index] for index in short_indexes]
    short_compression = [compression_ratios[index] for index in short_indexes]
    short_rouge1 = [rouge_1s[index] for index in short_indexes]
    short_rouge2 = [rouge_2s[index] for index in short_indexes]
    short_rouge3 = [rouge_3s[index] for index in short_indexes]
    short_rougeL = [rouge_Ls[index] for index in short_indexes]
    short_gold_length = [groundtruth_lengths[index] for index in short_indexes]
    short_gold_compression = [groundtruth_compression_ratios[index] for index in short_indexes]
    final_results['short'] = {'summary_length' : np.mean(short_length), 'gold_summary_length' : np.mean(short_gold_length), 'compression_ratio' : np.mean(short_compression), 'gold_compression_ratio' : np.mean(short_gold_compression), 'rouge1' : np.mean(short_rouge1), 'rouge2' : np.mean(short_rouge2), 'rouge3' : np.mean(short_rouge3), 'rougeL' : np.mean(short_rougeL), 'number' : len(short_candidates)}
    
    normal_indexes = [index for index, control_value in enumerate(control_values) if control_value == 'normal']
    normal_candidates = [candidates[index] for index in normal_indexes]
    normal_references = [references[index] for index in normal_indexes]
    normal_length = [summary_lengths[index] for index in normal_indexes]
    normal_compression = [compression_ratios[index] for index in normal_indexes]
    normal_rouge1 = [rouge_1s[index] for index in normal_indexes]
    normal_rouge2 = [rouge_2s[index] for index in normal_indexes]
    normal_rouge3 = [rouge_3s[index] for index in normal_indexes]
    normal_rougeL = [rouge_Ls[index] for index in normal_indexes]
    normal_gold_length = [groundtruth_lengths[index] for index in normal_indexes]
    normal_gold_compression = [groundtruth_compression_ratios[index] for index in normal_indexes]
    final_results['normal'] = {'summary_length' : np.mean(normal_length), 'gold_summary_length' : np.mean(normal_gold_length), 'compression_ratio' : np.mean(normal_compression), 'gold_compression_ratio' : np.mean(normal_gold_compression), 'rouge1' : np.mean(normal_rouge1), 'rouge2' : np.mean(normal_rouge2), 'rouge3' : np.mean(normal_rouge3), 'rougeL' : np.mean(normal_rougeL), 'number' : len(normal_candidates)}
    #get indexes of control values where it is long
    long_indexes = [index for index, control_value in enumerate(control_values) if control_value == 'long']
    long_candidates = [candidates[index] for index in long_indexes]
    long_references = [references[index] for index in long_indexes]
    long_length = [summary_lengths[index] for index in long_indexes]
    long_compression = [compression_ratios[index] for index in long_indexes]
    long_rouge1 = [rouge_1s[index] for index in long_indexes]
    long_rouge2 = [rouge_2s[index] for index in long_indexes]
    long_rouge3 = [rouge_3s[index] for index in long_indexes]
    long_rougeL = [rouge_Ls[index] for index in long_indexes]
    long_gold_length = [groundtruth_lengths[index] for index in long_indexes]
    long_gold_compression = [groundtruth_compression_ratios[index] for index in long_indexes]
    final_results['long'] = {'summary_length' : np.mean(long_length), 'gold_summary_length' : np.mean(long_gold_length), 'compression_ratio' : np.mean(long_compression), 'gold_compression_ratio' : np.mean(long_gold_compression), 'rouge1' : np.mean(long_rouge1), 'rouge2' : np.mean(long_rouge2), 'rouge3' : np.mean(long_rouge3), 'rougeL' : np.mean(long_rougeL), 'number' : len(long_candidates)}
    print("\n\nlength evaluation")

    for key in final_results.keys():
        if key == 'rouge_raw':
            continue
        print(f"--------------{key}----------------")
        res = final_results[key]
        for sub_key in res.keys():
            print(f"{sub_key} : {res[sub_key]}")
        print("-------------------------------------------------")
    print("----------------------------------------------------------------------------------------")
    return final_results


def get_fragment_density(article, summary):
    """
    Calculates the fragment density of a summary on an article.

    Density is defined as the average squared length of extracted fragments.

    Args:
        article (str): The article text.
        summary (str): The summary text.

    Returns:
        float: The fragment density of the summary on the article.
    """

    frags, article_tokens, summary_tokens = get_extractive_fragments(article, summary)
    density = float(sum([len(f)**2 for f in frags])) / float(len(summary_tokens))
    return density

def get_overlap(inp, out, ngram = 2):
    grams_inp = set(ngrams(word_tokenize(inp.lower()), ngram))
    grams_out = set(ngrams(word_tokenize(out.lower()), ngram))

    total = len(grams_out)
    common = len(grams_inp.intersection(grams_out))
    if total == 0:
        return 0
    else:
        return float(common) / float(total)
def get_extractive_fragments(article, summary):
    """
    Extracts fragments from an article that match sequences of words in a summary.

    Args:
        article (str): The article text.
        summary (str): The summary text.

    Returns:
        list: A list of lists, where each sublist represents a sequence of word indexes
            in the article that match a sequence in the summary.
        list: The tokenized article.
        list: The tokenized summary.
    """

    article_tokens = word_tokenize(article.lower())
    summary_tokens = word_tokenize(summary.lower())

    F = []  # List to store the extracted fragments
    i, j = 0, 0  # Indexes for iterating over article and summary tokens, respectively

    while i < len(summary_tokens):
        f = []  # List to store the current fragment
        while j < len(article_tokens):
            if summary_tokens[i] == article_tokens[j]:
                i_, j_ = i, j  # Store starting indexes of potential fragment
                #print(len(summary_tokens), len(article_tokens), i, j, i_, j_, summary_tokens[i_], article_tokens[j_])
                while (i_ < len(summary_tokens) and j_ < len(article_tokens)) and summary_tokens[i_] == article_tokens[j_]:
                    i_, j_ = i_ + 1, j_ + 1  # Update indexes while words match
                if len(f) < (i_ - i):  # Update fragment if a longer match is found
                    f = list(range(i, i_))
                j = j_  # Set j to the next position after the matched sequence
            else:
                j += 1  # Move to the next article token if no match found
        i += max(len(f), 1)  # Update i by the length of the extracted fragment or 1
        j = 1  # Reset j for the next iteration

        F.append(f)  # Append the extracted fragment to the list

    return F, article_tokens, summary_tokens




    



def get_extractive_coverage(article, summary):
    """
    Calculates the extractive coverage of a summary on an article.

    Coverage is defined as the ratio of words in the summary covered by fragments
    extracted from the article.

    Args:
        article (str): The article text.
        summary (str): The summary text.

    Returns:
        float: The extractive coverage of the summary on the article.
    """

    frags, article_tokens, summary_tokens = get_extractive_fragments(article, summary)
    coverage = float(sum([len(f) for f in frags])) / float(len(summary_tokens))
    return coverage

def get_topic_results(candidates, references, articles, control_values):
    rouge_results = get_rouge_score(candidates, references)
    rouge_1s = [result['rouge1']['f1'] for result in rouge_results['results']]
    rouge_2s = [result['rouge2']['f1'] for result in rouge_results['results']]
    rouge_3s = [result['rouge3']['f1'] for result in rouge_results['results']]
    rouge_Ls = [result['rougeL']['f1'] for result in rouge_results['results']]
    final_results = {}
    final_results['rouge_raw'] = rouge_results
    final_results['overall'] = {'rouge1' : np.mean(rouge_1s), 'rouge2' : np.mean(rouge_2s), 'rouge3' : np.mean(rouge_3s), 'rougeL' : np.mean(rouge_Ls), 'number' : len(candidates)}
    for key in final_results.keys():
        if key == 'rouge_raw':
            continue
        print(f"--------------{key}----------------")
        res = final_results[key]
        for sub_key in res.keys():
            print(f"{sub_key} : {res[sub_key]}")
        print("-------------------------------------------------")
    

def get_extractiveness_results(candidates, references, articles, control_values):
    rouge_results = get_rouge_score(candidates, references)
    article_referenced_rouge_results = get_rouge_score(candidates, articles)
    gold_article_referenced_rouge_results = get_rouge_score(articles, candidates)
    fragment_densities = [get_fragment_density(article, summary) for article, summary in zip(articles, candidates)]
    gold_fragment_densities = [get_fragment_density(article, reference) for article, reference in zip(articles, references)]
    extractive_coverages = [get_extractive_coverage(article, summary) for article, summary in zip(articles, candidates)]
    gold_extractive_coverages = [get_extractive_coverage(article, reference) for article, reference in zip(articles, references)]
    overlaps = [get_overlap(article, summary) for article, summary in zip(articles, candidates)]
    gold_overlaps = [get_overlap(article, reference) for article, reference in zip(articles, references)]
    
    f_value = [(rouge_2_precision + rouge_3_precision) / 2 for rouge_2_precision, rouge_3_precision in zip([result['rouge2']['precision'] for result in article_referenced_rouge_results['results']], [result['rouge3']['precision'] for result in article_referenced_rouge_results['results']])]
    gold_f_value = [(rouge_2_precision + rouge_3_precision) / 2 for rouge_2_precision, rouge_3_precision in zip([result['rouge2']['precision'] for result in gold_article_referenced_rouge_results['results']], [result['rouge3']['precision'] for result in gold_article_referenced_rouge_results['results']])]

    rouge_1s = [result['rouge1']['f1'] for result in rouge_results['results']]
    rouge_2s = [result['rouge2']['f1'] for result in rouge_results['results']]
    rouge_3s = [result['rouge3']['f1'] for result in rouge_results['results']]
    rouge_Ls = [result['rougeL']['f1'] for result in rouge_results['results']]

    final_results = {}
    final_results['rouge_raw'] = rouge_results
    final_results['overall'] = {'fragment_density' : np.mean(fragment_densities), 'gold_fragment_density' : np.mean(gold_fragment_densities), 'coverage' : np.mean(extractive_coverages), 'gold_coverage' : np.mean(gold_extractive_coverages), 'overlap' : np.mean(overlaps), 'gold_overlap' : np.mean(gold_overlaps), 'f_value' : np.mean(f_value), 'gold_f_value' : np.mean(gold_f_value), 'rouge1' : np.mean(rouge_1s), 'rouge2' : np.mean(rouge_2s), 'rouge3' : np.mean(rouge_3s), 'rougeL' : np.mean(rouge_Ls), 'number' : len(candidates)}
    normal_indexes = [index for index, control_value in enumerate(control_values) if control_value == 'normal']
    normal_fragment_densities = [fragment_densities[index] for index in normal_indexes]
    normal_extractive_coverages = [extractive_coverages[index] for index in normal_indexes]
    normal_overlaps = [overlaps[index] for index in normal_indexes]
    normal_f_value = [f_value[index] for index in normal_indexes]
    normal_rouge1 = [rouge_1s[index] for index in normal_indexes]
    normal_rouge2 = [rouge_2s[index] for index in normal_indexes]
    normal_rouge3 = [rouge_3s[index] for index in normal_indexes]
    normal_rougeL = [rouge_Ls[index] for index in normal_indexes]
    gold_normal_fragment_densities = [gold_fragment_densities[index] for index in normal_indexes]
    gold_normal_extractive_coverages = [gold_extractive_coverages[index] for index in normal_indexes]
    gold_normal_overlaps = [gold_overlaps[index] for index in normal_indexes]
    gold_normal_f_value = [gold_f_value[index] for index in normal_indexes]

    final_results['normal'] = {'fragment_density' : np.mean(normal_fragment_densities), 'gold_fragment_density' : np.mean(gold_normal_fragment_densities), 'coverage' : np.mean(normal_extractive_coverages), 'gold_coverage' : np.mean(gold_normal_extractive_coverages), 'overlap' : np.mean(normal_overlaps), 'gold_overlap' : np.mean(gold_normal_overlaps), 'f_value' : np.mean(normal_f_value), 'gold_f_value' : np.mean(gold_normal_f_value), 'rouge1' : np.mean(normal_rouge1), 'rouge2' : np.mean(normal_rouge2), 'rouge3' : np.mean(normal_rouge3), 'rougeL' : np.mean(normal_rougeL), 'number' : len(normal_indexes)}
    high_indexes = [index for index, control_value in enumerate(control_values) if control_value == 'high']
    high_fragment_densities = [fragment_densities[index] for index in high_indexes]
    high_extractive_coverages = [extractive_coverages[index] for index in high_indexes]
    high_overlaps = [overlaps[index] for index in high_indexes]
    high_f_value = [f_value[index] for index in high_indexes]
    high_rouge1 = [rouge_1s[index] for index in high_indexes]
    high_rouge2 = [rouge_2s[index] for index in high_indexes]
    high_rouge3 = [rouge_3s[index] for index in high_indexes]
    high_rougeL = [rouge_Ls[index] for index in high_indexes]
    gold_high_fragment_densities = [gold_fragment_densities[index] for index in high_indexes]
    gold_high_extractive_coverages = [gold_extractive_coverages[index] for index in high_indexes]
    gold_high_overlaps = [gold_overlaps[index] for index in high_indexes]
    gold_high_f_value = [gold_f_value[index] for index in high_indexes]
    final_results['high'] = {'fragment_density' : np.mean(high_fragment_densities), 'gold_fragment_density' : np.mean(gold_high_fragment_densities), 'coverage' : np.mean(high_extractive_coverages), 'gold_coverage' : np.mean(gold_high_extractive_coverages), 'overlap' : np.mean(high_overlaps), 'gold_overlap' : np.mean(gold_high_overlaps), 'f_value' : np.mean(high_f_value), 'gold_f_value' : np.mean(gold_high_f_value), 'rouge1' : np.mean(high_rouge1), 'rouge2' : np.mean(high_rouge2), 'rouge3' : np.mean(high_rouge3), 'rougeL' : np.mean(high_rougeL), 'number' : len(high_indexes)}

    fully_indexes = [index for index, control_value in enumerate(control_values) if control_value == 'fully']
    fully_fragment_densities = [fragment_densities[index] for index in fully_indexes]
    fully_extractive_coverages = [extractive_coverages[index] for index in fully_indexes]
    fully_overlaps = [overlaps[index] for index in fully_indexes]
    fully_f_value = [f_value[index] for index in fully_indexes]
    fully_rouge1 = [rouge_1s[index] for index in fully_indexes]
    fully_rouge2 = [rouge_2s[index] for index in fully_indexes]
    fully_rouge3 = [rouge_3s[index] for index in fully_indexes]
    fully_rougeL = [rouge_Ls[index] for index in fully_indexes]
    gold_fully_fragment_densities = [gold_fragment_densities[index] for index in fully_indexes]
    gold_fully_extractive_coverages = [gold_extractive_coverages[index] for index in fully_indexes]
    gold_fully_overlaps = [gold_overlaps[index] for index in fully_indexes]
    gold_fully_f_value = [gold_f_value[index] for index in fully_indexes]
    final_results['fully'] = {'fragment_density' : np.mean(fully_fragment_densities), 'gold_fragment_density' : np.mean(gold_fully_fragment_densities), 'coverage' : np.mean(fully_extractive_coverages), 'gold_coverage' : np.mean(gold_fully_extractive_coverages), 'overlap' : np.mean(fully_overlaps), 'gold_overlap' : np.mean(gold_fully_overlaps), 'f_value' : np.mean(fully_f_value), 'gold_f_value' : np.mean(gold_fully_f_value), 'rouge1' : np.mean(fully_rouge1), 'rouge2' : np.mean(fully_rouge2), 'rouge3' : np.mean(fully_rouge3), 'rougeL' : np.mean(fully_rougeL), 'number' : len(fully_indexes)}

    print("extractiveness evaluation")
    for key in final_results.keys():
        if key == 'rouge_raw':
            continue
        print(f"--------------{key}----------------")
        res = final_results[key]
        for sub_key in res.keys():
            print(f"{sub_key} : {res[sub_key]}")
        print("-------------------------------------------------")
    print("----------------------------------------------------------------------------------------")
    return final_results
    
    


def get_model_and_attributes(filename):
    all_attributes = ['length', 'extractiveness' , 'topic', 'specificity']
    basename = os.path.basename(filename)
    name, ext = os.path.splitext(basename)
    model = name.split("_")[0]
    splits = name.split("_")
    #if fused is in splits then only take list after fused
    if 'fused' in splits:
        fused_index = splits.index('fused')
        splits = splits[fused_index:]
    attributes = [attribute for attribute in splits if attribute in all_attributes]
    return model, attributes

def load_pickle(file):
    with open(file, "rb") as f:
        data = pkl.load(f)
    return data


def get_results(file, candidates, references, articles, attribute, control_values):
    print(f"{file} is being evaluated on {attribute}")
    if attribute == 'length':
        return get_length_results(candidates, references, articles, control_values)
    elif attribute == 'extractiveness':
        return get_extractiveness_results(candidates, references, articles, control_values)
    elif attribute == 'topic':
        return get_topic_results(candidates, references, articles, control_values)
    else:
        print("not implemented for this attribute", attribute)
        None 

def evaluate(file, supported_evaluation = ['length','extractiveness', 'topic'], model = None, attributes = None):
    if model is None and attributes is None:
        model, attributes = get_model_and_attributes(file)
    print(f"evaluating {model} with attributes {attributes}")
    data = load_pickle(file)



    if len(attributes) == 1:
        candidates = [item['predicted_summary'] for item in data.values()]
        references = [item['output'] for item in data.values()]
        articles = [item['input'] for item in data.values()]
        control_values = [item['control_value'][0] for item in data.values()]
        assert len(candidates) == len(references) == len(articles) == len(control_values),  f"Error in length mismatch candidates {len(candidates)} references {len(references)} articles {len(articles)} control_values {len(control_values)}"
        print("length of data is ", len(candidates))
        if attributes[0] in supported_evaluation:
            outputs = get_results(file, candidates, references, articles, attributes[0], control_values)
            return {attributes[0] : outputs}
        else:
            print(f"{attributes[0]} is not supported for evaluation")
            print(f"skipping {attributes[0]} for {os.path.basename(file)}")
            return {attributes[0] : None}
    else:
        #filter given none of the control values are ''
        #print(data.values())
        candidates = [item['predicted_summary'] for item in data.values() if item['control_value'][0] != '' and item['control_value'][1] != '']
        references = [item['output'] for item in data.values() if item['control_value'][0] != '' and item['control_value'][1] != '']
        articles = [item['input'] for item in data.values() if item['control_value'][0] != '' and item['control_value'][1] != '']
        first_control_values = []
        second_control_values = []
        for item in data.values():
            if item['control_value'][0] != '' and item['control_value'][1] != '':
                for index, attribute in enumerate(item['control_attribute']):
                    if attribute == attributes[0]:
                        first_control_values.append(item['control_value'][index])
                    if attribute == attributes[1]:
                        second_control_values.append(item['control_value'][index])
        assert len(candidates) == len(references) == len(articles) == len(first_control_values) == len(second_control_values), f"Error in length mismatch candidates {len(candidates)} references {len(references)} articles {len(articles)} control_values {len(first_control_values)}, {len(second_control_values)}"
        print("length of data is ", len(candidates))
        if attributes[0] not in supported_evaluation:
            print(f"{attributes[0]} is not supported for evaluation")
            print(f"skipping {attributes[0]} for {os.path.basename(file)}")
            first_outputs = None
        else:
            first_outputs = get_results(file, candidates, references, articles, attributes[0], first_control_values)
        if attributes[1] not in supported_evaluation:
            print(f"{attributes[1]} is not supported for evaluation")
            print(f"skipping {attributes[1]} for {os.path.basename(file)}")
            second_outputs = None
        else:
            second_outputs = get_results(file, candidates, references, articles, attributes[1], second_control_values)
        return {attributes[0] : first_outputs, attributes[1] : second_outputs}




def zero_shot_evaluation(zero_shot_directory):
    files = [os.path.join(zero_shot_directory, file) for file in os.listdir(zero_shot_directory) if file.endswith(".pkl")]
    results = {}
    for file in tqdm.tqdm(files):
        print(f"evaluating {file}")
        model_results = evaluate(file)
        results[file] = model_results
    #save results
    output_dir = "/scratch/tathagato/naacl/compiled_outputs"
    os.makedirs(output_dir, exist_ok = True)
    output_file = os.path.join(output_dir, "zero_shot_results.pkl")
    with open(output_file, "wb") as f:
        pkl.dump(results, f)

    return results


def single_adapter_joint_training_evaluation(single_adapter_joint_training_directory):
    # walk recursively through the directory and get all the files which ends with pkl
    experiments = os.listdir(single_adapter_joint_training_directory)
    results = {}
    for exp in tqdm.tqdm(experiments):
        pkl_files = [f for f in os.listdir(os.path.join(single_adapter_joint_training_directory, exp)) if f.endswith(".pkl")]
        model = exp.split("_")[0]
        attributes = [f for f in exp.split("_")[1:] if f in ['length', 'extractiveness', 'topic', 'specificity']]
        relevant_pickle_file = None
        for pkl_file in pkl_files:
            #number of and should be 2 in the file name
            if relevant_pickle_file is None and pkl_file.split("_").count('and') == 2:
                relevant_pickle_file = pkl_file
                break
        results[exp] = evaluate(os.path.join(single_adapter_joint_training_directory, exp, relevant_pickle_file), model = model, attributes = attributes)
    #save results
    output_dir = "/scratch/tathagato/naacl/compiled_outputs"
    os.makedirs(output_dir, exist_ok = True)
    output_file = os.path.join(output_dir, "single_adapter_joint_training_results.pkl")
    with open(output_file, "wb") as f:
        pkl.dump(results, f)
    return results

def single_sft_evaluation(single_sft_directory = "/scratch/tathagato/naacl/single_attribute_sft"):
    experiment_paths = [(folder, os.path.join(single_sft_directory, folder, "results.pkl")) for folder in os.listdir(single_sft_directory) if os.path.isdir(os.path.join(single_sft_directory, folder))]
    results = {}
    for experiment, result_path in tqdm.tqdm(experiment_paths):
        model_name = experiment.split("_")[0]
        attributes = experiment.split("_")[1:]
        model_results = evaluate(result_path, model = model_name, attributes = attributes)
        results[experiment] = model_results
    #save results
    output_dir = "/scratch/tathagato/naacl/compiled_outputs"
    os.makedirs(output_dir, exist_ok = True)
    output_file = os.path.join(output_dir, "single_sft_results.pkl")
    with open(output_file, "wb") as f:
        pkl.dump(results, f)
    return results

def multi_attribute_single_adapter_continuous_training__sft_evaluation(multi_attribute_single_adapter_continuous_training_directory = "/scratch/tathagato/naacl/multi_attribute_single_adapter_continued_sft/"):
    experiments = os.listdir(multi_attribute_single_adapter_continuous_training_directory)
    results = {}
    for exp in tqdm.tqdm(experiments):
        pkl_files = [f for f in os.listdir(os.path.join(multi_attribute_single_adapter_continuous_training_directory, exp)) if f.endswith(".pkl")]
        model = exp.split("_")[0]
        attributes = [f for f in exp.split("_")[1:] if f in ['length', 'extractiveness', 'topic', 'specificity']]
        relevant_pickle_file = None
        for pkl_file in pkl_files:
            #number of and should be 2 in the file name
            if relevant_pickle_file is None and pkl_file.split("_").count('and') == 2:
                relevant_pickle_file = pkl_file
                break
        results[exp] = evaluate(os.path.join(multi_attribute_single_adapter_continuous_training_directory, exp, relevant_pickle_file), model = model, attributes = attributes)
def single_dpo_evaluation(single_sft_directory = "/scratch/tathagato/naacl/single_attribute_dpo"):
    experiment_paths = [(folder, os.path.join(single_sft_directory, folder, "results.pkl")) for folder in os.listdir(single_sft_directory) if os.path.isdir(os.path.join(single_sft_directory, folder))]
    results = {}
    for experiment, result_path in tqdm.tqdm(experiment_paths):
        model_name = experiment.split("_")[0]
        attributes = experiment.split("_")[1:]
        model_results = evaluate(result_path, model = model_name, attributes = attributes)
        results[experiment] = model_results
    #save results
    output_dir = "/scratch/tathagato/naacl/compiled_outputs"
    os.makedirs(output_dir, exist_ok = True)
    output_file = os.path.join(output_dir, "single_dpo_results.pkl")
    with open(output_file, "wb") as f:
        pkl.dump(results, f)
    return results



def adapter_fusion_evaluation(adapter_fusion):
    folders = [os.path.join(adapter_fusion, folder) for folder in os.listdir(adapter_fusion) if os.path.isdir(os.path.join(adapter_fusion, folder))]
    files = []
    for folder in folders:
        basenames = [filename for filename, ext in [os.path.splitext(file) for file in os.listdir(folder)]]
        # for now I only care only about the joint prompts 
        splits = [basename.split("_") for basename in basenames]
        model_names = [split[0] for split in splits]
        #if number of ands is 2 then it is a joint prompt
        for model_name, split, basename in zip(model_names, splits, basenames):
            if split.count('and') == 2:
                files.append([model_name, os.path.join(folder, basename + ".pkl")])
    results = {}
    for model_name, file in files:
        print("evaluating ", model_name, file)
        model_results = evaluate(file)
        print("\n\n")
        results[file] = model_results
    #save results
    output_dir = "/scratch/tathagato/naacl/compiled_outputs"
    os.makedirs(output_dir, exist_ok = True)
    output_file = os.path.join(output_dir, "adapter_fusion_results.pkl")
    with open(output_file, "wb") as f:
        pkl.dump(results, f)

        
