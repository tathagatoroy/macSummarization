
# coding: utf-8

import argparse
import liblinear.liblinearutil as ll
import utils
import sys
from features import Space
from generatefeatures import ModelNewText
import tempfile
import os
import nltk
from concurrent.futures import ProcessPoolExecutor, as_completed
nltk.download('punkt')



RT = "./"

BRNCLSTSPACEFILE = RT+"cotraining_models/brnclst1gram.space"
SHALLOWSCALEFILE = RT+"cotraining_models/shallow.scale"
SHALLOWMODELFILE = RT+"cotraining_models/shallow.model"
NEURALBRNSCALEFILE = RT+"cotraining_models/neuralbrn.scale"
NEURALBRNMODELFILE = RT+"cotraining_models/neuralbrn.model"






def calculate_avg_specificity(preds):
    """Calculate average specificity score for all sentences."""
    return sum(preds) / len(preds)


def process_single_string(input_string):
    """Process a single input string: tokenize, get features, predict specificity."""
    # Tokenize the input string into words
    sentences = nltk.sent_tokenize(input_string)
    
    # Create a temporary file to store the tokenized sentences
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".txt") as tmpfile:
        for sentence in sentences:
            tokenized_sentence = " ".join(nltk.word_tokenize(sentence))
            tmpfile.write(tokenized_sentence + "\n")
        tmpfile_path = tmpfile.name
    
    try:
        # Get features for the tokenized sentences
        y, xs, xw = getFeatures(tmpfile_path)
        
        # Predict specificity for the sentences
        preds_comb, _, _ = predict(y, xs, xw)
        return calculate_avg_specificity(preds_comb)  # Return the average specificity score for the string
    finally:
        # Delete the temporary file after processing
        if os.path.exists(tmpfile_path):
            os.remove(tmpfile_path)

def process_strings_parallel(input_strings, max_workers=4):
    """Function to process multiple input strings, tokenize them, and calculate specificity in parallel."""
    
    # Use ProcessPoolExecutor for parallel processing
    specificity_scores = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_string, input_string) for input_string in input_strings]
        
        # Gather results as they are completed
        for future in as_completed(futures):
            specificity_scores.append(future.result())
    
    # Write the specificity scores to an output file (dummy file here)
    #outputfile = "specificity_scores_parallel.txt"
    #writeSpecificity(specificity_scores, outputfile)
    
    # Calculate and return the average specificity score
    # avg_specificity = calculate_avg_specificity(specificity_scores)
    
    return specificity_scores







def initBrnSpace():
    s = Space(101)
    s.loadFromFile(BRNCLSTSPACEFILE)
    return s

def readScales(scalefile):
    scales = {}
    with open(scalefile) as f:
        for line in f:
            k,v = line.strip().split("\t")
            scales[int(k)] = float(v)
        f.close()
    return scales

brnclst = utils.readMetaOptimizeBrownCluster()
embeddings = utils.readMetaOptimizeEmbeddings()
brnspace = initBrnSpace()
scales_shallow = readScales(SHALLOWSCALEFILE)
scales_neuralbrn = readScales(NEURALBRNSCALEFILE)
model_shallow = ll.load_model(SHALLOWMODELFILE)
model_neuralbrn = ll.load_model(NEURALBRNMODELFILE)

def simpleScale(x, trainmaxes=None):
    maxes = trainmaxes if trainmaxes!=None else {}
    if trainmaxes == None:
        for itemd in x:
            for k,v in itemd.items():
                if k not in maxes or maxes[k] < abs(v): maxes[k] = abs(v)
    newx = []
    for itemd in x:
        newd = dict.fromkeys(itemd)
        for k,v in itemd.items():
            if k in maxes and maxes[k] != 0: newd[k] = (v+0.0)/maxes[k]
            else: newd[k] = 0.0
        newx.append(newd)
    return newx,maxes

def getFeatures(fin):
    aligner = ModelNewText(brnspace,brnclst,embeddings)
    aligner.loadFromFile(fin)
    aligner.fShallow()
    aligner.fNeuralVec()
    aligner.fBrownCluster()
    y,xs = aligner.transformShallow()
    _,xw = aligner.transformWordRep()
    return y,xs,xw

def score(p_label, p_val):
    ret = []
    for l,prob in zip(p_label,p_val):
        m = max(prob)
        if l == 1: ret.append(1-m)
        else: ret.append(m)
    return ret

def predict(y,xs,xw):
    xs,_ = simpleScale(xs,scales_shallow)
    xw,_ = simpleScale(xw,scales_neuralbrn)
    p_label, p_acc, p_val = ll.predict(y,xs,model_shallow,'-q -b 1')
    ls_s = score(p_label,p_val)
    p_label, p_acc, p_val = ll.predict(y,xw,model_neuralbrn,'-q -b 1')
    ls_w = score(p_label,p_val)
    return [(x+y)/2 for x,y in zip(ls_s,ls_w)],ls_s,ls_w

def writeSpecificity(preds, outf):
    with open(outf,'w') as f:
        for x in preds:
            f.write("%f\n" % x)
        f.close()
    print("Output to "+outf+" done.")

def run(identifier, sentlist):
    ## main function to run speciteller and return predictions
    ## sentlist should be a list of sentence strings, tokenized;
    ## identifier is a string serving as the header of this sentlst
    aligner = ModelNewText(brnspace,brnclst,embeddings)
    aligner.loadSentences(identifier, sentlist)
    aligner.fShallow()
    aligner.fNeuralVec()
    aligner.fBrownCluster()
    y,xs = aligner.transformShallow()
    _,xw = aligner.transformWordRep()
    preds_comb, preds_s, preds_w = predict(y,xs,xw)
    return preds_comb
    

if __name__ == "__main__":
    # argparser = argparse.ArgumentParser()
    # argparser.add_argument("--inputfile", help="input raw text file, one sentence per line, tokenized", required=True)
    # argparser.add_argument("--outputfile", help="output file to save the specificity scores", required=True)
    # argparser.add_argument("--write_all_preds", help="write predictions from individual models in addition to the overall one", action="store_true")
    # # argparser.add_argument("--tokenize", help="tokenize input sentences?", required=True)
    # sys.stderr.write("SPECITELLER: please make sure that your input sentences are WORD-TOKENIZED for better prediction.\n")
    # args = argparser.parse_args()
    # y,xs,xw = getFeatures(args.inputfile)
    # preds_comb, preds_s, preds_w = predict(y,xs,xw)
    # writeSpecificity(preds_comb,args.outputfile)
    # if args.write_all_preds:
    #     writeSpecificity(preds_s,args.outputfile+".s")
    #     writeSpecificity(preds_w,args.outputfile+".w")

    # Example usage
# if __name__ == "__main__":
    # input_texts = [
    #     "This is the first input string. Here is another one. This is the third input string.",
    #     "This is the second input string. It has a different meaning.",
    #     "And this is the third input string for testing."
    # ]
    input_texts = [
    "A cat is sleeping.",
    "Midnight, a black cat, is sleeping.",
    "Midnight, the black cat, is sleeping on a couch.",
    "Midnight, the black cat, is sleeping on a vintage leather couch.",
    "Midnight, the black cat, is sleeping on a vintage leather couch in the living room.",
    "Midnight, the black cat, is sleeping on a vintage leather couch in the cozy living room of the Johnson residence.",
    "Midnight, the black cat, is sleeping on a vintage leather couch in the cozy living room of the Johnson residence on a sunny afternoon.",
    "Midnight, the black cat, is sleeping on a vintage leather couch in the cozy living room of the Johnson residence on a sunny afternoon of June 15th, 2024.",
    "Midnight, the black cat, is sleeping peacefully on a vintage leather couch in the cozy living room of the Johnson residence on a sunny afternoon of June 15th, 2024, at 3 PM.",
    "Midnight, the black cat, is sleeping peacefully on a vintage leather couch in the cozy living room of the Johnson residence on a sunny afternoon of June 15th, 2024, at 3 PM, while Mr. Johnson enjoys a cup of coffee and reads the newspaper."
]


    specifity_scores = process_strings_parallel(input_texts, max_workers=4)
    print(specifity_scores)

