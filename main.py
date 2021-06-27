import dl_translate as dlt
import string
import nltk
import numpy as np
import pandas as pd
import json
import uuid
#nltk.download('punkt')
from nltk.translate.bleu_score import sentence_bleu
import math
import re
from collections import Counter
from strsimpy.cosine import Cosine
from bert_score import score
from strsimpy.metric_lcs import MetricLCS

# Returns value between 1 and 0; 1 = equal; 0 = no similarity. Uses strsimpy package to calc cosine similarity
def cosine(input_text, output_text):
    cosine = Cosine(2)
    p0 = cosine.get_profile(input_text)
    p1 = cosine.get_profile(output_text)
    return cosine.similarity_profiles(p0, p1)


# Returns True if text is spun, returns False if text is equal
def evaluate_output(input, output):
    if input == output:
        return False
    else:
        return True


# Returns True if text is spun, returns False if text is equal
def evaluate_output_tokens(input, output):
    if input == output:
        return False
    else:
        return True


# Translates "input_text" in language "src_lang" to language "tgt_lang" and back to "src_lang" using given "model"
def spin(model, src_lang, tgt_lang, input_text):
    intermediate = model.translate(input_text, source=src_lang, target=tgt_lang)  # Intermediate sentence
    return model.translate(intermediate, source=tgt_lang, target=src_lang)  # Result sentence


# Prints input, output and evaluation line by line
def print_results(model, src_lang, tgt_lang, input_text):
    output = spin(model=model, src_lang=src_lang, tgt_lang=tgt_lang, input_text=input_text)
    print("-------------------------------------------------------------------------------------------------------")
    print("Input:  " + input_text)
    print("Output: " + output)
    print("Spun:   " + str(evaluate_output(input_text, output)))


# Prints input, output and evaluation line by line
def print_result_tokens(model, src_lang, tgt_lang, input_text):
    output = spin(model=model, src_lang=src_lang, tgt_lang=tgt_lang, input_text=input_text)
    punctRemover = str.maketrans('', '', string.punctuation)
    output = output.translate(punctRemover)               # remove punctuation to compare words only
    input_text = input_text.translate(punctRemover)
    output_tokens = nltk.word_tokenize(output)                  # split sentence into tokens
    input_tokens = nltk.word_tokenize(input_text)
    print("-------------------------------------------------------------------------------------------------------")
    print("Input:       " + ' '.join(input_tokens))
    print("Output:      " + ' '.join(output_tokens))
    print("Spun:        " + str(evaluate_output(input_tokens, output_tokens)))
    print("Cosine:      " + str(cosine(input_text, output)))
    print("MetricLCS:   " + str(metric_lcs.distance(input_text, output)))
    P, R, F1 = score([input_text], [output], lang="en", verbose=True)
    print(f"BERTSc:  {F1.mean():.3f}")


models = [dlt.TranslationModel("mbart50"), dlt.TranslationModel("m2m100")]
src_lang = "German"
tgt_lang = "English"

# print_results(model=models[0], src_lang=src_lang,tgt_lang=tgt_lang,input_text=input_text)

'''
text_list = []
max_entries = 50
cur_entry = 0
corpusfile = open('res\en\eng_news_2020_10K\eng_news_2020_10K-sentences.txt', 'r', encoding='utf-8')
for line in corpusfile.readlines():
    if cur_entry < max_entries:
        text_list.append(line.split("\t")[1].rstrip()) # remove leading number, tab and trailing line feed
        cur_entry += 1
        print(line)
        # print(line.split("\t")[1])
    else:
        break


text_dict = {'input': [], 'output_de': [], 'output_cn': []}

cur_entry = 0
for sent in text_list:
    text_dict['input'].append(sent)
    text_dict['output_de'].append(spin(models[0], 'English', 'German', sent))
    text_dict['output_cn'].append(spin(models[0], 'English', 'Chinese', sent))
    cur_entry += 1
    print(cur_entry)

data = pd.DataFrame(text_dict)
file_name = 'bin\\' + str(uuid.uuid4()) + '.json'
data.to_json(path_or_buf=file_name)
data = pd.read_json(file_name)
'''
'''
for sentence in text_list:
    print_results(models[0],src_lang,tgt_lang,sentence)


for sentence in text_list:
    print_result_tokens(models[0],src_lang,tgt_lang,sentence)
'''

data = pd.read_json('bin\\1ec24800-8669-48bc-8d30-a1288d777c8e.json')
print(data)