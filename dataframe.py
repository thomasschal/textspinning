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


models = [dlt.TranslationModel("mbart50"), dlt.TranslationModel("m2m100")]
src_lang = "German"
tgt_lang = "English"

text_list = []
rnd_list = []
text_dict = {'text': [], 'random': []}
corpusfile = open('res\de\deu_de-news-wrt_2019_1K\deu_de-news-wrt_2019_1K-sentences.txt', 'r', encoding='utf-8')
for line in corpusfile.readlines():
    text_list.append(line.split("\t")[1].rstrip()) # remove leading number, tab and trailing line feed
    rnd_list.append('Test')
    # print(line.split("\t")[1])

for text in text_list:
    text_dict['text'].append(text)

for random in rnd_list:
    text_dict['random'].append(random)

data = pd.DataFrame(text_dict)
data.shape
print(data)
file_name = 'res\dumps\\' + str(uuid.uuid4()) + '.json'
data.to_json(path_or_buf=file_name)
data = pd.read_json(file_name)
print(data)