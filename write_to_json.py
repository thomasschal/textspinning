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


# Translates "input_text" in language "src_lang" to language "tgt_lang" and back to "src_lang" using given "model"
def spin(model, src_lang, tgt_lang, input_text):
    intermediate = model.translate(input_text, source=src_lang, target=tgt_lang)  # Intermediate sentence
    return model.translate(intermediate, source=tgt_lang, target=src_lang)  # Result sentence


models = [dlt.TranslationModel("mbart50"), dlt.TranslationModel("m2m100")]

text_list = []
max_entries = 1000
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
file_name = 'bin\\translations-' + str(uuid.uuid4()) + '.json'
data.to_json(path_or_buf=file_name)
