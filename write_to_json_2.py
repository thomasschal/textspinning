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


def spin(model, lang_list, input_text, options):
    intermediate = input_text
    length = len(lang_list)
    for i in range(length):
        if i < length-1:
            print(i)
            print(lang_list[i])
            print(lang_list)
            intermediate = model.translate(intermediate,
                                           source=lang_list[i],
                                           target=lang_list[i+1],
                                           batch_size=32,
                                           generation_options=options)
    return intermediate


#models = [dlt.TranslationModel("mbart50"), dlt.TranslationModel("m2m100")]
models = [dlt.TranslationModel("mbart50")]
language_combinations_de = [
                            ['German', 'English', 'German'],
                            ['German', 'Portuguese', 'German'],
                            ['German', 'English', 'Portuguese', 'German']
                            ]
language_combinations_en = [
                            ['English', 'German', 'English'],
                            ['English', 'Portuguese', 'English'],
                            ['English', 'German', 'Portuguese', 'English']
                            ]
opt_list = [dict(num_beams=1),
            dict(num_beams=10),
            dict(num_beams=1, do_sample=True, top_k=0, temperature=0.7),
            dict(num_beams=1, do_sample=True, top_k=0, temperature=0.3)]
opt1 = dict(num_beams=1)

text_list_en = []
text_list_de = []
max_entries = 100
cur_entry = 0

corpusfile_en = open('res/en/SUTSCHE_EN.csv', 'r', encoding='utf-8')
corpusfile_de = open('res/de/SUTSCHE_DE.csv', 'r', encoding='utf-8')

for line in corpusfile_de.readlines():
    if cur_entry < max_entries:
        text_list_de.append(line)
        cur_entry += 1
        print(line)
    else:
        break

cur_entry = 0

for line in corpusfile_en.readlines():
    if cur_entry < max_entries:
        text_list_en.append(line)
        cur_entry += 1
        print(line)
    else:
        break

text_dict = {'input': [], 'output': []}

cur_entry = 0
lang_str = ''

for opt in opt_list:
    for index_model in range(len(models)):
        for lang_combo in language_combinations_en:
            for sent in text_list_en:
                text_dict['input'].append(sent)
                text_dict['output'].append(spin(models[index_model], lang_combo, sent, opt))
                cur_entry += 1
                print(cur_entry)
            if index_model == 0:
                model_name = 'mbart50'
            else:
                model_name = 'm2m'
            for lang in lang_combo:
                if lang_str == '':
                    lang_str = lang
                else:
                    lang_str = lang_str + lang
            data = pd.DataFrame(text_dict)
            file_name = 'bin\\' + model_name + '-' + lang_str + '-' + str(opt_list.index(opt)) + '-' + str(uuid.uuid4()) + '.json'
            data.to_json(path_or_buf=file_name)
            lang_str = ''
            model_name = ''
            data = None
            text_dict = {'input': [], 'output': []}
