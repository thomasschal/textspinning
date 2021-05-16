import dl_translate as dlt
import string
import nltk
import numpy
#nltk.download('punkt')
from nltk.translate.bleu_score import sentence_bleu
import math
import re
from collections import Counter
from strsimpy.cosine import Cosine


def cosine(input_text, output_text):
    cosine = Cosine(2)
    p0 = cosine.get_profile(input_text)
    p1 = cosine.get_profile(output_text)
    return cosine.similarity_profiles(p0, p1)


def text_to_vector(tokens):
    return Counter(tokens)


def cosine_distance(u, v):
    """
    Returns the cosine of the angle between vectors v and u. This is equal to
    u.v / |u||v|.
    """
    return numpy.dot(u, v) / (math.sqrt(numpy.dot(u, u)) * math.sqrt(numpy.dot(v, v)))


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
    print("Input:  " + ' '.join(input_tokens))
    print("Output: " + ' '.join(output_tokens))
    print("Spun:   " + str(evaluate_output(input_tokens, output_tokens)))
    print("Cosine: " + str(cosine(input_text, output)))


models = [dlt.TranslationModel("mbart50"), dlt.TranslationModel("m2m100")]
model = dlt.TranslationModel("mbart50")
src_lang = "German"
tgt_lang = "English"
input_text = "Der UN-Generalsekretär sprach auf der letzten Pressekonferenz von Angriffen auf die Persönlichkeitsrechte der Bürger Südamerikas."

#print_results(model=models[0], src_lang=src_lang,tgt_lang=tgt_lang,input_text=input_text)


text_list = []
corpusfile = open('res\deu_de-news-wrt_2019_1K\deu_de-news-wrt_2019_1K-sentences.txt', 'r', encoding='utf-8')
for line in corpusfile.readlines():
    text_list.append(line.split("\t")[1].rstrip()) # remove leading number, tab and trailing line feed
    #print(line.split("\t")[1])

'''
for sentence in text_list:
    print_results(models[0],src_lang,tgt_lang,sentence)
'''

for sentence in text_list:
    print_result_tokens(models[0],src_lang,tgt_lang,sentence)
