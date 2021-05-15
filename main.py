import dl_translate as dlt
import string
import nltk
#nltk.download('punkt')
from nltk.translate.bleu_score import sentence_bleu
import math
import re
from collections import Counter

WORD = re.compile(r"\w+")


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)


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


for sentence in text_list:
    print_results(models[0],src_lang,tgt_lang,sentence)

'''
for sentence in text_list:
    print_result_tokens(models[0],src_lang,tgt_lang,sentence)
    '''