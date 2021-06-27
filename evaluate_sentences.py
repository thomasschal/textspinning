import pandas as pd
import json
from strsimpy.cosine import Cosine
from bert_score import score
from strsimpy.metric_lcs import MetricLCS
import uuid

# Returns value between 1 and 0; 1 = equal; 0 = no similarity. Uses strsimpy package to calc cosine similarity
def cosine(input_text, output_text):
    cosine = Cosine(2)
    p0 = cosine.get_profile(input_text)
    p1 = cosine.get_profile(output_text)
    return cosine.similarity_profiles(p0, p1)


eval_cn = {'input': [], 'output_cn': [], 'bertscore': [], 'cosinesim': [], 'metriclcs': []}
eval_de = {'input': [], 'output_de': [], 'bertscore': [], 'cosinesim': [], 'metriclcs': []}

data = pd.read_json('bin\\translations-5b7509e1-a056-4952-b2cc-7fea82f51390.json')
metric_lcs = MetricLCS()
# input, output_de, output_cn

for index, row in data.iterrows():
    print(row['input'], row['output_de'], row['output_cn'])
    eval_cn['input'].append(row['input'])
    eval_de['input'].append(row['input'])
    eval_cn['output_cn'].append(row['output_cn'])
    eval_de['output_de'].append(row['output_de'])
    P, R, F1 = score([row['input']], [row['output_cn']], lang="en", verbose=True)
    eval_cn['bertscore'].append(f"{F1.mean():.3f}")
    P, R, F1 = score([row['input']], [row['output_de']], lang="en", verbose=True)
    eval_de['bertscore'].append(f"{F1.mean():.3f}")
    eval_cn['cosinesim'].append(str(cosine(row['input'], row['output_cn'])))
    eval_de['cosinesim'].append(str(cosine(row['input'], row['output_de'])))
    eval_cn['metriclcs'].append(str(1 - metric_lcs.distance(row['input'], row['output_cn'])))
    eval_de['metriclcs'].append(str(1 - metric_lcs.distance(row['input'], row['output_de'])))

eval_cn_data = pd.DataFrame(eval_cn)
eval_de_data = pd.DataFrame(eval_de)

file_name_cn = 'bin\\eval_cn_data-' + str(uuid.uuid4()) + '.json'
file_name_de = 'bin\\eval_de_data-' + str(uuid.uuid4()) + '.json'
eval_cn_data.to_json(path_or_buf=file_name_cn)
eval_de_data.to_json(path_or_buf=file_name_de)
