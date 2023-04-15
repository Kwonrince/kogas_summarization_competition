import os
import pandas as pd
import json
from itertools import chain
from tqdm import tqdm
#https://github.com/nlee0212/KoBART-summarization

#%% 뉴스기사
train = {'news':[],'summary':[]}
test = {'news':[],'summary':[]}

train_file = json.load(open('./train_data/Training/신문기사_train_original.json', encoding='UTF8'))
train_file_doc = train_file['documents']

for data in train_file_doc:
        train['news'].append(' '.join(list(chain(*[[j['sentence'] for j in i] for i in data['text']]))))
        train['summary'].append(' '.join(data['abstractive']))
    
print(len(train['news']),len(train['summary']))

val_file = json.load(open('./train_data/Validation/신문기사_vaild_original.json', encoding='UTF8'))
val_file_doc = val_file['documents']

for data in val_file_doc:
        test['news'].append(' '.join(list(chain(*[[j['sentence'] for j in i] for i in data['text']]))))
        test['summary'].append(' '.join(data['abstractive']))
    
print(len(test['news']),len(test['summary']))

train_df = pd.DataFrame(train)
test_df = pd.DataFrame(test)

#train_df.to_csv('./KoBART-summarization/data/train.tsv', index=False, sep='\t', encoding='utf-8')
#test_df.to_csv('./KoBART-summarization/data/test.tsv', index=False, sep='\t', encoding='utf-8')

#%% 법률
train = {'news':[],'summary':[]}
test = {'news':[],'summary':[]}

train_file = json.load(open('./train_data/Training/법률_train_original.json', encoding='UTF8'))
train_file_doc = train_file['documents']

for data in train_file_doc:
        train['news'].append(' '.join(list(chain(*[[j['sentence'] for j in i] for i in data['text']]))))
        train['summary'].append(' '.join(data['abstractive']))
    
print(len(train['news']),len(train['summary']))

val_file = json.load(open('./train_data/Validation/법률_vaild_original.json', encoding='UTF8'))
val_file_doc = val_file['documents']

for data in val_file_doc:
        test['news'].append(' '.join(list(chain(*[[j['sentence'] for j in i] for i in data['text']]))))
        test['summary'].append(' '.join(data['abstractive']))
    
print(len(test['news']),len(test['summary']))

train_df = pd.DataFrame(train)
test_df = pd.DataFrame(test)

#train_df.to_csv('./KoBART-summarization/data/train_law.tsv', index=False, sep='\t', encoding='utf-8')
#test_df.to_csv('./KoBART-summarization/data/test_law.tsv', index=False, sep='\t', encoding='utf-8')
