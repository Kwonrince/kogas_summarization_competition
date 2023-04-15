import pandas as pd
import numpy as np
import random
import json
import os
import re
from tqdm import tqdm
import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

#%%
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(42)
#%%
with open('./data/new_test.jsonl', 'r', encoding='utf-8') as f:
    jsonl = list(f)

test_data = []
for json_str in jsonl:
    test_data.append(json.loads(json_str))

submission = pd.read_csv('data/new_sample_submission.csv')

#%%
for data in test_data:
    did = int(data['id'])
    summary = data['article_original'][0] + data['article_original'][-1]
    idx = submission[submission['id']==did].index
    submission.loc[idx, 'summary'] = summary

#submission.to_csv('./data/baseline.csv', index=False)
########################################여기까지 baseline code############################################
#%% test df 생성
id_ = []
media_ = []
article_original_ = []
for doc in test_data:
    id_.append(doc['id'])
    media_.append(doc['media'])
    article_original_.append(' '.join(doc['article_original']))

test_df = pd.DataFrame({'id':id_, 'media':media_, 'article_original':article_original_})

#%% kobart - gogamza - epoch15 - repeat
tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')
model = BartForConditionalGeneration.from_pretrained('./KoBART-summarization/kobart_summary_epoch15').to('cuda')

test_df.summary = 0

for idx, article in tqdm(enumerate(test_df['article_original'])):
    input_ids = [tokenizer.bos_token_id] + tokenizer.encode(article) + [tokenizer.eos_token_id]
    if len(input_ids) > 1026:
        input_ids = input_ids[:1026]
    try:
        summary_ids = model.generate(torch.tensor([input_ids]).to('cuda'), max_length=1024, length_penalty=1, num_beams=5, no_repeat_ngram_size=3, repetition_penalty=3.0, do_sample=True, top_p=0.92, top_k=120, temperature=0.9, early_stopping=True,)
        summary = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
        test_df.loc[idx, 'summary'] = summary
    except:
        summary_ids = model.generate(torch.tensor([input_ids]).to('cuda'), max_length=1024, length_penalty=1, num_beams=5, no_repeat_ngram_size=3, repetition_penalty=3.0, early_stopping=True,)
        summary = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
        test_df.loc[idx, 'summary'] = summary
        
for id_, summary_ in zip(test_df['id'], test_df['summary']):
    idx = submission[submission['id']==int(id_)].index
    summary_ = ' '.join(re.sub('\n','',summary_).split())
    submission.loc[idx, 'summary'] = summary_

submission.to_csv('./data/kobart_epoch15_repeat.csv', index=False)

'''
#%% kobart-law : 사용x
tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')
model = BartForConditionalGeneration.from_pretrained('./KoBART-summarization/kobart_law').to('cuda')

test_df.summary = 0
test_df = test_df[6528:]
test_df = test_df.reset_index()
test_df = test_df[['id','media','article_original']]

for idx, article in enumerate(test_df['article_original']):
    input_ids = [tokenizer.bos_token_id] + tokenizer.encode(article) + [tokenizer.eos_token_id]
    if len(input_ids) > 1026:
        input_ids = input_ids[:1026]
    try:
        summary_ids = model.generate(torch.tensor([input_ids]).to('cuda'), max_length=1024, length_penalty=1, num_beams=5, no_repeat_ngram_size=3, repetition_penalty=3.0, do_sample=True, top_p=0.92, top_k=120, temperature=0.9, early_stopping=True,)
        summary = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
        summary = ' '.join(re.sub('\n','',summary).split())
        test_df.loc[idx, 'summary'] = summary
    except:
        summary_ids = model.generate(torch.tensor([input_ids]).to('cuda'), max_length=1024, length_penalty=1, num_beams=5, no_repeat_ngram_size=3, repetition_penalty=3.0, early_stopping=True,)
        summary = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
        summary = ' '.join(re.sub('\n','',summary).split())        
        test_df.loc[idx, 'summary'] = summary

law_df = test_df.copy()
law_df.to_csv('./data/law_df.csv', index=False)


#%% gas to submission
law_df = pd.read_csv('data/law_df.csv')

submission = pd.read_csv('data/kobart_epoch15_repeat.csv')

for id_, summary_ in zip(law_df['id'], law_df['summary']):
    idx = submission[submission['id']==int(id_)].index
    summary_ = ' '.join(re.sub('\n','',summary_).split())
    submission.loc[idx, 'summary'] = summary_

submission.to_csv('./data/kobart_epoch15_repeat_law.csv', index=False)
'''