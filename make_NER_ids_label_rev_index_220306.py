import os
import pandas as pd
from transformers import BertTokenizer
import six
import re
import json
import numpy as np
import time
import gc
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from transformers import TFElectraModel
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Activation, Lambda, Input
from tf2crf import CRF, ModelWithCRFLoss

from NER_util import keras_pad_fn
from konlpy.tag import Mecab
mecab = Mecab()

df_NER = pd.read_csv("uniq_keywords, codes, uniq_token_keywords_220306.csv") 
df_summary = pd.read_csv("2014-2019_AIHUB_text_shuffle_220306.csv")
# df_keyword_code_freq = pd.read_csv("keyword-code_freq.csv")
# with open("dict_keyword_freqdist.json", 'rb') as f:
#     dict_keyword_freqdist = json.load(f)
save_filename = 'NER_ids_label_rev_index_220306.csv'

keywords = df_NER['keyword']
codes = df_NER['code']
token_keywords = df_NER['token_keyword']
summarys = df_summary['text']
# token_summarys = df_summary['token_summary']
token_summarys = df_summary['mecab_text']
# set_keyword_code = dict(zip(keywords, codes)) # 키워드와 코드를 딕셔너리로 만들어줌
set_token_keyword_code = dict(zip(token_keywords, codes)) # 토큰화된 키워드와 코드를 딕셔너리로 만들어줌

del df_NER
del df_summary
gc.collect()

# tokenizer = BertTokenizer.from_pretrained('vocab.txt', do_lower_case=False) # 기존 vocab
tokenizer = BertTokenizer.from_pretrained('wpm-vocab-all.txt', do_lower_case=False) # NTIS(2012-2019) + AIHUB 데이터로 만든 vocab(32000)

max_len = 256
if os.path.isfile(save_filename): # 파일이 존재하면 continue
    print("{} 파일이 존재합니다".format(save_filename))
    df = pd.read_csv(save_filename)
    idx = len(df) # 기존 파일 길이
    del df # 변수 삭제
    gc.collect() # 메모리 삭제
    print("기존 파일 길이 : ", idx)
else: # 파일이 존재하지 않으면 DataFrame을 만들고 파일 저장
    print("{} 파일이 존재하지 않습니다".format(save_filename))
    df = pd.DataFrame(columns=['summary', 'NER_ids', 'NER_label'])
    df.to_csv(save_filename, index=False, mode='a', encoding='utf-8')
    idx = 0 # 기존 파일이 없다면 0으로 초기화
    print("{} 파일 생성".format(save_filename))

for summary, token_summary in tqdm(zip(summarys[idx:], token_summarys[idx:]), total = len(summarys[idx:])): # idx 값부터 시작
#     token_summary = tokenizer.tokenize(summary)
#     token_summary = tokenizer.tokenize(' '.join(mecab.morphs(summary)))
    try:
        token_summary = token_summary[1:-1].replace("'", "").split(", ")
    except:
        print("summarys {}번째".format(idx))
        print("이상한 token summary : ", token_summary)
        continue
        
    y_data = []
    str_y_data = []
    token_summary = token_summary[:max_len]
    list_of_ner_label = ['O' for i in range(len(token_summary))]
    is_keyword = False # 키워드 존재 유무 확인
    for token_keyword in (token_keywords): # 토큰화된 키워드 리스트에서 토큰화된 키워드를 하나씩 꺼냄
        chk = False 
        token_list = token_keyword.split() # token_keyword를 리스트로 바꿔줌
        try:
            match_count = token_summary.count(token_list[0]) # 토큰화된 키워드의 첫 번째 토큰이 토큰화된 요약에 존재하는 개수
        except:
            continue # 토큰화된 키워드의 첫 번째 토큰이 토큰화된 요약에 없으면 continue
        
        for _ in range(match_count): # 토큰화된 키워드의 첫 번째 토큰이 토큰화된 요약에 존재하는 개수만큼 반복
            try:
                first_idx = token_summary.index(token_list[0]) # 토큰화된 키워드의 첫 번째 토큰이 토큰화된 요약에서 등장하는 index값
            except:
                print("\n")
                print("summarys {}번째".format(idx))
                print("token_list 길이 : ", len(token_list))
                print("token_summary 안에 {} 이(가) 존재하지 않음 : ".format(token_list[0]))
                continue
            token_list_len = len(token_list)

            if token_list[0] == token_summary[first_idx]: # 토큰화된 키워드의 첫 번째 토큰이 토큰화된 요약의 first_idx번째 토큰과 같다면
                for i, token in enumerate(token_list): # 이후 토큰들도 토큰화된 키워드의 토큰과 같은지 확인
                    if first_idx + i >= len(token_summary): # 다음 토큰 인덱스가 토큰화된 요약의 길이보다 크거나 같다면 break
                        chk = False
                        break
                    if token != token_summary[first_idx+i]: # 이후 토큰이 토큰화된 키워드의 토큰과 다르다면 break
                        chk = False
                        break
                    chk = True

            if chk: # 토큰화된 키워드와 모두 일치하는 토큰이라면
                if list_of_ner_label[first_idx] == 'O': # 이미 처리된 태그가 아니라면 태깅 작업을 해줌
                    for i in range(first_idx, first_idx + token_list_len):
                        if i >= max_len: # 다음 토큰 인덱스가 max_len보다 크거나 같다면 break
                            break
                        if i == first_idx: # 첫 번째 토큰이라면 B 태그
                            ner_tag = 'B▁' + set_token_keyword_code[token_keyword]
                            list_of_ner_label[i] = ner_tag
                            token_summary[i] = '[REP]' # 태깅된 요약 토큰은 다시 매칭되지 않도록 [REP]로 변경
                        else: # 두 번째부터의 토큰이라면 I 태그
                            ner_tag = 'I▁' + set_token_keyword_code[token_keyword]
                            list_of_ner_label[i] = ner_tag
                            token_summary[i] = '[REP]' # 태깅된 요약 토큰은 다시 매칭되지 않도록 [REP]로 변경
                is_keyword = True # 키워드가 존재
    idx += 1
    
    with open("ner_to_index_rev_220214.json", 'rb') as f:
        ner_to_index = json.load(f)
    # ner_str -> ner_ids -> cls + ner_ids + sep -> cls + ner_ids + sep + pad + pad .. + pad
    if len(list_of_ner_label) < max_len - 1:
        list_of_ner_ids = [ner_to_index['[CLS]']] + [ner_to_index[ner_tag] for ner_tag in list_of_ner_label] + [ner_to_index['[SEP]']]
    else:
        list_of_ner_ids = [ner_to_index['[CLS]']] + [ner_to_index[list_of_ner_label[i]] for i in range(max_len-2)] + [ner_to_index['[SEP]']]
    list_of_ner_ids = keras_pad_fn([list_of_ner_ids], pad_id=2, maxlen=max_len)[0]
    
    if is_keyword: # 키워드가 존재하지 않았다면 저장하지 않음
        y_data.append(list_of_ner_ids)
        str_y_data.append(list_of_ner_label)
        res_df = pd.DataFrame({'summary' : summary, 'NER_ids' : y_data, 'NER_label' : str_y_data})
        res_df.to_csv(save_filename, index=False, header=False, mode='a', encoding='utf-8') # 파일로 하나씩 저장
