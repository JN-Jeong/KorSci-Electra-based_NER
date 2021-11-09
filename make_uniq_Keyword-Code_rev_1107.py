import os
import pandas as pd
from transformers import BertTokenizer
import six
import re
import json
import numpy as np
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from transformers import TFElectraModel
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Activation, Lambda, Input
from tf2crf import CRF, ModelWithCRFLoss

tokenizer = BertTokenizer.from_pretrained('vocab.txt', do_lower_case=False)

def df_refine(df):
    # df = df.replace(',', ' ', regex=True)
    df = df.replace('[^ 가-힣0-9a-zA-Z!@#$%^&*.\n,]|[가나다라마바사]\)|[0-9]-?[0-9]?\)', '', regex=True)
    df = df[df['연구분야분류코드1']!='']
    df = df[df['과제_한글키워드']!='']
    df = df[df['연구내용요약']!='']
    df = df[df['연구분야분류코드1']!='-']
    df = df[df['과제_한글키워드']!='-']
    df = df[df['연구내용요약']!='-']
    df = df[df["과제_한글키워드"].str.len() > 1] # 키워드가 한 글자인 경우는 제외 (한 글자로는 옻, 배 2개가 있었음)
    df = df[df["과제_한글키워드"] != '보안'] # 키워드가 보안인 경우는 제외 (요약 내용이 제대로 나오지 않음)
    # df['과제_한글키워드'] = df['과제_한글키워드'].str.replace(' ', '')
    df = df.reset_index(drop=True)
    return df

# word에 한글과 영문 글자가 포함되어 있지 않으면 삭제 해주기
# 입력은 키워드(문자열)
# 반환은 bool, 한글과 영어가 포함된 키워드는 True를 반환, 한글과 영어가 포함되지 않은 키워드라면 False를 반환
def check_hangul(word):
    hangul_pattern = re.compile("["
                                u"\U0000AC00-\U0000D7AF"  # 한글 문자
                                u"\U00000041-\U0000007A"   # 영어 알파벳
                                "]+", flags=re.UNICODE)
    res = hangul_pattern.search(word)
    if res == None:
        return False
    else:
        return True
    
dir_path = 'jn'

keywords = []
summarys = []
codes = []
for file in os.listdir(dir_path):
    if 'xls' in file:
        df = df_refine(pd.read_excel(os.path.join(dir_path, file)))
        for i in tqdm(range(len(df))):
#         for i in range(500):
            if '99' in df['연구분야분류코드1'].iloc[i]: # 분류되지 않은 태그는 제외
                continue
            keyword = df['과제_한글키워드'].iloc[i].replace(' ,', '').replace(', ', ',')
            summary = re.sub(r"\s{1,}", " ", df['연구내용요약'].iloc[i]) # 2번 이상 반복되는 공백문자를 공백문자 하나로 변경
            if check_hangul(keyword): # 키워드에 한글과 영어가 포함되어 있으면 추가
                keywords.append(keyword)
                codes.append(df['연구분야분류코드1'].iloc[i])
            
            if check_hangul(summary): # 요약에 한글과 영어가 포함되어 있으면 추가
                summarys.append(summary)
            

print("중복 제거 전 summarys 길이 : ", len(summarys))
summarys = list(set(summarys)) # 요약 중복 제거
print("중복 제거 후 summarys 길이 : ", len(summarys))

df_key_code = pd.DataFrame({"keyword" : keywords, "code" : codes})
df_summary = pd.DataFrame({"summary" : summarys})
print("전처리 전 df_summarys 길이 : ", len(summarys))
idx = []
for i in range(len(df_summary)):
    if '보안상 생략' in df_summary['summary'].iloc[i]:
        idx.append(i)
    elif '세부내용 생략' in df_summary['summary'].iloc[i]:
        idx.append(i)
    elif '세부사항 생략' in df_summary['summary'].iloc[i]:
        idx.append(i)
    elif '보안과제임으로 비공개' in df_summary['summary'].iloc[i]:
        idx.append(i)
    elif '보안과제로 생략' in df_summary['summary'].iloc[i]:
        idx.append(i)
df_summary = df_summary.drop(df_summary.index[idx])
df_summary = df_summary[df_summary['summary'].str.len() > 20] # 요약 길이가 20 이하이면 제외
df_summary = df_summary.drop_duplicates(['summary']) # 요약 중복 제거
df_summary = df_summary.reset_index(drop=True)
print("전처리 후 summarys 길이 : ", len(df_summary))

print("키워드에 한글과 영어가 포함되어 있지 않으면 삭제")
print("삭제 전 키워드 길이 : ", len(df_key_code))
idx = []
for i in range(len(df_key_code)):
    if not check_hangul(df_key_code['keyword'].iloc[i]):
        idx.append(i)
df_key_code = df_key_code.drop(df1.index[idx])
df_key_code = df_key_code.reset_index(drop=True)
print("삭제 후 키워드 길이 : ", len(df_key_code))

each_keywords = []
each_codes = []
for keyword, code in zip(keywords, codes):
    keyword = keyword.split(',')
    if len(code) < 4: # 코드 길이가 4미만이면 제외
        continue
    for word in keyword:
        # 약간의 정제
        word = word.strip()
        word = word.replace('\n', ' ')
        word = re.sub('[0-9]\.', '', word)
        if len(word) < 2: # 키워드가 2글자 이상이 아니라면 제외
            continue
        if not word: # word가 비어있다면 제외
            continue
        try:
            if len(word.split()[0]) > 10: # 키워드 길이가 10을 초과하면 제외
                continue
        except:
            print("이상한 키워드 : ", (word))
        each_keywords.append(word)
        each_codes.append(code[:4])
        
set_codes = set(each_codes)
print("code 개수 : ", len(set_codes))
print("code list")
print(set_codes)

# ner_to_index.json 파일 만들기
uniq_code = set_codes
ner_to_index = ['[CLS]']
ner_to_index.append('[SEP]')
ner_to_index.append('[PAD]')
ner_to_index.append('[MASK]')
ner_to_index.append('O')
for code in uniq_code:
    ner_to_index.append("B▁{}".format(code))
    ner_to_index.append("I▁{}".format(code))
idx = 0
set_ner_to_index = {}
for ner in ner_to_index:
    set_ner_to_index[ner] = idx
    idx += 1
with open("ner_to_index_rev_1030.json", "w") as json_file:
    json.dump(set_ner_to_index, json_file)
# ner_to_index.json 파일 만들기 끝
        
df_eachKey_eachCode = pd.DataFrame({"each_keyword" : each_keywords, "each_code" : each_codes})
df_eachKey_eachCode = df_eachKey_eachCode[df_eachKey_eachCode['each_keyword'].str.len() < 20] # 키워드 길이가 20이상이면 제외
df_eachKey_eachCode = df_eachKey_eachCode.reset_index(drop=True) # 인덱스 초기화
print("키워드 길이가 20 미만인 키워드 개수 : ", len(df_eachKey_eachCode)) # 키워드 개수

print("중복 제거 전 키워드 개수 : ", len(df_eachKey_eachCode)) # 825457
uniq_keywords = list(set(df_eachKey_eachCode['each_keyword'])) # 중복없는 키워드들
print("중복 없는 키워드 개수 : ", len(uniq_keywords)) # 232410

dict_keyword_freqdist = {}
for keyword in tqdm(df_eachKey_eachCode['each_keyword']):
    list_keywords = list(df_eachKey_eachCode[df_eachKey_eachCode['each_keyword'] == keyword]['each_code'])
    freqdist = nltk.FreqDist(list_keywords)
    dict_keyword_freqdist[keyword] = freqdist
    
list_keyword = []
list_len_freqdist = []
for keyword in tqdm(df_eachKey_eachCode['each_keyword']):
    list_keyword.append(keyword)
    list_len_freqdist.append(len(dict_keyword_freqdist[keyword]))
    
df_keyword_code_freq = pd.DataFrame({"keyword" : list_keyword, "freq" : list_len_freqdist})
df_keyword_code_freq = df_keyword_code_freq.drop_duplicates('keyword')
df_keyword_code_freq = df_keyword_code_freq.reset_index(drop=True)
# df_keyword_code_freq.to_csv("keyword-code_freq.csv", index=False)

# 중복 코드를 갖는 키워드는 빈도가 많은 코드로 태깅
temp_df = df_eachKey_eachCode[df_eachKey_eachCode.duplicated('each_keyword', keep=False)]
list_uniq_keyword = []
list_uniq_code = []
for word in tqdm(uniq_keywords):
    keyword_df = temp_df[temp_df['each_keyword'] == word]
    if len(keyword_df) < 1: # 중복되는 키워드가 아니라면
#         print(word)
        try:
            key_code = df_eachKey_eachCode[df_eachKey_eachCode['each_keyword'] == word].iloc[0]['each_code']
        except:
            print(word)
            print(df_eachKey_eachCode[df_eachKey_eachCode['each_keyword'] == word])
        list_uniq_keyword.append(word)
        list_uniq_code.append(key_code)
        continue
    set_code = {}
    for i in range(len(keyword_df)):
        code = temp_df[temp_df['each_keyword'] == word].iloc[i]['each_code']
        if code not in set_code:
            set_code[code] = 0
        set_code[code] += 1
#     key_code = sorted(set_code.items(), key=lambda item:-item[1])[0][0] # 정렬 시간이 꽤 잡아먹을듯... 수정이 필요해보임
    key_code = max(list(set_code)) # 빈도 수가 같다면 코드의 알파벳이 알파벳 순서상 뒤에 오는 코드로 선택됨

    freq = int(df_keyword_code_freq[df_keyword_code_freq['keyword']==word]['freq'])
    if freq <= 5: # 키워드에 대응되는 분류코드가 5개 이하라면 키워드로 인정
        sort_code_freq = sorted(dict_keyword_freqdist[keyword].items(), key=lambda item:-item[1])
        if len(sort_code_freq) > 1:
            if sort_code_freq[0][1] >= 100: # 가장 큰 빈도수가 100개 이상이라면 키워드로 인정
                if sort_code_freq[0][1] - sort_code_freq[1][1] > sort_code_freq[0][1]*0.1: # 빈도 수 1순위, 2순위의 차가 1순위의 10%보다 크면 키워드로 인정
                    list_uniq_keyword.append(word)
                    list_uniq_code.append(key_code)
            else:
                if sort_code_freq[0][1] - sort_code_freq[1][1] > 3: # 빈도 수 1순위, 2순위의 차가 3보다 크면 키워드로 인정
                    list_uniq_keyword.append(word)
                    list_uniq_code.append(key_code)
        elif len(sort_code_freq) == 1:
            list_uniq_keyword.append(word)
            list_uniq_code.append(key_code)

    
# 요약과 토큰화된 요약 데이터프레임 만들기
token_summarys = []
for i in tqdm(range(len(df_summary))):
    token_summarys.append(tokenizer.tokenize(df_summary['summary'].iloc[i]))
df_summary = pd.DataFrame({"summary" : list(df_summary['summary']), "token_summary" : token_summarys})
print("df_summary 길이 : ", len(df_summary))

file_name = "2014-2017_summary_1030.csv"
print("{} 파일 저장".format(file_name))
df_summary.to_csv(file_name, index=False)
# 요약과 토큰화된 요약 데이터프레임 만들기 끝

# 키워드, 코드, 토큰화된 키워드 데이터프레임 만들기
list_token_keyword = []
for keyword in tqdm(uniq_keywords):
    temp = m.morphs(keyword)
    temp = tokenizer.tokenize(' '.join(temp))
    temp_str = ''
    for i in temp:
        temp_str += i + ' '
    temp_str = temp_str.strip()
    list_token_keyword.append(temp_str)
df_uniq_key_code_token = pd.DataFrame({"keyword" : list_uniq_keyword, "code" : list_uniq_code, "token_keyword" : list_token_keyword})
    
df_uniq_key_code_token = df_uniq_key_code_token[df_uniq_key_code_token['token_keyword'] != '[UNK]']
df_uniq_key_code_token = df_uniq_key_code_token.drop_duplicates(['keyword']) # 키워드 중복 제거
df_uniq_key_code_token = df_uniq_key_code_token.drop_duplicates(['token_keyword']) # 토큰화된 키워드 중복 제거
df_uniq_key_code_token = df_uniq_key_code_token.sort_values(by=["keyword"],key=lambda x:x.str.len(), ascending=False)
df_uniq_key_code_token = df_uniq_key_code_token.reset_index(drop=True)
print("df_uniq_key_code_token 길이 : ", len(df_uniq_key_code_token))

file_name = "uniq_keywords, codes, uniq_token_keywords_1030.csv"
print("{} 파일 저장".format(file_name))
df_uniq_key_code_token.to_csv(file_name, index=False)
# 키워드, 코드, 토큰화된 키워드 데이터프레임 만들기 끝