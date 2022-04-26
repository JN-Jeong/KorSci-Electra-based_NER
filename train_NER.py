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
from sklearn.metrics import classification_report
from seqeval.metrics import classification_report as classification_report_seqeval

import tensorflow as tf
from tensorflow import keras
from transformers import TFElectraModel, TFBertModel
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Activation, Lambda, Input
from tf2crf import CRF, ModelWithCRFLoss

import argparse
import time

from konlpy.tag import Mecab
mecab = Mecab()

def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
parser = argparse.ArgumentParser(description='test')
parser.add_argument(
    '-mecab', help='insert whether apply meacb or not', default=False, type=bool)
parser.add_argument(
    '-max_length', help='insert max length', default=256, type=int)
parser.add_argument(
    '-lr', help='insert learning rate', default=5e-5, type=float)
parser.add_argument(
    '-model', help='insert model(electra or bert)', default=None, type=str)
parser.add_argument(
    '-data', help='insert data file name', default=None, type=str)
parser.add_argument(
    '-ner', help='insert ner to index file name', default=None, type=str)
parser.add_argument(
    '-desc', help='insert description to model', default=None, type=str)
parser.add_argument(
    '-slicing', help='insert the number of slicing', default=-1, type=int)
parser.add_argument(
    '-top_K', help='insert the number of K', default=1, type=int)
args = parser.parse_args()
    
print('max_length : {}'.format(args.max_length))
print('learning rate : {}'.format(args.lr))
print('Desc : {}'.format(args.desc))
print('Model : {}'.format(args.model))
print('slicing : {}'.format(args.slicing))
print('top K : {}'.format(args.top_K))
print('data file : {}'.format(args.data))
print('ner file : {}'.format(args.ner))

MAX_LENGTH = args.max_length

# tokenizer = BertTokenizer.from_pretrained('vocab.txt', do_lower_case=False) # 기존 vocab
tokenizer = BertTokenizer.from_pretrained('wpm-vocab-all.txt', do_lower_case=False) # NTIS(2012-2019) + AIHUB 데이터로 만든 vocab(32000)

with open(args.ner, 'rb') as f:
    ner_to_index = json.load(f)
index_to_ner = {}
for key, value in ner_to_index.items():
    index_to_ner[value] = key
    
df_datas = pd.read_csv(args.data)

if args.slicing == -1:
    x_data = df_datas['summary'].tolist()
    y_data = df_datas['NER_ids'].tolist()
else:
    x_data = df_datas['summary'].tolist()[:args.slicing]
    y_data = df_datas['NER_ids'].tolist()[:args.slicing]
    
for i in range(len(y_data)):
    y_data[i] = [int(i) for i in (y_data[i][1:-1].split())]
str_y_data = df_datas['NER_label']

str_train_x, str_test_x, train_y, test_y = train_test_split(x_data, y_data, test_size=0.3, random_state=72)

print(f"Train : {len(str_train_x)} / Test : {len(str_test_x)}")

encode_train_data = []
input_ids =[]
attention_masks =[]
token_type_ids =[]
for line in tqdm(str_train_x):
    line = ' '.join(mecab.morphs(line)) # mecab 적용, encode하면 tokenizer.tokenize 해준 것과 같은 결과 나옴
    encoded_dict = tokenizer.encode_plus(line, \
                                         add_special_tokens = True,\
                                         pad_to_max_length=True,\
                                         max_length=MAX_LENGTH, 
                                        return_attention_mask=True,
                                        truncation = True)

    input_id=encoded_dict['input_ids']
    attention_mask=encoded_dict['attention_mask']
    token_type_id = encoded_dict['token_type_ids']
    
    input_ids.append(input_id)
    attention_masks.append(attention_mask)
    token_type_ids.append(token_type_id)

train_input_ids=np.array(input_ids, dtype=int)
train_attention_masks=np.array(attention_masks, dtype=int)
train_token_type_ids=np.array(token_type_ids, dtype=int)
train_x=(train_input_ids, train_attention_masks, train_token_type_ids)

print("Original Text : ", x_data[0])
print("Tokenizer Text : ", tokenizer.tokenize(x_data[0]))
print("Encode Text : ", (tokenizer.encode(x_data[0], add_special_tokens = True, max_length=MAX_LENGTH)))

np_train_y = np.array(train_y)
np_test_y = np.array(test_y)

BATCH_SIZE = 32
vocab_size = len(tokenizer)
tag_size = len(ner_to_index)

# 모델 설정
what_model = args.model

if 'electra' in what_model:
    if what_model == 'electra':
        model = TFElectraModel.from_pretrained('../nscc_mecab_base/torch_model', from_pt=True)
    elif what_model == 'electra_final':
        model = TFElectraModel.from_pretrained('../nscc_mecab_base_final/torch_model', from_pt=True)
    elif what_model == 'electra_monologg':
        model = TFElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator", from_pt=True)
    electra_layers = model.get_layer('electra')
    input_layer = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype='int32') # [(None, 256)]
    electra_l = electra_layers(input_layer)[0]

    # X = tf.keras.layers.Lambda(lambda seq: seq[:, 0, :])(electra_l)
    X = tf.keras.layers.Dropout(0.1)(electra_l)

    X = tf.keras.layers.Dropout(0.1)(X)
    X = tf.keras.layers.Dense(units=256, activation="relu")(X)
    X = tf.keras.layers.Dropout(0.1)(X)

    output_layer = tf.keras.layers.Dense(MAX_LENGTH, activation="relu")(X) # negative loss를 피하기 위해서 relu 함수를 사용해준다?
    #     output_layer = tf.keras.layers.Dense(MAX_LENGTH,activation='softmax')(X)
    crf = CRF(units=tag_size)
    output_layer = crf(output_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    print(model.summary())
    model = ModelWithCRFLoss(model, sparse_target=True)
    model.build(input_shape=(None, MAX_LENGTH))
    print(model.summary())
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss = keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=args.top_K)]
    )

    checkpoint_filepath = "checkpoints/" + what_model + "_NER_" + args.desc + "/" + what_model + "_{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_filepath)
    model.save_weights(checkpoint_filepath.format(epoch=0))

elif 'bert' in what_model:
    if what_model == 'bert':
        model = TFBertModel.from_pretrained('bert-base-multilingual-cased')
    elif what_model == 'korscibert':
        model = TFBertModel.from_pretrained('../KorsciBert/torch_model', from_pt=True)
    elif what_model == 'kluebert':
        model = TFBertModel.from_pretrained('klue/bert-base', from_pt=True)
    bert_layers = model.get_layer('bert')
    input_layer = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype='int32') # [(None, 256)]
    bert_l = bert_layers(input_layer)[0]

    # X = tf.keras.layers.Lambda(lambda seq: seq[:, 0, :])(electra_l)
    X = tf.keras.layers.Dropout(0.1)(bert_l)

    X = tf.keras.layers.Dropout(0.1)(X)
    X = tf.keras.layers.Dense(units=256, activation="relu")(X)
    X = tf.keras.layers.Dropout(0.1)(X)

    output_layer = tf.keras.layers.Dense(MAX_LENGTH, activation="relu")(X) # negative loss를 피하기 위해서 relu 함수를 사용해준다? -> 효과 없음
    #     output_layer = tf.keras.layers.Dense(MAX_LENGTH,activation='softmax')(X)
    crf = CRF(units=tag_size)
    output_layer = crf(output_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    print(model.summary())
    model = ModelWithCRFLoss(model, sparse_target=True)
    model.build(input_shape=(None, MAX_LENGTH))
    print(model.summary())
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss = keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=args.top_K)]
    )

    checkpoint_filepath = "checkpoints/" + what_model + "_NER_" + args.desc + "/" + what_model + "_{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_filepath)
    model.save_weights(checkpoint_filepath.format(epoch=0))

elif 'LSTM' in what_model:
    embedding_dim = 768 ############### 
    hidden_units = 768 ###############
    dropout_ratio = 0.3

    sequence_input = Input(shape=(MAX_LENGTH,),dtype=tf.int32, name='sequence_input')

    model_embedding = Embedding(input_dim=vocab_size,
                                output_dim=embedding_dim,
                                input_length=MAX_LENGTH)(sequence_input)

    model_bilstm = Bidirectional(LSTM(units=hidden_units, return_sequences=True))(model_embedding)

    model_dropout = TimeDistributed(Dropout(dropout_ratio))(model_bilstm)

    model_dense = TimeDistributed(Dense(tag_size, activation='relu'))(model_dropout)

    crf = CRF(units=tag_size)
    output_layer = crf(model_dense)

    base = Model(inputs=sequence_input, outputs=output_layer)
    print(base.summary())
    model = ModelWithCRFLoss(base, sparse_target=True)
    model.build(input_shape=(None, MAX_LENGTH))
    print(model.summary())

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss = keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=args.top_K)]
    )
    
    checkpoint_filepath = "checkpoints/" + what_model + "_NER_" + args.desc + "/" + what_model + "_{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_filepath)
    model.save_weights(checkpoint_filepath.format(epoch=0))

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss_val', patience=3)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    verbose=1,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
#     period=1
    save_freq='epoch'
)
history = model.fit(train_x, np_train_y, batch_size=BATCH_SIZE,validation_split=0.2, epochs=100, callbacks=[callback, model_checkpoint_callback])

# 모델 저장
model_filepath = "models/" + what_model + "_NER_" + args.desc + "_model"
model.save(model_filepath)

latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)

# model.load_weights('checkpoints/electra_NER_21.10.26-3/electra_0000.ckpt')

encode_test_data = []
input_ids =[]
attention_masks =[]
token_type_ids =[]
for line in tqdm(str_test_x):
#     if args.mecab:
# #         line = mecab.morphs(line) # mecab 적용
#         line = tokenizer.tokenize(' '.join(mecab.morphs(line))) # mecab 적용
    encoded_dict = tokenizer.encode_plus(line, \
                                         add_special_tokens = True,\
                                         pad_to_max_length=True,\
                                         max_length=MAX_LENGTH, 
                                        return_attention_mask=True,
                                        truncation = True)

    input_id=encoded_dict['input_ids']
    attention_mask=encoded_dict['attention_mask']
    token_type_id = encoded_dict['token_type_ids']
    
    input_ids.append(input_id)
    attention_masks.append(attention_mask)
    token_type_ids.append(token_type_id)

test_input_ids=np.array(input_ids, dtype=int)
test_attention_masks=np.array(attention_masks, dtype=int)
test_token_type_ids=np.array(token_type_ids, dtype=int)
test_x=(test_input_ids, test_attention_masks, test_token_type_ids)

y_pred = model.predict(test_x)
for i in y_pred[:20]:
    print(i)
    
list_y_pred = list(y_pred)
list_test_input_ids = list(test_input_ids)

# score_his = model.evaluate(test_x, test_y)
# print(score_his)

target_names = []
none_target_names = ['[CLS]', '[SEP]', '[PAD]', '[MASK]', 'O']
for ner_tag in ner_to_index.keys():
    if ner_tag in none_target_names:
        continue
    else:
        target_names.append(ner_tag)
#     target_names.append(ner_tag)
    
print("len test_y : ", len(test_y))
print("len y_pred : ", len(y_pred))
print("len str_test_x : ", len(str_test_x))
print("len test_x[0](ids) : ", len(test_x[0]))
print("target names ; ", target_names)
print("tag size : ", tag_size)
print("code 종류 : ", (tag_size-5)//2)

report_test_y = [] # classification report용 test_y(실제값) 리스트를 만들어줌 (태깅 코드를 하나씩 리스트로 추가해줌)
report_test_y_code = []
report_y_pred = [] # classification report용 y_pred(예측값) 리스트를 만들어줌 (태깅 코드를 하나씩 리스트로 추가해줌)
report_y_pred_code = []
for test_ids_list, pred_ids_list in tqdm(zip(np_test_y, y_pred), total = len(test_y)):
    test_y_code = []
    y_pred_code = []
    for test_ids, pred_ids in zip(test_ids_list, pred_ids_list):
        report_test_y.append(test_ids)
        report_y_pred.append(pred_ids)
        test_code = index_to_ner[test_ids].replace('B▁', 'B-').replace('I▁', 'I-').replace('[CLS]', 'O').replace('[SEP]', 'O').replace('[PAD]', 'O')
        pred_code = index_to_ner[pred_ids].replace('B▁', 'B-').replace('I▁', 'I-').replace('[CLS]', 'O').replace('[SEP]', 'O').replace('[PAD]', 'O')
        if test_code == '[CLS]' or test_code == '[SEP]' or test_code == '[PAD]':
            continue
        if not test_code:
            print("test : ", index_to_ner[test_ids])
            print("pred : ", index_to_ner[pred_ids])
        test_y_code.append(test_code)
        y_pred_code.append(pred_code)
    report_test_y_code.append(test_y_code)
    report_y_pred_code.append(y_pred_code)
        
label_index_to_print = list(range(5, tag_size))  # ner label indice except '[CLS]', '[SEP]', '[PAD]', '[MASK]' and 'O' tag
# label_index_to_print = list(range(tag_size))  # ner label indice except '[CLS]', '[SEP]', '[PAD]', '[MASK]' and 'O' tag
# print(classification_report(y_true=report_test_y, y_pred=report_y_pred, target_names=target_names, labels=label_index_to_print, digits=4))

print("len report_y_pred : ", len(report_y_pred))
print("len report_test_y : ", len(report_test_y))
print("type y_pred : ", type(y_pred))
print("type y_pred[0] : ", type(y_pred[0]))

report_file_name = 'NER/report/{model}/report_NER_{model}_{desc}.txt'.format(model=args.model, desc=args.desc)
with open(report_file_name, 'w') as f:
#         f.write(classification_report(test_y, np.array(pred_y), target_names=categories.keys()))
    f.write(classification_report(y_true=report_test_y, y_pred=report_y_pred, target_names=target_names, labels=label_index_to_print, digits=4))
    
predict_file_name = 'NER/predict/{model}/predict_NER_{model}_{desc}.csv'.format(model=args.model, desc=args.desc)
df_predict = pd.DataFrame({'input' : str_test_x, 'encode_input' : list_test_input_ids, 'target' : test_y, 'predict' : list_y_pred})
df_predict.to_csv(predict_file_name, index=False)
    
report_file_name = 'NER/report/{model}/report_NER_code_{model}_{desc}.txt'.format(model=args.model, desc=args.desc)
with open(report_file_name, 'w') as f:
    f.write(classification_report_seqeval(y_true=report_test_y_code, y_pred=report_y_pred_code))

print('example_NER-all_keyword.py 실행 끝')
    