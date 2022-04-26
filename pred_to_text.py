# enc_input : 인코딩된 input 데이터, pred : 예측 데이터
# return 값은 B, I 라벨링된 텍스트 데이터
def pred_to_BI_text(enc_input, pred):
    idx = 0
    label_texts = ''
    for i in range(len(pred)):
        num = int(pred[i])
        if (num) > 4:
            label_texts += tokenizer.decode(enc_input[idx:i]).replace('#', '') + ' '
            label_text = "<{}:{}>".format(tokenizer.decode(enc_input[i]).replace(' ', '').replace('#', ''), index_to_ner[num])
            label_texts += label_text + ' '
            idx = i+1
            
    label_texts = ' '.join(label_texts.split())
    return label_texts

# enc_input : 인코딩된 input 데이터, pred : 예측 데이터
# return 값은 코드 라벨링된 텍스트 데이터
def pred_to_text(enc_input, pred): 
    idx = 0
    prev_idx = 0
    label_texts = ''
    B_chk = False
    while idx < len(pred):
        num = int(pred[idx])
        if num > 4: # 태그인 경우
            label_texts += tokenizer.decode(enc_input[prev_idx:idx]).replace('#', '')  # 태그가 등장하기 전 인덱스 값들을 텍스트로 변환
            text = ''
            while int(pred[idx]) > 4: # 다음 인덱스들이 태그가 아닐 때까지 반복
                if not B_chk: # B_chk 가 False 이면 (B 태그가 등장하지 않았을 때)
                    if 'B▁' in index_to_ner[int(pred[idx])]: # B태그라면
                        code = index_to_ner[int(pred[idx])][2:] # 코드를 저장
                        text += tokenizer.decode(enc_input[idx]).replace('#', '').replace(' ', '') # 인덱스 값을 텍스트로 변환
                        B_chk = True # 이전 인덱스 값이 B태그라는 표시
                        idx += 1 # 다음 인덱스 확인
                    elif 'I▁' in index_to_ner[int(pred[idx])]: # I태그라면
                        text += tokenizer.decode(enc_input[idx]).replace('#', '').replace(' ', '') # 인덱스 값을 텍스트로 변환
                        idx += 1 # 다음 인덱스 확인
                        B_chk = False # 이전 인덱스 값이 B태그가 아니라는 표시
                        break
                    else: # B, I 태그가 아니라면
                        text += tokenizer.decode(enc_input[idx]).replace('#', '').replace(' ', '') # 인덱스 값을 텍스트로 변환
                        B_chk = False # 이전 인덱스 값이 B태그가 아니라는 표시
                        break
                elif B_chk: # B_chk 가 True 이면 (이전에 B 태그가 등장했을 때)
                    if 'I▁' in index_to_ner[int(pred[idx])]: # I 태그라면
                        text += tokenizer.decode(enc_input[idx]).replace('#', '').replace(' ', '') # 인덱스 값을 텍스트로 변환
                        idx += 1 # 다음 인덱스 확인
                        B_chk = False # 이전 인덱스 값이 B태그가 아니라는 표시
                    elif 'B▁' in index_to_ner[int(pred[idx])]:
                        B_chk = False # 이전 인덱스 값이 B태그가 아니라는 표시
                        break
                    else: # B, I 태그가 아니라면
                        B_chk = False # # 이전 인덱스 값이 B태그가 아니라는 표시
                        break
            if len(text) > 0:
                label_text = "<{}:{}>".format(text, code)
                label_texts += ' ' + label_text + ' '
            prev_idx = idx
    #         idx += 1
        else: # 태그가 아닌 경우
            idx += 1
    
    label_texts = ' '.join(label_texts.split())
    return label_texts

df_pred = pd.read_csv('predict_NER_{model}_{desc}.csv')