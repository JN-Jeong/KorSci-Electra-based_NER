import numpy as np
import six

def keras_pad_fn(token_ids_batch, maxlen, pad_id=0, padding='post', truncating='post'):
    padded_token_ids_batch = pad_sequences(token_ids_batch,
                                            value=pad_id,  # vocab.transform_token2idx(PAD),
                                            padding=padding,
                                            truncating=truncating,
                                            maxlen=maxlen)
    return padded_token_ids_batch

# pad_sequences_fn in keras.preprocessing.sequence.pad_sequences
def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length.
    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.
    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the end.
    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.
    Pre-padding is the default.
    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.
    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`
    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    for x in sequences:
        try:
            lengths.append(len(x))
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

def transform_token2idx(token, show_oov=False):
        try:
            return tokenizer.convert_tokens_to_ids(token)
        except:
            if show_oov is True:
                print("key error: " + str(token))
            token = ["UNK"]
            return tokenizer.convert_tokens_to_ids(token)
        
def list_of_string_to_list_of_tokens(X_str_batch):
    X_token_batch = [X_str for X_str in X_str_batch]
    return X_token_batch

def list_of_tokens_to_list_of_cls_sep_token_ids(X_token_batch):
    X_ids_batch = []
    for X_tokens in X_token_batch:
        X_tokens = '[CLS]' + X_tokens + '[SEP]'
        X_ids_batch.append([transform_token2idx(X_token) for X_token in X_tokens])
    return X_ids_batch

def list_of_string_to_arr_of_cls_sep_pad_token_ids(X_str_batch):
#     X_token_batch = list_of_string_to_list_of_tokens(X_str_batch)
    X_token_batch = X_str_batch
#     print("여기 : ", X_token_batch)
    X_ids_batch = list_of_tokens_to_list_of_cls_sep_token_ids(X_token_batch)
#     pad_X_ids_batch = self._pad(X_ids_batch, pad_id=self._vocab.PAD_ID, maxlen=self._maxlen)
    pad_X_ids_batch = keras_pad_fn(X_ids_batch, pad_id=0, maxlen=256)

    return pad_X_ids_batch

def transform_target_fn(label_text, tokens, max_len=256):
    regex_ner = re.compile('<(.+?):[A-Z]{2}[0-9]+>') # NER Tag가 4자리(e.g. LC05)인 경우
    regex_filter_res = regex_ner.finditer(label_text)

    list_of_ner_tag = []
    list_of_ner_text = []
    list_of_tuple_ner_start_end = []

    tokenize_ner_texts = []
    for match_item in regex_filter_res:
        ner_tag = match_item[0][-5:-1]
        ner_text = match_item[1]
        tokenize_ner_text = tokenizer.tokenize(ner_text)
        tokenize_ner_texts.append(tokenize_ner_text)
    
    list_of_ner_label = []
    iter_ = 0
    match_count = 0
#     print("len tokens : ", len(tokens))
#     print('tokenize_ner_texts : ', tokenize_ner_texts)
    while iter_ < len(tokens):
        if len(tokenize_ner_texts) > 0:
            for i in range(len(tokenize_ner_texts[0])):
                if iter_ == len(tokens):
                    break
                if tokenize_ner_texts[0][i] in tokens[iter_]:
    #             """
    #             키워드는 '건강기능식' 이고 text내에는 '건강기능식품' 이라고 되어있으면 
    #             키워드를 토큰화 했을 때는 건강 + ##기능 + ##식'이 되고 
    #             text를 토큰화 했을 때는 '건강 + ##기능 + ##식품'이 되는 예제가 있어서 
    #             text 토큰화 내에서 키워드 토큰화가 존재하면 if문을 만족하도록 하기 위해
    #             tokenize_ner_texts[0][i] == tokens[iter_] 대신 tokenize_ner_texts[0][i] in tokens[iter_] 으로 써줌
    #             """
                    match_count += 1
                    if i == 0:
                        ner_tag_ = 'B▁' + ner_tag
                        list_of_ner_label.append(ner_tag_)
                        iter_ += 1
                    else:
                        ner_tag_ = 'I▁' + ner_tag
                        list_of_ner_label.append(ner_tag_)
                        iter_ += 1

                    if i+1 == len(tokenize_ner_texts[0]): # tokenize_ner_texts 내의 text를 다 꺼냈다면 첫번째 index값 삭제
                        tokenize_ner_texts.pop(0)
                else:
                    list_of_ner_label.append('O')
                    iter_ += 1
                    break
        else:
            list_of_ner_label.append('O')
            iter_ += 1
        
#     print("text tokenize : ", tokens)
#     print("text encode : ", tokenizer.encode(tokens))
#     print("labels tokenize : ", list_of_ner_label)
#     print("len text tokenize : ", len(tokens))
#     print("len tokenize labels : ", len(list_of_ner_label))
#     print("match count : ", match_count)
    
    with open("ner_to_index.json", 'rb') as f:
        ner_to_index = json.load(f)
    # ner_str -> ner_ids -> cls + ner_ids + sep -> cls + ner_ids + sep + pad + pad .. + pad
    if len(list_of_ner_label) < max_len - 1:
        list_of_ner_ids = [ner_to_index['[CLS]']] + [ner_to_index[ner_tag] for ner_tag in list_of_ner_label] + [ner_to_index['[SEP]']]
    else:
        list_of_ner_ids = [ner_to_index['[CLS]']] + [ner_to_index[list_of_ner_label[i]] for i in range(max_len-2)] + [ner_to_index['[SEP]']]
    list_of_ner_ids = keras_pad_fn([list_of_ner_ids], pad_id=0, maxlen=max_len)[0]
    list_of_ner_ids = list_of_ner_ids.tolist()

    return list_of_ner_ids, list_of_ner_label

def transform_target_fn(df_summarys, df_NER, max_len=256): # 요약 텍스트 데이터프레임과 키워드, 코드 태깅 데이터프레임을 받음
    regex_ner = re.compile('<(.+?):[A-Z]{2}[0-9]+>') # NER Tag가 4자리(e.g. LC05)인 경우
    regex_filter_res = regex_ner.finditer(label_text)

    list_of_ner_tag = []
    list_of_ner_text = []
    list_of_tuple_ner_start_end = []

    tokenize_ner_texts = []
    for match_item in regex_filter_res:
        ner_tag = match_item[0][-5:-1]
        ner_text = match_item[1]
        tokenize_ner_text = tokenizer.tokenize(ner_text)
        tokenize_ner_texts.append(tokenize_ner_text)
        
    # tokens : summary 토큰화한거
    # 
    keywords = df_NER['keyword']
    codes = df_NER['code']
    token_keywords = df_NER['token_keyword']
    summarys = temp_summary['summary']
    set_keyword_code = dict(zip(keywords, codes)) # 키워드와 코드를 딕셔너리로 만들어줌
    set_token_keyword_code = dict(zip(token_keywords, codes)) # 토큰화된 키워드와 코드를 딕셔너리로 만들어줌
    
    token_summarys = []
    for summary in tqdm(summarys):
        token_summarys.append(tokenizer.tokenize(summary))
    
    list_of_ner_label = []
    iter_ = 0
    match_count = 0
    
    for token_summary in token_summarys:
        while iter_ < len(token_summary):
            if len(tokenize_ner_texts) > 0:
                for i in range(len(tokenize_ner_texts[0])):
                    if iter_ == len(tokens):
                        break
                    if tokenize_ner_texts[0][i] in tokens[iter_]:
        #             """
        #             키워드는 '건강기능식' 이고 text내에는 '건강기능식품' 이라고 되어있으면 
        #             키워드를 토큰화 했을 때는 건강 + ##기능 + ##식'이 되고 
        #             text를 토큰화 했을 때는 '건강 + ##기능 + ##식품'이 되는 예제가 있어서 
        #             text 토큰화 내에서 키워드 토큰화가 존재하면 if문을 만족하도록 하기 위해
        #             tokenize_ner_texts[0][i] == tokens[iter_] 대신 tokenize_ner_texts[0][i] in tokens[iter_] 으로 써줌
        #             """
                        match_count += 1
                        if i == 0:
                            ner_tag_ = 'B▁' + ner_tag
                            list_of_ner_label.append(ner_tag_)
                            iter_ += 1
                        else:
                            ner_tag_ = 'I▁' + ner_tag
                            list_of_ner_label.append(ner_tag_)
                            iter_ += 1

                        if i+1 == len(tokenize_ner_texts[0]): # tokenize_ner_texts 내의 text를 다 꺼냈다면 첫번째 index값 삭제
                            tokenize_ner_texts.pop(0)
                    else:
                        list_of_ner_label.append('O')
                        iter_ += 1
                        break
            else:
                list_of_ner_label.append('O')
                iter_ += 1

    #     print("text tokenize : ", tokens)
    #     print("text encode : ", tokenizer.encode(tokens))
    #     print("labels tokenize : ", list_of_ner_label)
    #     print("len text tokenize : ", len(tokens))
    #     print("len tokenize labels : ", len(list_of_ner_label))
    #     print("match count : ", match_count)

        with open("ner_to_index.json", 'rb') as f:
            ner_to_index = json.load(f)
        # ner_str -> ner_ids -> cls + ner_ids + sep -> cls + ner_ids + sep + pad + pad .. + pad
        if len(list_of_ner_label) < max_len - 1:
            list_of_ner_ids = [ner_to_index['[CLS]']] + [ner_to_index[ner_tag] for ner_tag in list_of_ner_label] + [ner_to_index['[SEP]']]
        else:
            list_of_ner_ids = [ner_to_index['[CLS]']] + [ner_to_index[list_of_ner_label[i]] for i in range(max_len-2)] + [ner_to_index['[SEP]']]
        list_of_ner_ids = keras_pad_fn([list_of_ner_ids], pad_id=0, maxlen=max_len)[0]
        list_of_ner_ids = list_of_ner_ids.tolist()

    return list_of_ner_ids, list_of_ner_label