# coding=utf-8

"""
@File  : evaluation_bert.py
@Author: Xu Qiqiang
@Date  : 2020/12/6 0006
"""
import tensorflow as tf
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


def test_bert_token():
    max_length_test = 20
    test_sentence = '曝梅西已通知巴萨他想离开'

    # add special tokens
    test_sentence_with_special_tokens = '[CLS]' + test_sentence + '[SEP]'
    tokenized = tokenizer.tokenize(test_sentence_with_special_tokens)
    print('tokenized', tokenized)

    # convert tokens to ids in WordPiece
    input_ids = tokenizer.convert_tokens_to_ids(tokenized)

    # precalculation of pad length, so that we can reuse it later on
    padding_length = max_length_test - len(input_ids)

    # map tokens to WordPiece dictionary and add pad token for those text shorter than our max length
    input_ids = input_ids + ([0] * padding_length)

    # attention should focus just on sequence with non padded tokens
    attention_mask = [1] * len(input_ids)

    # do not focus attention on padded tokens
    attention_mask = attention_mask + ([0] * padding_length)

    # token types, needed for example for question answering, for our purpose we will just set 0 as we have just one sequence
    token_type_ids = [0] * max_length_test
    bert_input = {
        "token_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask
    }
    print(bert_input)


def convert_example_to_feature(review):
    return tokenizer.encode_plus(review,
                                 add_special_tokens= True, # add [CLS], [SEP]
                                 max_length=max_length, # max length of the text that can go to BERT
                                 pad_to_max_length=True, # add [PAD] tokens
                                 return_attention_mask=True, # add attention mask to not focus on pad tokens
                                )

# map to the expected input to TFBertForSequenceClassification, see here


def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
    return {
      "input_ids": input_ids,
      "token_type_ids": token_type_ids,
      "attention_mask": attention_masks,
  }, label


def encode_examples(ds, limit=-1):
    # prepare list, so that we can build up final TensorFlow dataset from slices.
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []
    if limit > 0:
        ds = ds.take(limit)

    for index, row in ds.iterrows():
        review = row["text"]
        label = row["y"]
        bert_input = convert_example_to_feature(review)

        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])
    return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)


if __name__ == '__main__':
    test_bert_token()
