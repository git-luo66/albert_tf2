#encoding:utf-8

import os
from utils import tokenization
import tensorflow as tf
from model.albert import AlbertConfig
import numpy as np
from model.albert_model import get_model
from utils.Dataprocess import data, file_based_convert_examples_to_features
from utils.read_tfrecode import read_dataset

labels = [c[1:].strip() for c in open(os.path.join("./data", "label.txt")).readlines()]
class InputExample(object):
    def __init__(self, guid, text_a, label=None):
        self.guid = guid
        self.text_a = text_a
        self.label = label

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id, is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example

def convert_single_example(example, label_list, max_seq_length, tokenizer):
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)

    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]
    feature = InputFeatures(
        input_ids=input_ids, input_mask=input_mask,
        segment_ids=segment_ids, label_id=label_id,
        is_real_example=True)
    return feature

def convert_example(example, label_list, max_seq_length, tokenizer):
    if isinstance(example,list):
        features = []
        for (ex_index, example) in enumerate(example):
            feature = convert_single_example(example, label_list,max_seq_length, tokenizer)
            features.append(feature)
    return features

def process_data(sentence):
    tokenizer = tokenization.FullTokenizer(vocab_file='./pre_model/vocab.txt',
                                           do_lower_case=True)
    if isinstance(sentence, list):
        exam = [InputExample(guid="", text_a=x,  label='辱骂') for x in sentence]
    else:
        exam = [InputExample(guid="", text_a=sentence, label='辱骂')]
    features = convert_example(exam, labels, 128, tokenizer)

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)

    return np.array(all_input_ids), np.array(all_input_mask), np.array(all_segment_ids)


def predict(checkpoint_path, data_path):
    with open(data_path, "r+") as f:
        list_data = f.readlines()
        data = []
        true_label = []
        for l in list_data:
            # print(l)
            tmp_data = l.replace("\n", "")
            tmp = tmp_data.split('     ')
            data.append(tmp[0])
            b = int(tmp[1])
            true_label.append(b)

    all_input_ids, all_input_mask, all_segment_ids = process_data(data)
    # print(len(all_input_ids), len(all_segment_ids), len(all_input_mask),len(labels))

    inputs={}
    inputs["input_word_ids"] = all_input_ids
    inputs["input_mask"] = all_input_mask
    inputs["input_type_ids"] = all_segment_ids

    albert_config = AlbertConfig.from_json_file("./pre_model/albert_config.json")
    num_labels = len(labels)
    cls = get_model(
        albert_config=albert_config,
        max_seq_length=128,
        num_labels=num_labels,
    )

    print('model loaded.')
    latest_ckpt = tf.train.latest_checkpoint(checkpoint_path)
    print(latest_ckpt)

    cls.load_weights(latest_ckpt)
    predict = cls.predict(inputs)

    # for k, one in enumerate(predict):
    #     label = labels[int(np.argmax(one))]
    #
    #     print(data[k]+"    "+label)

    # print("预测值 = ",predict)

    true_num = 0
    k = 0
    for p_logit, t_label in zip(predict, true_label):
        pre_label = int(np.argmax(p_logit))

        if pre_label == t_label:
            true_num += 1
            label = labels[int(np.argmax(p_logit))]
            print(data[k] + "    " + label)
        k += 1
    impove = true_num/len(data)
    print("样本总数：", len(data))
    print("预测正确的数量：", true_num)
    print("预测准确率：", impove)



if __name__=="__main__":

    data_path = "./test_data/cj_test.txt"
    predict('./output_model', data_path)
