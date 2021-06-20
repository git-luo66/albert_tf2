#encoding:utf-8

import tensorflow as tf
from model.optimization import LAMB, AdamWeightDecay, WarmUp

from model.albert import AlbertModel
import argparse
from utils.model_utils import metric_fn
parser = argparse.ArgumentParser()

parser.add_argument("--classifier_dropout", default=0.1, help="classification layer dropout")

args = parser.parse_args()


def get_model(albert_config, max_seq_length, num_labels):
    """Returns keras fuctional model"""
    float_type = tf.float32
    hidden_dropout_prob = args.classifier_dropout  # as per original code relased
    input_word_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
    input_mask = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
    input_type_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')

    albert_layer = AlbertModel(config=albert_config, float_type=float_type)

    pooled_output, _ = albert_layer(input_word_ids, input_mask, input_type_ids)

    initializer = tf.keras.initializers.TruncatedNormal(stddev=albert_config.initializer_range)

    output = tf.keras.layers.Dropout(rate=hidden_dropout_prob)(pooled_output)

    output = tf.keras.layers.Dense(
        num_labels,
        kernel_initializer=initializer,
        name='output',
        dtype=float_type)(
        output)
    predict = tf.keras.layers.Activation(tf.nn.softmax)(output)

    model = tf.keras.Model(
        inputs={
            'input_word_ids': input_word_ids,
            'input_mask': input_mask,
            'input_type_ids': input_type_ids
        },
        outputs=predict)

    return model
