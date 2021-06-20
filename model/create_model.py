#encoding:utf-8

import tensorflow as tf
from model.optimization import LAMB, AdamWeightDecay, WarmUp

from model.albert import AlbertModel
import argparse
from utils.model_utils import metric_fn
parser = argparse.ArgumentParser()

parser.add_argument("--classifier_dropout", default=0.1, help="classification layer dropout")
parser.add_argument("--optimizer", default="AdamW", help="Optimizer for training LAMB/AdamW")
parser.add_argument("--weight_decay", default=0.01, help= "weight_decay")
parser.add_argument("--adam_epsilon", default=1e-6, help="adam_epsilon")
parser.add_argument("--task_name", default="scc", help="The name of the task to train.")
parser.add_argument("--output_dir", help="the path of output",
                    default="./output_model", type=str)
parser.add_argument("--init_checkpoint", help="the pretrain model path",
                    default=None, type=str)

args = parser.parse_args()



def get_model(albert_config, max_seq_length, num_labels, init_checkpoint, lr_rate,
              num_train_steps, num_warmup_steps):
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


    if args.init_checkpoint:
        albert_model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids],
                                      outputs=[pooled_output])
        albert_model.load_weights(init_checkpoint)
        print("the pretrain model resumed success")
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

    learning_rate = tf.constant(value=lr_rate, shape=[], dtype=tf.float32)
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=learning_rate,
                                                                     decay_steps=num_train_steps, end_learning_rate=0.0)
    if num_warmup_steps:
        learning_rate_fn = WarmUp(initial_learning_rate=learning_rate,
                                  decay_schedule_fn=learning_rate_fn,
                                  warmup_steps=num_warmup_steps)
    if args.optimizer == "LAMB":
        optimizer_fn = LAMB
    else:
        optimizer_fn = AdamWeightDecay

    optimizer = optimizer_fn(
        learning_rate=learning_rate_fn,
        weight_decay_rate=args.weight_decay,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=args.adam_epsilon,
        exclude_from_weight_decay=['layer_norm', 'bias'])

    if args.task_name.lower() == 'sts':
        loss_fct = tf.keras.losses.MeanSquaredError()
        model.compile(optimizer=optimizer, loss=loss_fct, metrics=metric_fn())
    else:
        loss_fct = tf.keras.losses.CategoricalCrossentropy()
        model.compile(optimizer=optimizer, loss=loss_fct, metrics=metric_fn())

    return model




