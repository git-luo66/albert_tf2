# encoding:utf-8
import os
import numpy as np
from tensorflow.keras import backend as K

import tensorflow as tf

from model import optimization, performance, keras_utils


def get_optimizer(initial_lr, steps_per_epoch, epochs, warmup_steps, use_float16=False):
    optimizer = optimization.create_optimizer(initial_lr, steps_per_epoch * epochs, warmup_steps)
    optimizer = performance.configure_optimizer(
        optimizer,
        use_float16=use_float16,
        use_graph_rewrite=False)
    return optimizer


def get_loss_fn():
    """Gets the classification loss function."""

    def classification_loss_fn(labels, logits):
        """Classification loss."""
        K.print_tensor(labels, message=',y_true = ')
        K.print_tensor(logits, message=',y_predict = ')
        # labels = tf.squeeze(labels)
        # log_probs = tf.nn.log_softmax(logits, axis=-1)
        log_probs = tf.math.log(logits)
        K.print_tensor(log_probs, message=',y_log = ')
        # one_hot_labels = tf.one_hot(
        #     tf.cast(labels, dtype=tf.int32), depth=num_classes, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(
            tf.cast(labels, dtype=tf.float32) * tf.cast(log_probs, tf.float32), axis=-1)
        return tf.reduce_mean(per_example_loss)

    return classification_loss_fn


def metric_fn():
    return [
            tf.keras.metrics.Recall(name='recall', dtype=tf.float32),
            tf.keras.metrics.CategoricalAccuracy(name="Accuracy", dtype=tf.float32),
            tf.keras.metrics.Precision(name="Precision", dtype=tf.float32),
            ]

def get_callbacks(train_batch_size, log_steps, model_dir):
    custom_callback = keras_utils.TimeHistory(
        batch_size=train_batch_size,
        log_steps=log_steps,
        logdir=os.path.join(model_dir, 'logs'))

    summary_callback = tf.keras.callbacks.TensorBoard(os.path.join(model_dir, 'graph'), update_freq='batch')

    checkpoint_path = os.path.join(model_dir, 'checkpoint-{epoch:02d}')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, monitor='Accuracy', save_weights_only=True)

    return [custom_callback, summary_callback, checkpoint_callback]
