import time

import tensorflow as tf
from absl import logging


class BatchTimestamp(object):
    """A structure to store batch time stamp."""

    def __init__(self, batch_index, timestamp):
        self.batch_index = batch_index
        self.timestamp = timestamp

    def __repr__(self):
        return "'BatchTimestamp<batch_index: {}, timestamp: {}>'".format(
            self.batch_index, self.timestamp)


class TimeHistory(tf.keras.callbacks.Callback):
    """Callback for Keras models."""

    def __init__(self, batch_size, log_steps, logdir=None):
        """Callback for logging performance.

        Args:
          batch_size: Total batch size.
          log_steps: Interval of steps between logging of batch level stats.
          logdir: Optional directory to write TensorBoard summaries.
        """
        # TODO(wcromar): remove this parameter and rely on `logs` parameter of
        # on_train_batch_end()
        self.batch_size = batch_size
        super(TimeHistory, self).__init__()
        self.log_steps = log_steps
        self.last_log_step = 0
        self.steps_before_epoch = 0
        self.steps_in_epoch = 0
        self.start_time = None

        if logdir:
            self.summary_writer = tf.summary.create_file_writer(logdir)
        else:
            self.summary_writer = None

        # Logs start of step 1 then end of each step based on log_steps interval.
        self.timestamp_log = []

        # Records the time each epoch takes to run from start to finish of epoch.
        self.epoch_runtime_log = []

    @property
    def global_steps(self):
        """The current 1-indexed global step."""
        return self.steps_before_epoch + self.steps_in_epoch

    @property
    def average_steps_per_second(self):
        """The average training steps per second across all epochs."""
        return self.global_steps / sum(self.epoch_runtime_log)

    @property
    def average_examples_per_second(self):
        """The average number of training examples per second across all epochs."""
        return self.average_steps_per_second * self.batch_size

    def on_train_end(self, logs=None):
        self.train_finish_time = time.time()

        if self.summary_writer:
            self.summary_writer.flush()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_batch_begin(self, batch, logs=None):
        if not self.start_time:
            self.start_time = time.time()

        # Record the timestamp of the first global step
        if not self.timestamp_log:
            self.timestamp_log.append(BatchTimestamp(self.global_steps,
                                                     self.start_time))

    def on_batch_end(self, batch, logs=None):
        """Records elapse time of the batch and calculates examples per second."""
        self.steps_in_epoch = batch + 1
        steps_since_last_log = self.global_steps - self.last_log_step
        if steps_since_last_log >= self.log_steps:
            now = time.time()
            elapsed_time = now - self.start_time
            steps_per_second = steps_since_last_log / elapsed_time
            examples_per_second = steps_per_second * self.batch_size

            self.timestamp_log.append(BatchTimestamp(self.global_steps, now))
            logging.info(
                'TimeHistory: %.2f seconds, %.2f examples/second between steps %d '
                'and %d', elapsed_time, examples_per_second, self.last_log_step,
                self.global_steps)

            if self.summary_writer:
                with self.summary_writer.as_default():
                    tf.summary.scalar('global_step/sec', steps_per_second,
                                      self.global_steps)
                    tf.summary.scalar('examples/sec', examples_per_second,
                                      self.global_steps)

            self.last_log_step = self.global_steps
            self.start_time = None

    def on_epoch_end(self, epoch, logs=None):
        epoch_run_time = time.time() - self.epoch_start
        self.epoch_runtime_log.append(epoch_run_time)

        self.steps_before_epoch += self.steps_in_epoch
        self.steps_in_epoch = 0
