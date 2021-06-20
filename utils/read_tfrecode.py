#encoding:utf-8

import tensorflow as tf

def decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
        example[name] = t

    return example

def single_file_dataset(input_file, name_to_features):

    d = tf.data.TFRecordDataset(input_file)
    d = d.map(lambda record: decode_record(record, name_to_features))

    if isinstance(input_file, str) or len(input_file) == 1:
        options = tf.data.Options()
        # options.experimental_distribute.auto_shard = False
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        d = d.with_options(options)
    return d

def read_dataset(file_path,label_list,seq_length,
                              batch_size,
                              is_training=True):
    """Creates input dataset from (tf)records files for train/eval."""
    name_to_features = {
        'input_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
        'input_mask': tf.io.FixedLenFeature([seq_length], tf.int64),
        'segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
        'label_ids': tf.io.FixedLenFeature([len(label_list)], tf.int64),
        'is_real_example': tf.io.FixedLenFeature([], tf.int64),
    }
    dataset = single_file_dataset(file_path, name_to_features)


    def _select_data_from_record(record):
        x = {
            'input_word_ids': record['input_ids'],
            'input_mask': record['input_mask'],
            'input_type_ids': record['segment_ids']
        }
        y = record['label_ids']
        # y = tf.one_hot(tf.cast(y, dtype=tf.int32), depth=num_classes, dtype=tf.float32)
        return (x, y)

    dataset = dataset.map(_select_data_from_record)
    if is_training:
        dataset = dataset.shuffle(100)
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(1024)
    return dataset
