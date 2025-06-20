#! /usr/bin/env python

import datetime
import tensorflow as tf
import numpy as np
import os
import csv
import pickle
import data_helpers
from absl import app, flags

flags.DEFINE_boolean("eval_test", False, "Evaluate on test data")
flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory")
flags.DEFINE_string("dataset", "mr", "Dataset: mr, sst1, sst2")
flags.DEFINE_integer("batch_size", 50, "Batch size")
flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/test.pos", "MR dataset positive examples")
flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/test.neg", "MR dataset negative examples")

FLAGS = flags.FLAGS


def main(_):
    print("\nParameters:")
    for attr, value in sorted(FLAGS.flag_values_dict().items()):
        print(f"{attr.upper()}: {value}")
    print("")

    # Load vocabulary and sequence length
    with open("vocab.pkl", "rb") as f:
        word_to_idx, max_document_length = pickle.load(f)

    # Load model
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if not checkpoint_file:
        raise ValueError(f"Checkpoint not found in {FLAGS.checkpoint_dir}")

    graph = tf.compat.v1.Graph()
    with graph.as_default():
        session_conf = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        with tf.compat.v1.Session(config=session_conf) as sess:
            # Load model graph
            saver = tf.compat.v1.train.import_meta_graph(f"{checkpoint_file}.meta")
            saver.restore(sess, checkpoint_file)

            # Get input placeholders and prediction operation
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Load test data
            if FLAGS.eval_test:
                x_raw, y_all = data_helpers.load_data_and_labels(
                    dataset_type=FLAGS.dataset,
                    positive_data_file=FLAGS.positive_data_file,
                    negative_data_file=FLAGS.negative_data_file,
                    data_split="test"
                )
                y_test = np.argmax(y_all, axis=1)
            else:
                # Default examples (when no test set is available)
                x_raw = ["a masterpiece four years in the making", "everything is off."]
                y_test = [1, 0]

            # Convert text to sequences (padding/truncating)
            x_test = []
            for text in x_raw:
                seq = [word_to_idx[word] if word in word_to_idx else 0 for word in text.split()]
                if len(seq) > max_document_length:
                    seq = seq[:max_document_length]
                else:
                    seq += [0] * (max_document_length - len(seq))
                x_test.append(seq)
            x_test = np.array(x_test)
            print(f"Input data shape: {x_test.shape}")

            # Batch prediction
            batches = data_helpers.batch_iter(
                list(zip(x_test, [[]] * len(x_test))),
                FLAGS.batch_size,
                1,
                shuffle=False
            )
            all_predictions = []
            for x_batch, _ in batches:
                batch_pred = sess.run(predictions, {input_x: x_batch, dropout_keep_prob: 1.0})
                all_predictions.extend(np.argmax(batch_pred, axis=1))

            # Calculate accuracy (if labels are available)
            if y_test is not None:
                correct = np.sum(np.array(all_predictions) == y_test)
                print(f"Total examples: {len(y_test)}, Accuracy: {correct / len(y_test):.4f}")

            # Save predictions
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            out_dir = os.path.join(FLAGS.checkpoint_dir, "..", "predictions", timestamp)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "test_prediction.csv")
            with open(out_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Text", "Prediction"])
                for text, pred in zip(x_raw, all_predictions):
                    writer.writerow([text, pred])
            print(f"Predictions saved to {out_path}")


if __name__ == '__main__':
    app.run(main)