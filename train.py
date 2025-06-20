#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
import pickle
from absl import app, flags
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

flags.DEFINE_float("dev_sample_percentage", 0.1, "Validation split ratio (from train set)")
flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Original MR positive data")
flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Original MR negative data")
flags.DEFINE_string("word2vec_path", "", "Path to GoogleNews-vectors-negative300.bin")
flags.DEFINE_string("dataset", "mr", "Dataset: mr, sst1, sst2")

flags.DEFINE_integer("embedding_dim", 300, "Embedding dimension")
flags.DEFINE_string("filter_sizes", "3,4,5", "Filter sizes")
flags.DEFINE_integer("num_filters", 300, "Filters per size")
flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability")
flags.DEFINE_float("l2_reg_lambda", 0.001, "L2 regularization")
flags.DEFINE_float("learning_rate", 1.0, "Learning rate for Adadelta optimizer")

flags.DEFINE_integer("batch_size", 50, "Batch size")
flags.DEFINE_integer("num_epochs", 200, "Maximum training epochs")
flags.DEFINE_integer("evaluate_every", 100, "Evaluate every steps")
flags.DEFINE_integer("checkpoint_every", 100, "Checkpoint every steps")
flags.DEFINE_integer("num_checkpoints", 5, "Max checkpoints to keep")
flags.DEFINE_boolean("multichannel", False, "Use multichannel architecture")
flags.DEFINE_string("model_variant", "CNN-rand", "Model variant: CNN-rand/CNN-static/CNN-non-static/CNN-multichannel")
flags.DEFINE_boolean("use_attention", False, "Enable attention mechanism")
flags.DEFINE_boolean("use_dilated", False, "Enable dilated convolution")
flags.DEFINE_integer("dilation_rate", 1, "Dilation rate for dilated convolution")
flags.DEFINE_integer("early_stop_patience", 12, "Early stopping patience")

FLAGS = flags.FLAGS


def preprocess():
    """Data preprocessing: load data, build vocabulary, split train/validation sets"""
    print("Loading data...")
    dataset_type = FLAGS.dataset

    if dataset_type == "mr":
        original_pos = FLAGS.positive_data_file
        original_neg = FLAGS.negative_data_file
        if not os.path.exists(original_pos) or not os.path.exists(original_neg):
            raise FileNotFoundError("MR original data not found: rt-polarity.pos/neg")

        pos_all = [s.strip() for s in open(original_pos, "r", encoding='latin-1').readlines()]
        neg_all = [s.strip() for s in open(original_neg, "r", encoding='latin-1').readlines()]
        all_examples = pos_all + neg_all
        all_labels = [1] * len(pos_all) + [0] * len(neg_all)

        np.random.seed(42)
        shuffle_indices = np.random.permutation(len(all_examples))
        shuffled_examples = [all_examples[i] for i in shuffle_indices]
        shuffled_labels = [all_labels[i] for i in shuffle_indices]

        test_size = int(0.1 * len(shuffled_examples))
        test_examples = shuffled_examples[:test_size]
        test_labels = shuffled_labels[:test_size]
        train_examples = shuffled_examples[test_size:]
        train_labels = shuffled_labels[test_size:]

        data_dir = "./data/rt-polaritydata"
        os.makedirs(data_dir, exist_ok=True)

        train_pos = [text for text, label in zip(train_examples, train_labels) if label == 1]
        train_neg = [text for text, label in zip(train_examples, train_labels) if label == 0]
        with open(os.path.join(data_dir, "train.pos"), "w", encoding='latin-1') as f:
            f.write("\n".join(train_pos))
        with open(os.path.join(data_dir, "train.neg"), "w", encoding='latin-1') as f:
            f.write("\n".join(train_neg))

        test_pos = [text for text, label in zip(test_examples, test_labels) if label == 1]
        test_neg = [text for text, label in zip(test_examples, test_labels) if label == 0]
        with open(os.path.join(data_dir, "test.pos"), "w", encoding='latin-1') as f:
            f.write("\n".join(test_pos))
        with open(os.path.join(data_dir, "test.neg"), "w", encoding='latin-1') as f:
            f.write("\n".join(test_neg))

        x_text, y = data_helpers.load_data_and_labels(
            dataset_type,
            positive_data_file=os.path.join(data_dir, "train.pos"),
            negative_data_file=os.path.join(data_dir, "train.neg"),
            data_split="train"
        )

    elif dataset_type in ["sst1", "sst2"]:
        x_train_text, y_train = data_helpers.load_data_and_labels(
            dataset_type, data_split="train"
        )
        x_dev_text, y_dev = data_helpers.load_data_and_labels(
            dataset_type, data_split="dev"
        )
        x_text = x_train_text + x_dev_text
        y = np.concatenate([y_train, y_dev], axis=0)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_type}")

    max_document_length = max([len(x.split()) for x in x_text])
    vocab = {}
    for text in x_text:
        for word in text.split():
            vocab[word] = vocab.get(word, 0) + 1
    sorted_vocab = sorted(vocab.items(), key=lambda x: (-x[1], x[0]))
    word_to_idx = {word: i + 1 for i, (word, _) in enumerate(sorted_vocab)}
    vocab_size = len(word_to_idx) + 1

    x = []
    for text in x_text:
        seq = [word_to_idx[word] if word in word_to_idx else 0 for word in text.split()]
        seq = seq[:max_document_length] if len(seq) > max_document_length else seq + [0] * (
                    max_document_length - len(seq))
        x.append(seq)
    x = np.array(x)

    if dataset_type == "mr":
        np.random.seed(10)
        shuffle_indices = np.random.permutation(len(y))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]

        dev_split = -int(FLAGS.dev_sample_percentage * len(y))
        x_train, x_dev = x_shuffled[:dev_split], x_shuffled[dev_split:]
        y_train, y_dev = y_shuffled[:dev_split], y_shuffled[dev_split:]
    else:
        x_train = []
        for text in x_train_text:
            seq = [word_to_idx[word] if word in word_to_idx else 0 for word in text.split()]
            seq = seq[:max_document_length] if len(seq) > max_document_length else seq + [0] * (
                        max_document_length - len(seq))
            x_train.append(seq)
        x_train = np.array(x_train)

        x_dev = []
        for text in x_dev_text:
            seq = [word_to_idx[word] if word in word_to_idx else 0 for word in text.split()]
            seq = seq[:max_document_length] if len(seq) > max_document_length else seq + [0] * (
                        max_document_length - len(seq))
            x_dev.append(seq)
        x_dev = np.array(x_dev)

    embedding_matrix = None
    if FLAGS.model_variant != "CNN-rand" and FLAGS.word2vec_path:
        print("Loading word2vec...")
        word2vec = data_helpers.load_word2vec(FLAGS.word2vec_path, word_to_idx)
        if word2vec is not None:
            if FLAGS.multichannel or FLAGS.model_variant == "CNN-multichannel":
                embedding_matrix = np.stack([word2vec] * 2)
            else:
                embedding_matrix = word2vec

    with open("vocab.pkl", "wb") as f:
        pickle.dump((word_to_idx, max_document_length), f)

    print(f"Vocabulary Size: {vocab_size}")
    print(f"Train/Dev Split: {len(y_train)}/{len(y_dev)}")

    return x_train, y_train, x_dev, y_dev, vocab_size, max_document_length, embedding_matrix


def train(x_train, y_train, x_dev, y_dev, vocab_size, max_doc_len, embedding_matrix):
    """Main model training function"""
    filter_sizes = list(map(int, FLAGS.filter_sizes.split(",")))
    num_classes = y_train.shape[1]
    multichannel = FLAGS.multichannel or FLAGS.model_variant == "CNN-multichannel"
    trainable = FLAGS.model_variant in ["CNN-non-static", "CNN-multichannel"]

    input_x, input_y, dropout_keep_prob, predictions, loss, accuracy = TextCNN(
        sequence_length=max_doc_len,
        num_classes=num_classes,
        vocab_size=vocab_size,
        embedding_size=FLAGS.embedding_dim,
        filter_sizes=filter_sizes,
        num_filters=FLAGS.num_filters,
        l2_reg_lambda=FLAGS.l2_reg_lambda,
        embedding_matrix=embedding_matrix,
        trainable=trainable,
        multichannel=multichannel,
        use_attention=FLAGS.use_attention,
        use_dilated=FLAGS.use_dilated,
        dilation_rate=FLAGS.dilation_rate
    )

    global_step = tf.compat.v1.Variable(0, name="global_step", trainable=False)
    optimizer = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=FLAGS.learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    train_log = os.path.join(log_dir, "train.log")
    val_log = os.path.join(log_dir, "val.log")

    checkpoint_dir = os.path.abspath(os.path.join("runs", timestamp))
    os.makedirs(checkpoint_dir, exist_ok=True)
    saver = tf.compat.v1.train.Saver(max_to_keep=FLAGS.num_checkpoints)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        val_steps = []
        best_val_acc = 0.0
        early_stop_counter = 0

        batches_per_epoch = int(np.ceil(len(x_train) / FLAGS.batch_size))
        completed_epochs = 0
        epoch_times = []
        epoch_start_time = time.time()

        def train_step(x_batch, y_batch):
            feed_dict = {
                input_x: x_batch,
                input_y: y_batch,
                dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, loss_val, acc_val = sess.run(
                [train_op, global_step, loss, accuracy],
                feed_dict=feed_dict
            )
            train_losses.append(loss_val)
            train_accs.append(acc_val)
            print(f"Step {step}: Train Loss={loss_val:.4f}, Acc={acc_val:.4f}")
            with open(train_log, "a") as f:
                f.write(f"{step},{loss_val},{acc_val}\n")

        def val_step(x_batch, y_batch):
            feed_dict = {
                input_x: x_batch,
                input_y: y_batch,
                dropout_keep_prob: 1.0
            }
            step, loss_val, acc_val = sess.run(
                [global_step, loss, accuracy],
                feed_dict=feed_dict
            )
            val_steps.append(step)
            val_losses.append(loss_val)
            val_accs.append(acc_val)
            print(f"Step {step}: Val Loss={loss_val:.4f}, Acc={acc_val:.4f}")
            with open(val_log, "a") as f:
                f.write(f"{step},{loss_val},{acc_val}\n")
            return acc_val

        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)),
            FLAGS.batch_size,
            FLAGS.num_epochs
        )

        for i, (x_batch, y_batch) in enumerate(batches):
            train_step(x_batch, y_batch)
            current_step = sess.run(global_step)

            if (i + 1) % batches_per_epoch == 0:
                completed_epochs += 1
                epoch_end_time = time.time()
                epoch_time = epoch_end_time - epoch_start_time
                epoch_times.append(epoch_time)
                print(f"Epoch {completed_epochs} training time: {epoch_time:.4f} seconds")
                epoch_start_time = time.time()

            if current_step % FLAGS.evaluate_every == 0:
                print("\nValidation:")
                val_acc = val_step(x_dev, y_dev)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    early_stop_counter = 0
                    saver.save(sess, os.path.join(checkpoint_dir, "model"), global_step=current_step)
                    print(f"[BEST MODEL] Validation accuracy improved to {best_val_acc:.4f}")
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= FLAGS.early_stop_patience:
                        print(f"[EARLY STOP] No improvement for {FLAGS.early_stop_patience} steps, terminating training")
                        if epoch_start_time != 0:
                            partial_epoch_time = time.time() - epoch_start_time
                            print(
                                f"Partial epoch {completed_epochs + 1} training time: {partial_epoch_time:.4f} seconds")
                        break
                print("")

            if current_step % FLAGS.checkpoint_every == 0:
                saver.save(sess, os.path.join(checkpoint_dir, "model"), global_step=current_step)

        if epoch_start_time != 0 and (i + 1) % batches_per_epoch != 0:
            partial_epoch_time = time.time() - epoch_start_time
            print(f"Partial epoch {completed_epochs + 1} training time: {partial_epoch_time:.4f} seconds")

        if epoch_times:
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            print(f"\n[TIME STATISTICS]")
            print(f"Completed epochs: {completed_epochs}")
            print(f"Average epoch training time: {avg_epoch_time:.4f} seconds")
            print(f"Estimated total training time: {avg_epoch_time * completed_epochs:.4f} seconds")
        else:
            print("[TIME STATISTICS] No complete epochs to calculate average time.")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
    plt.plot(val_steps, val_losses, 'o-', label='Val Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(len(train_accs)), train_accs, label='Train Acc')
    plt.plot(val_steps, val_accs, 'o-', label='Val Acc')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(log_dir, "training_curves.png")
    plt.savefig(plot_path)
    print(f"Training curves saved to {plot_path}")
    print(f"[FINAL RESULT] Best validation accuracy: {best_val_acc:.4f}")
    plt.close()


def main(_):
    print("\nParameters:")
    for attr, value in sorted(FLAGS.flag_values_dict().items()):
        print(f"{attr.upper()}: {value}")
    print("")

    x_train, y_train, x_dev, y_dev, vocab_size, max_doc_len, embedding_matrix = preprocess()
    train(x_train, y_train, x_dev, y_dev, vocab_size, max_doc_len, embedding_matrix)


if __name__ == '__main__':
    app.run(main)
