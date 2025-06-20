import numpy as np
import re
import gensim
import os


def clean_str(string):
    """Clean and normalize string, add spaces around special characters"""
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(dataset_type, positive_data_file=None, negative_data_file=None, data_split="train"):
    """Load dataset and process labels:
    - mr: load from positive_data_file/negative_data_file
    - sst1/sst2: load from official train/dev/test files (format: "label text")
    """
    if dataset_type == "mr":
        if not positive_data_file or not negative_data_file:
            raise ValueError("MR dataset requires both positive_data_file and negative_data_file")

        positive_examples = [s.strip() for s in open(positive_data_file, "r", encoding='latin-1').readlines()]
        negative_examples = [s.strip() for s in open(negative_data_file, "r", encoding='latin-1').readlines()]

        x_text = positive_examples + negative_examples
        x_text = [clean_str(sent) for sent in x_text]

        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        y = np.concatenate([positive_labels, negative_labels], 0)

    elif dataset_type in ["sst1", "sst2"]:
        data_dir = f"./data/{dataset_type}"
        file_path = os.path.join(data_dir, f"{data_split}.txt")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{dataset_type} {data_split} file not found: {file_path}")

        x_text, labels = [], []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    parts = line.split(" ", 1)
                if len(parts) != 2:
                    continue
                sent, label_str = parts
                try:
                    label = int(label_str)
                except ValueError:
                    continue
                x_text.append(clean_str(sent))
                labels.append(label)

        num_classes = 5 if dataset_type == "sst1" else 2
        y = np.eye(num_classes)[labels]

    else:
        raise ValueError(f"Unsupported dataset: {dataset_type}")

    return x_text, y


def load_word2vec(embedding_path, vocab, embedding_dim=300):
    """Load pre-trained word vectors (supports GloVe and Word2Vec formats)"""
    try:
        with open(embedding_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if len(first_line.split()) != 2:
                import tempfile
                import os

                vocab_size = sum(1 for _ in open(embedding_path, 'r', encoding='utf-8'))

                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as tmp:
                    tmp_path = tmp.name
                    tmp.write(f"{vocab_size} {embedding_dim}\n")
                    tmp.write(open(embedding_path, 'r', encoding='utf-8').read())

                word2vec = gensim.models.KeyedVectors.load_word2vec_format(
                    tmp_path, binary=False, limit=500000
                )

                os.unlink(tmp_path)
            else:
                word2vec = gensim.models.KeyedVectors.load_word2vec_format(
                    embedding_path, binary=False, limit=500000
                )

        if word2vec.vector_size != embedding_dim:
            raise ValueError(f"Word vector dimension mismatch! Model requires {embedding_dim}D, got {word2vec.vector_size}D")

        embedding_matrix = np.zeros((len(vocab) + 1, embedding_dim))
        for word, idx in vocab.items():
            if word in word2vec:
                embedding_matrix[idx] = word2vec[word]
        return embedding_matrix
    except Exception as e:
        print(f"[WARNING] Failed to load word vectors: {e}")
        return None


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """Generate batches of data"""
    x_data = np.array([d[0] for d in data])
    y_data = np.array([d[1] for d in data])
    data_size = len(x_data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1

    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            x_shuffled = x_data[shuffle_indices]
            y_shuffled = y_data[shuffle_indices]
        else:
            x_shuffled = x_data
            y_shuffled = y_data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield (x_shuffled[start_index:end_index], y_shuffled[start_index:end_index])
