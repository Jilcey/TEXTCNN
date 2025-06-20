import os
import argparse
import numpy as np
from wordcloud import WordCloud
import data_helpers
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

nltk.download('stopwords')
EN_STOPWORDS = set(stopwords.words('english'))


def generate_model_aware_wordcloud(dataset_type, positive_data_file=None, negative_data_file=None,
                                   save_path="wordcloud_optimized.png"):
    print(f"Generating model-aware word cloud (stopwords filtered): {dataset_type}")

    if dataset_type == "mr":
        if not positive_data_file or not negative_data_file:
            raise ValueError("MR dataset requires both positive_data_file and negative_data_file")
        x_text, _ = data_helpers.load_data_and_labels(
            dataset_type,
            positive_data_file=positive_data_file,
            negative_data_file=negative_data_file,
            data_split="train"
        )
    elif dataset_type in ["sst1", "sst2"]:
        x_text, _ = data_helpers.load_data_and_labels(dataset_type, data_split="train")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_type}")

    filtered_texts = []
    for text in x_text:
        cleaned = data_helpers.clean_str(text)
        words = [w for w in cleaned.split() if w not in EN_STOPWORDS]
        filtered_texts.append(" ".join(words))

    vocab = {}
    for text in filtered_texts:
        for word in text.split():
            vocab[word] = vocab.get(word, 0) + 1

    wordcloud = WordCloud(
        width=1000, height=600,
        background_color='white',
        max_words=200,
        contour_width=1,
        contour_color='steelblue',
        colormap='tab20b'
    ).generate_from_frequencies(vocab)

    wordcloud.to_file(save_path)
    print(f"Optimized word cloud saved to: {save_path}")

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate model-aware word cloud (stopwords filtered)")
    parser.add_argument("--dataset", type=str, default="mr", choices=["mr", "sst1", "sst2"],
                        help="Dataset type: mr/sst1/sst2")
    parser.add_argument("--positive_data_file", type=str, default="./data/rt-polaritydata/rt-polarity.pos",
                        help="Path to positive data file (required for MR dataset)")
    parser.add_argument("--negative_data_file", type=str, default="./data/rt-polaritydata/rt-polarity.neg",
                        help="Path to negative data file (required for MR dataset)")
    parser.add_argument("--output", type=str, default="wordcloud_optimized.png",
                        help="Output file path")
    args = parser.parse_args()

    if args.dataset == "mr" and (
            not os.path.exists(args.positive_data_file) or not os.path.exists(args.negative_data_file)):
        raise ValueError("For MR dataset, both positive_data_file and negative_data_file must exist.")

    generate_model_aware_wordcloud(
        args.dataset,
        args.positive_data_file,
        args.negative_data_file,
        args.output
    )
