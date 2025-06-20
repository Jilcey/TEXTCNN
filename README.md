# [An Improvement Study on TextCNN Integrated with Dilated Convolution and Attention Mechanism for Text Classification]

This project is an improvement based on the [cnn-text-classification-tf](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/) repository. The original repository is licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).

This project is solely for the purpose of a course paper and is not for commercial use.

## Original Repository Information
- Original repository link: [cnn-text-classification-tf](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
- Original repository license: [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

This is a re-implementation and enhancement of the framework proposed in Kim's paper [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882), using TensorFlow. 

## Requirements

- Python 3.12
- Tensorflow == 2.18.0
- Numpy == 1.26.0
- gensim == 4.3.3
- Matplotlib == 3.10.0
- scikit-learn == 1.6.0

## Training

Print parameters:

```bash
./train.py --help
```
GloVe pre-trained word vectors can be downloaded from https://github.com/stanfordnlp/GloVe and placed in the .data/embedding/ directory during runtime. When training the model, the word vector input dimension must be consistent with the pre-trained word vector dimension. For example, if 300-dimensional GloVe is used, the model embedding layer dimension must be set to 300.

```
optional arguments:
  -h, --help            show this help message and exit
  --embedding_dim EMBEDDING_DIM
                        Dimensionality of character embedding (default: 300)
  --filter_sizes FILTER_SIZES
                        Comma-separated filter sizes (default: '3,4,5')
  --num_filters NUM_FILTERS
                        Number of filters per filter size (default: 128)
  --l2_reg_lambda L2_REG_LAMBDA
                        L2 regularizaion lambda (default: 0.0)
  --dropout_keep_prob DROPOUT_KEEP_PROB
                        Dropout keep probability (default: 0.5)
  --batch_size BATCH_SIZE
                        Batch Size (default: 64)
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 100)
  --evaluate_every EVALUATE_EVERY
                        Evaluate model on dev set after this many steps
                        (default: 100)
  --checkpoint_every CHECKPOINT_EVERY
                        Save model after this many steps (default: 100)
  --allow_soft_placement ALLOW_SOFT_PLACEMENT
                        Allow device soft device placement
  --noallow_soft_placement
  --log_device_placement LOG_DEVICE_PLACEMENT
                        Log placement of ops on devices
  --nolog_device_placement
  --use_dilated DILATED             Use dilated convolution (default: False)
  --dilation_rate DILATION_RATE     Dilation rate (default: 1)
  --use_attention ATTENTION         Use attention mechanism (default: False)
  --attention_dim ATTENTION_DIM     Attention layer dimension (default: 64)
  --model_variant VARIANT           Model variant (default: CNN-rand)
  --dataset DATASET                 Dataset: mr/sst1/sst2 (default: mr)
  --data_dir DATA_DIR               Data directory (default: ./data)
  --word2vec_path PATH              Pre-trained word vectors path (default: none)
  --use_pretrained PRETRAINED       Use pre-trained vectors (default: False)
  --learning_rate LR                Learning rate (default: 1)
  --early_stop_patience PATIENCE    Early stopping patience (default: 12)
```

To give a specific example, if you want to use the CNN-static variant and use both dilated convolution and attention mechanisms, using the SST2 dataset, you can use the following script:
```bash
python train.py --dataset=sst2 --model_variant=CNN-static --word2vec_path=./data/embedding/glove.6B.300d.txt --embedding_dim=300 --use_attention True --use_dilated True --dilation_rate 2
```
Other situations are similar.

## Evaluating
Using the same example above:
```bash
python eval.py --checkpoint_dir=runs/20250610-211022 --eval_test=True --dataset=sst2
```
Replace the checkpoint dir with the output from the training. To use your own data, change the `eval.py` script to load your data.

## Generate database word cloud
You can generate a word cloud for a data set, taking the mr data set as an example:
```bash
python word.py --dataset mr
```

## References
See the original paper for details.
