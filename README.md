# Cross-lingual Learning-to-Rank
- This repository is about *Cross-lingual Learning-to-Rank with Shared Representations. Shota Sasaki, Shuo Sun, Shigehiko Schamoni, Kevin Duh and Kentaro Inui. NAACL2018*
  - https://www.aclweb.org/anthology/N18-2073/

## Table of contents
  - [Usage](#usage)
    - [Requirements](#requirements)
    - [How to inference](#how-to-inference)
  - [Resource](#resource)


## Usage

### Requirements
- Python ver. 2.7
- chainer==5.4.0
- numpy==1.15.0

### How to inference
```
$ python src/inference.py \
--vocab_path [path_to_vocaburaly_directory] \
--model_path [path_to_model_snapshot] \
--data_path [path_to_target_data] \
--doc_lang [document_language] \
--n_hdim [for_feed_forward_layer]
```

For example,
```
$ cd resource
$ wget https://github.com/losyer/clir/releases/download/results_v1.0/sw.tar.gz
$ tar -zxvf sw.tar.gz
$ wget https://github.com/losyer/clir/releases/download/vocab_v1.0/vocabulary.zip
$ unzip vocabulary.zip
$ cd ..

$ python src/inference.py \
--vocab_path resource/vocabulary \
--model_path resource/sw/in_language/deep_model_100/model_epoch_1 \
--data_path resource/input_examples/train_l100.txt \
--doc_lang sw \
--n_hdim 100
```

## Resource
See https://github.com/losyer/clir/tree/master/resource

  

