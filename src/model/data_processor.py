# -*- coding: utf-8 -*-
import os
import sys
import json
import numpy as np
import cPickle
from collections import defaultdict
from itertools import groupby   
from chainer import cuda
from tqdm import tqdm

class DataProcessor(object):

    def __init__(self, args):
        self.train_data_path = args.data_path+"data_en_{}_neg4/train.txt".format(args.doc_lang)
        self.dev_data_path = args.data_path+"data_en_{}_neg39/dev.txt".format(args.doc_lang)
        self.test_data_path = args.data_path+"data_en_{}_neg39/test1.txt".format(args.doc_lang)
        self.test = args.test # if true, use tiny datasets for quick test
        self.cnn_ksize = args.cnn_ksize
        self.sub_sample_train = args.sub_sample_train
        self.sub_sample_data_limit = args.sub_sample_data_limit
        self.extract_parameter = args.extract_parameter

        # Vocabulary for sentence pairs
        # word2vec vocabulary: vocab outside this will be considered as <unk>
        print("loading vocabulary...")
        self.word2vec_vocab_q = self.load_vocab(args.vocab_path+"/vocab_enwiki.txt", args.vocab_size)
        self.word2vec_vocab_d = self.load_vocab(args.vocab_path+"/vocab_"+args.doc_lang+"wiki.txt", args.vocab_size)
        print("done")

        if args.create_vocabulary:
            self.vocab_q = defaultdict(lambda: len(self.vocab_q))
            self.vocab_q["<pad>"]
            self.vocab_q["<unk>"]
            self.vocab_d = defaultdict(lambda: len(self.vocab_d))
            self.vocab_d["<pad>"]
            self.vocab_d["<unk>"]
        else:
            print('load vocab from cPickle')
            with open(args.vocab_path+"/en_{}_vocab_for_index.txt".format(args.doc_lang),"rb") as f_q,\
                 open(args.vocab_path+"/{}_vocab_for_index.txt".format(args.doc_lang),"rb") as f_d:
                self.vocab_q = cPickle.load(f_q)
                self.vocab_d = cPickle.load(f_d)
            print('done')

        self.device = args.gpu
        self.encode_type = args.encode_type

    def load_vocab(self, path, size):
        vocab = {}
        for i, w in enumerate(open(path, "r")):
            if i < size:
                vocab[w.strip()] = 1
        return vocab

    def prepare_dataset(self):
        # load train/dev/test data
        print("loading dataset...")
        self.train_data, _ = self.load_dataset_for_neg_sampled_data("train")
        self.dev_data, self.n_dev_qd_pairs   = self.load_dataset_for_neg_sampled_data("dev")
        self.test_data, self.n_test_qd_pairs  = self.load_dataset_for_neg_sampled_data("test")

        if self.test:
            print("tiny dataset for quick test...")
        print("done")

    def load_dataset_for_neg_sampled_data(self, _type):

        data_limit = 10
        if _type == "train":
            path = self.train_data_path
            data_limit = 32
            if self.sub_sample_train:
                data_limit = self.sub_sample_data_limit
            # data_limit = 20
        elif _type == "dev":
            path = self.dev_data_path
        elif _type == "test":
            path = self.test_data_path

        dataset = []

        if not(self.test):
            with open(path, "r") as input_data:
                total_line = len([0 for _ in  input_data])
        else:
            total_line = 0

        n_qd_pairs = []
        count = 0
        with open(path, "r") as input_data:
            # for i, line in tqdm(enumerate(input_data), total=total_line):
            for i, line in enumerate(input_data):
                rel, query, doc = line.strip().split("\t")

                if rel == str(2):
                    if i != 0:
                        n_qd_pairs.append(count)
                        count = 0
                    if self.test and len(dataset) >= data_limit:
                        break
                    if self.sub_sample_train and _type == 'train' and len(dataset) >= data_limit:
                        break
                    first_line_query = query
                    doc_rel = doc
                else:
                    count += 1
                    assert first_line_query == query
                    assert rel == str(0)
                    doc_nonrel = doc

                    # convert text data to index data as numpy array
                    if self.extract_parameter:
                        x1s = np.array([1,0], dtype=np.int32)
                        x2s = np.array([1,0], dtype=np.int32)
                        x3s = np.array([0,0], dtype=np.int32)
                    else:
                        arg1 = [self.vocab_q[token] if token in self.word2vec_vocab_q else self.vocab_q["<unk>"] for token in query.strip('.').split()]
                        arg2 = [self.vocab_d[token] if token in self.word2vec_vocab_d else self.vocab_d["<unk>"] for token in doc_rel.split()]
                        arg3 = [self.vocab_d[token] if token in self.word2vec_vocab_d else self.vocab_d["<unk>"] for token in doc_nonrel.split()]

                        if self.encode_type == "cnn":
                            if len(arg1) < self.cnn_ksize:
                                arg1 += [0 for _ in range(self.cnn_ksize-len(arg1))]
                            if len(arg2) < self.cnn_ksize:
                                arg2 += [0 for _ in range(self.cnn_ksize-len(arg2))]
                            if len(arg3) < self.cnn_ksize:
                                arg3 += [0 for _ in range(self.cnn_ksize-len(arg3))]

                        x1s = np.array(arg1, dtype=np.int32)
                        x2s = np.array(arg2, dtype=np.int32)
                        x3s = np.array(arg3, dtype=np.int32)

                    t = np.array([0], dtype=np.int32)

                    dataset.append((x1s, x2s, x3s, t))

            else:
                n_qd_pairs.append(count)

        return dataset, n_qd_pairs
