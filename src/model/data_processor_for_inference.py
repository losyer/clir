# -*- coding: utf-8 -*-
import numpy as np
import cPickle
from collections import defaultdict
import logging
formatter = '%(levelname)s : %(asctime)s : %(message)s'
logging.basicConfig(level=logging.DEBUG, format=formatter)

class DataProcessor_for_inference(object):

    def __init__(self, args):
        self.test_data_path = args.data_path
        self.test = args.test # if true, use tiny datasets for quick test

        # Vocabulary for sentence pairs
        # word2vec vocabulary: vocab outside this will be considered as <unk>
        logging.info("loading w2v vocabulary ...")
        self.word2vec_vocab_q = self.load_vocab(args.vocab_path+"vocab_enwiki.txt", args.vocab_size)
        self.word2vec_vocab_d = self.load_vocab(args.vocab_path+"vocab_"+args.doc_lang+"wiki.txt", args.vocab_size)
        logging.info("loading w2v vocabulary ... done")

        logging.info("loading vocabulary from cPickle dump ...")
        with open(args.vocab_path+"en_{}_vocab_for_index.txt".format(args.doc_lang),"rb") as f_q,\
             open(args.vocab_path+"{}_vocab_for_index.txt".format(args.doc_lang),"rb") as f_d:
            self.vocab_q = cPickle.load(f_q)
            self.vocab_d = cPickle.load(f_d)
        logging.info("loading vocabulary from cPickle dump ... dump")

        self.device = args.gpu
        self.encode_type = args.encode_type

    def load_vocab(self, path, size):
        return {w.strip():1 for i, w in enumerate(open(path, "r")) if i < size} 

    def prepare_dataset(self):
        logging.info("loading dataset ...")
        self.test_data = self.load_dataset("test")

        if self.test:
            logging.info("tiny dataset for quick test...")
        logging.info("loading dataset ... done")

    def load_dataset(self, _type):
        dataset = []

        with open(self.test_data_path, "r") as input_data:
            for i, line in enumerate(input_data):
                rel, query, doc = line.strip().split("\t")

                # convert text data to index data as numpy array
                arg1, arg2 = [], []
                for token in query.strip('.').split():
                    if token in self.word2vec_vocab_q and token in self.vocab_q:
                        arg1.append(self.vocab_q[token])
                    else:
                        arg1.append(self.vocab_q["<unk>"])
                        
                for token in doc.split():
                    if token in self.word2vec_vocab_d and token in self.vocab_d:
                        arg2.append(self.vocab_d[token])
                    else:
                        arg2.append(self.vocab_d["<unk>"])

                # each query and document must have enough length (4 at least) when you use CNN
                if self.encode_type == "cnn":
                    if len(arg1) < 4:
                        arg1 += [0 for _ in range(4-len(arg1))]
                    if len(arg2) < 4:
                        arg2 += [0 for _ in range(4-len(arg2))]

                x1s = np.array(arg1, dtype=np.int32)
                x2s = np.array(arg2, dtype=np.int32)
                t = np.array([0], dtype=np.int32)

                dataset.append((x1s, x2s, t))
                if self.test and len(dataset) >= 100:
                    break
                    
        return dataset
