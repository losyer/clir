# -*- coding: utf-8 -*-
import chainer
import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from chainer import Variable as V
from util import cos_sim
from my_func import EmbedID_minus_pad
RS = np.random.RandomState(0)

def embed_seq_batch(embed, seq_batch, dropout=0.):
    xs_f = []
    for x in seq_batch:
        x = embed(x)
        x.unchain_backward()
        x = F.dropout(x, ratio=dropout) 
        xs_f.append(x)
    return xs_f

class Network_deep_for_inference(Chain):
    def __init__(self, args, n_vocab_q, n_vocab_d, train=True):
        self.train = train
        self.n_layer = args.n_layer
        self.n_hdim = args.n_hdim
        self.embed_dim = args.embed_dim
        self.n_vocab_q = n_vocab_q
        self.n_vocab_d = n_vocab_d
        self.encode_type = args.encode_type
        self.weighted_sum = args.weighted_sum
        self.cnn_out_channels = args.cnn_out_channels
        self.cnn_ksize = args.cnn_ksize
        self.device = args.gpu

        super(Network_deep_for_inference, self).__init__(
            embed_q=L.EmbedID(self.n_vocab_q, self.embed_dim, initialW=RS.normal(scale=0.5,size=(n_vocab_q, self.embed_dim)), ignore_label=0),
            embed_d=L.EmbedID(self.n_vocab_d, self.embed_dim, initialW=RS.normal(scale=0.5,size=(n_vocab_d, self.embed_dim)), ignore_label=0),
            term_weight_q=EmbedID_minus_pad(self.n_vocab_q, 1, initialW=RS.normal(scale=0.5,size=(n_vocab_q, 1)), ignore_label=0),
            term_weight_d=EmbedID_minus_pad(self.n_vocab_d, 1, initialW=RS.normal(scale=0.5,size=(n_vocab_d, 1)), ignore_label=0),
            lstm_q=L.NStepLSTM(n_layers=1, in_size=self.embed_dim, out_size=self.embed_dim, dropout=0.5),
            lstm_d=L.NStepLSTM(n_layers=1, in_size=self.embed_dim, out_size=self.embed_dim, dropout=0.5),
            conv_q=L.Convolution2D(in_channels=1, out_channels=self.cnn_out_channels, ksize=(self.cnn_ksize, self.embed_dim)),
            conv_d=L.Convolution2D(in_channels=1, out_channels=self.cnn_out_channels, ksize=(self.cnn_ksize, self.embed_dim)),
            l1=L.Linear(in_size=self.cnn_out_channels*2, out_size=self.n_hdim),
            l2=L.Linear(in_size=self.n_hdim, out_size=self.n_hdim),
            l3=L.Linear(in_size=self.n_hdim, out_size=self.n_hdim),
            l4=L.Linear(in_size=self.n_hdim, out_size=self.n_hdim),
            lo=L.Linear(in_size=self.n_hdim, out_size=1),
        )

    def __call__(self, xs1, xs2):
        batchsize1, batchsize2 = len(xs1), len(xs2)

        if self.encode_type == "cnn":
            h_query = self.encode_cnn(xs1, self.embed_q)
            h_doc_rel = self.encode_cnn(xs2, self.embed_d)

        elif self.encode_type == "lstm":
            e_seq_batch_q = embed_seq_batch(self.embed_q, xs1)
            e_seq_batch_d_rel = embed_seq_batch(self.embed_d, xs2)

            h_query, c, y = self.lstm_q(hx=None, cx=None, xs=e_seq_batch_q)
            h_doc_rel, c, y = self.lstm_d(hx=None, cx=None, xs=e_seq_batch_d_rel)

            h_query = F.reshape(h_query, (batchsize1, self.embed_dim))
            h_doc_rel = F.reshape(h_doc_rel, (batchsize2, self.embed_dim))

        elif self.encode_type == "avg":
            h_query = self.seq_encode(xs1, self.embed_q, self.term_weight_q)
            h_doc_rel = self.seq_encode(xs2, self.embed_d, self.term_weight_d)

            h_query = F.reshape(h_query, (batchsize1, self.embed_dim))
            h_doc_rel = F.reshape(h_doc_rel, (batchsize2, self.embed_dim))
        else:
            print("Error: encode_type is invalid")
            exit()

        concat_feature_rel = F.concat([h_query, h_doc_rel], axis=1)
        rel_score = self.cal_score(concat_feature_rel)

        return F.reshape(rel_score, (batchsize1, 1))


    def cal_score(self, feature_vec):
        h_mid = F.relu(F.dropout(self.l1(feature_vec)))
        if self.n_layer > 1:
            h_mid = F.relu(F.dropout(self.l2(h_mid)))
            if self.n_layer > 2:
                h_mid = F.relu(F.dropout(self.l3(h_mid)))
                if self.n_layer > 3:
                    h_mid = F.relu(F.dropout(self.l4(h_mid)))

        ho = self.lo(h_mid)
        return ho

    def encode_cnn(self, xs, embed):
        xs_embed = embed(xs)
        xs_embed.unchain_backward()

        batchsize, seq_len, _ = xs_embed.shape
        xs_conv = F.tanh(self.conv_q(F.reshape(xs_embed, (batchsize, 1, seq_len, self.embed_dim))))
        xs_avg_pool = F.average_pooling_2d(xs_conv, ksize=(xs_conv.shape[2],1))
        xs_avg_pool = F.reshape(xs_avg_pool, (batchsize, self.cnn_out_channels))
        return xs_avg_pool

    def seq_encode(self, xs, embed, term_weight_embed):
        embed_xs = embed(xs)
        embed_xs.unchain_backward()

        if self.encode_type == "avg" and self.weighted_sum:
            batchsize, seq_len, _ = embed_xs.shape
            term_weight = term_weight_embed(xs)
            normalized_weight = self.normalize_weight(term_weight)
            
            weight = F.tile(normalized_weight, self.embed_dim)
            # weight = F.broadcast_to(normalized_weight, (batchsize, seq_len, self.embed_dim))
            
            sum_embed_xs = F.sum(embed_xs * weight, axis=1)
        else:
            sum_embed_xs = F.sum(embed_xs, axis=1)

        return sum_embed_xs

    def normalize_weight(self, term_weight):
        batchsize, seq_len, _ = term_weight.shape
        weight_sum = F.sum(F.exp(term_weight), axis=1)

        # weight = F.tile(weight_sum, seq_len)
        weight = F.broadcast_to(weight_sum, (batchsize, seq_len))
        normalized_weight = F.exp(term_weight) / F.reshape(weight, (batchsize, seq_len, 1))

        return normalized_weight



