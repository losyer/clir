# coding: utf-8
import argparse
import os
os.environ["CHAINER_SEED"] = "1"
import chainer
from chainer import cuda, serializers
import random
import numpy as np
random.seed(0)
np.random.seed(0)
chainer.config.cudnn_deterministic = True
from model import DataProcessor_for_inference, Network_deep_for_inference
from model import converter_for_lstm, concat_examples
import cPickle
import logging
formatter = '%(levelname)s : %(asctime)s : %(message)s'
logging.basicConfig(level=logging.DEBUG, format=formatter)

def compute_score(nn, iterator, converter, args):
    all_score = []
    logging.info('start to inference ...')
    with chainer.using_config('train', False):
        for batch in iterator:
            xs1, xs2, y = converter(batch, device=args.gpu)
            score = nn(xs1, xs2)
            all_score += score.data.flatten().tolist()

    print(all_score[0:5])
    print(all_score[5:10])
    print(all_score[10:15])
    logging.info('start to inference ... done')
    print('# of scores: {}'.format(len(all_score)))

    return all_score

def main(args):

    # data setup
    data_processor = DataProcessor_for_inference(args)
    data_processor.prepare_dataset()
    vocab_q = data_processor.vocab_q
    vocab_d = data_processor.vocab_d

    # netowrk setup
    nn = Network_deep_for_inference(args, len(vocab_q), len(vocab_d))

    # gpu setup
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        nn.to_gpu()

    logging.info('load parameter ...')
    serializers.load_npz(args.model_path, nn, path='updater/model:main/predictor/')
    logging.info('load parameter ... done')

    # iterator setup to make batch
    test_iter = chainer.iterators.SerialIterator(data_processor.test_data, args.batchsize, repeat=False, shuffle=False)

    # converter setup
    if args.encode_type == "lstm":
        converter = converter_for_lstm
    else:
        converter = concat_examples
    all_score = compute_score(nn, test_iter, converter, args)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()

    # general option
    parser.add_argument('--test', action='store_true', help='use tiny dataset')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batchsize', type=int,default=200)
    # parser.add_argument('--doc_lang', type=str, default="sw")
    parser.add_argument('--doc_lang', choices=['ja', 'de', 'fr', 'tl', 'sw'])
    parser.add_argument('--vocab_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_path', type=str)

    # model parameter/option
    parser.add_argument('--n_layer', type=int, default=1)
    parser.add_argument('--n_hdim', type=int, default=100)
    parser.add_argument('--cnn_ksize', type=int, default=4)
    parser.add_argument('--embed_dim', type=int, default=100)
    parser.add_argument('--cnn_out_channels', type=int, default=100)
    parser.add_argument('--vocab_size', type=int, default=100000)

    parser.add_argument('--encode_type', type=str, default="cnn")
    parser.add_argument('--weighted_sum', action='store_true')
    
    args = parser.parse_args()
    main(args)

