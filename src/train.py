# coding: utf-8
import os
HOME = os.getenv("HOME")
os.environ["CHAINER_SEED"] = "1"
import chainer
import chainer.links as L
import chainer.optimizers as O
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from chainer import cuda, serializers
import random
import numpy as np
random.seed(0)
np.random.seed(0)
import argparse
chainer.config.cudnn_deterministic = True

from model import Network, Network_deep, DataProcessor, RankingEvaluator
from model import converter_for_lstm, concat_examples
from datetime import datetime
import cPickle
import json

def main(args):
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        chainer.cuda.cupy.random.seed(0)
        t = np.array([0], dtype=np.int32)
        tmp = cuda.to_gpu(t)

    # setup result directory
    start_time = datetime.now().strftime('%Y%m%d_%H_%M_%S')
    if args.test:
        start_time = "test_" + start_time
    result_dest = args.result_dir + '/' + start_time
    result_abs_dest = os.path.abspath(result_dest)
    if not args.extract_parameter:
        os.makedirs(result_dest)
        with open(os.path.join(result_abs_dest, "settings.json"), "w") as fo:
            fo.write(json.dumps(vars(args), sort_keys=True, indent=4))

    # data setup
    data_processor = DataProcessor(args)
    data_processor.prepare_dataset()
    vocab_q = data_processor.vocab_q
    vocab_d = data_processor.vocab_d

    if args.create_vocabulary:
        print('dump')
        with open(args.vocab_path+"en_{}_vocab_for_index.txt".format(args.doc_lang),"wb") as f_q,\
             open(args.vocab_path+"{}_vocab_for_index.txt".format(args.doc_lang),'wb') as f_d:
            cPickle.dump(dict(vocab_q), f_q)
            cPickle.dump(dict(vocab_d), f_d)
        print('done')

    # model setup
    if args.deep:
        nn = Network_deep(args, len(vocab_q), len(vocab_d))
    else:    
        nn = Network(args, len(vocab_q), len(vocab_d))
    if args.load_embedding:
        nn.load_embeddings(args, vocab_q, "query")
        nn.load_embeddings(args, vocab_d, "document")

    model = L.Classifier(nn, lossfun=F.hinge)
    model.compute_accuracy = False
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    # optimizer setup
    if args.optimizer == "adagrad":
        optimizer = O.AdaGrad(lr=0.05)
    else:
        optimizer = O.Adam()
    optimizer.setup(model)

    # converter setup
    if args.encode_type == "lstm":
        converter = converter_for_lstm
    else:
        converter = concat_examples

    # iterator, updater and trainer setup
    train_iter = chainer.iterators.SerialIterator(data_processor.train_data, args.batchsize)
    dev_iter = chainer.iterators.SerialIterator(data_processor.dev_data, args.batchsize, repeat=False, shuffle=False)
    test_iter = chainer.iterators.SerialIterator(data_processor.test_data, args.batchsize, repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, converter=converter, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=result_dest)

    model_path = '/'.join(args.model_path.split('/')[0:-1])+'/'
    model_epoch = args.model_path.split('/')[-1].split('_')[-1]
    print('model path = ' + model_path + 'model_epoch_{}'.format(model_epoch))

    if args.load_snapshot:
        print("loading snapshot...")
        from IPython.core.debugger import Pdb; Pdb().set_trace()
        serializers.load_npz(model_path +'model_epoch_{}'.format(model_epoch), trainer)
        # serializers.load_npz(model_path +'model_epoch_{}'.format(model_epoch), model, path='updater/model:main/')
        print('done')
        exit()

    if args.extract_parameter:
        print('extract parameter...')
        serializers.load_npz(model_path +'model_epoch_{}'.format(model_epoch), trainer)
        if args.deep:
            p1 = model.predictor.l1
            p2 = model.predictor.l2
            p3 = model.predictor.l3
            p4 = model.predictor.l4
            serializers.save_npz(model_path+"l1.npz", p1)
            serializers.save_npz(model_path+"l2.npz", p2)
            serializers.save_npz(model_path+"l3.npz", p3)
            serializers.save_npz(model_path+"l4.npz", p4)
        p5 = model.predictor.conv_q
        serializers.save_npz(model_path+"conv_q.npz", p5)
        print('done')
        exit()

    if args.load_parameter:
        print("loading parameter...")
        if args.deep:
            p1 = model.predictor.l1
            p2 = model.predictor.l2
            p3 = model.predictor.l3
            p4 = model.predictor.l4
            serializers.load_npz(model_path+ "l1.npz", p1)
            serializers.load_npz(model_path+ "l2.npz", p2)
            serializers.load_npz(model_path+ "l3.npz", p3)
            serializers.load_npz(model_path+ "l4.npz", p4)
        p5 = model.predictor.conv_q
        serializers.load_npz(model_path+ "conv_q.npz", p5)
        print('done')

    # Evaluation setup
    iters = {"dev": dev_iter, "test": test_iter}
    trainer.extend(RankingEvaluator(iters, model, args, result_abs_dest, data_processor.n_test_qd_pairs, converter=converter, device=args.gpu))

    # # Log reporter setup
    trainer.extend(extensions.LogReport(log_name='log'))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss_dev', 'validation/main/loss_test']))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.extend(extensions.snapshot(filename='model_epoch_{.updater.epoch}'),
        trigger=chainer.training.triggers.MinValueTrigger('validation/main/loss_dev'))

    trainer.run()

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu  ', dest='gpu', type=int,default=-1, help='negative value indicates CPU')

    # training parameter
    parser.add_argument('--epoch', dest='epoch', type=int,default=5, help='number of epochs to learn')
    parser.add_argument('--batchsize', dest='batchsize', type=int,default=32, help='learning minibatch size')
    parser.add_argument('--doc_lang', dest='doc_lang', type=str, default="ja")
    parser.add_argument('--encode_type', dest='encode_type', type=str, default="cnn")
    parser.add_argument('--op', dest='optimizer', type=str, default="adam")
    parser.add_argument('--sub_sample_data_limit', type=int, default=31716, help='')

    # training flag
    parser.add_argument('--deep', action='store_true', help='')
    parser.add_argument('--load_embedding', action='store_true', help='')
    parser.add_argument('--load_parameter', action='store_true', help='')
    parser.add_argument('--load_snapshot', action='store_true', help='')
    parser.add_argument('--weighted_sum', action='store_true', help='')
    parser.add_argument('--sub_sample_train', action='store_true', help='')
    parser.add_argument('--test', action='store_true', help='use tiny dataset')

    # other flag
    parser.add_argument('--create_vocabulary', action='store_true', help='')
    parser.add_argument('--extract_parameter', action='store_true', help='')

    # model parameter
    parser.add_argument('--n_layer', dest='n_layer', type=int, default=1, help='# of layer')
    parser.add_argument('--n_hdim', dest='n_hdim', type=int, default=200, help='dimension of hidden layer')
    parser.add_argument('--vocab_size', dest='vocab_size', type=int, default=100000, help='')
    parser.add_argument('--cnn_out_channels', dest='cnn_out_channels', type=int, default=100, help='')
    parser.add_argument('--cnn_ksize', dest='cnn_ksize', type=int, default=4, help='')
    parser.add_argument('--embed_dim', dest='embed_dim', type=int, default=100, help='# of layer')

    # data path
    parser.add_argument('--vocab_path', dest='vocab_path', type=str, default=HOME+"/clir/vocab/")
    parser.add_argument('--data_path', dest='data_path', type=str,default="/path/")
    parser.add_argument('--vec_path', dest='vec_path', type=str, default=HOME+"/word2vec/trunk/")
    parser.add_argument('--model_path', dest='model_path', type=str, default='')

    parser.add_argument('--result_dir', type=str, default='')
    args = parser.parse_args()
    main(args)

