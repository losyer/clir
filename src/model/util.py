# coding: utf-8
import sys
import numpy as np
import six
from collections import namedtuple
import chainer
from chainer import cuda, Function, Variable, optimizers, serializers, utils
from chainer import Link, Chain, variable
import chainer.functions as F
import chainer.links as L

def cos_sim(x, y):
    if len(x.shape) > 2:
        norm_x = F.normalize(F.squeeze(F.squeeze(x,axis=(2,)),axis=(2,)))
        norm_y = F.normalize(F.squeeze(F.squeeze(y,axis=(2,)),axis=(2,)))
    else:
        norm_x = F.normalize(x)
        norm_y = F.normalize(y)
    return F.batch_matmul(norm_x, norm_y, transa=True)

class SelectiveWeightDecay(object):
    name = 'SelectiveWeightDecay'

    def __init__(self, rate, decay_params):
        self.rate = rate
        self.decay_params = decay_params

    def kernel(self):
        return cuda.elementwise(
            'T p, T decay', 'T g', 'g += decay * p', 'weight_decay')

    def __call__(self, opt):
        rate = self.rate
        for name, param in opt.target.namedparams():
            if name in self.decay_params:
                p, g = param.data, param.grad
                with cuda.get_device_from_id(p) as dev:
                    if int(dev) == -1:
                        g += rate * p
                    else:
                        self.kernel()(p, rate, g)

# basically this is same as the one on chainer's repo.
# I added padding option (padding=0) to be always true

# def concat_examples(batch, device=None, padding=None):
def concat_examples(batch, device=None, padding=0):
    if len(batch) == 0:
        raise ValueError('batch is empty')

    if device is None:
        def to_device(x):
            return x
    elif device < 0:
        to_device = cuda.to_cpu
    else:
        def to_device(x):
            return cuda.to_gpu(x, device, cuda.Stream.null)

    first_elem = batch[0]

    if isinstance(first_elem, tuple):
        result = []
        if not isinstance(padding, tuple):
            padding = [padding] * len(first_elem)

        for i in six.moves.range(len(first_elem)):
            if i == len(first_elem)-1:
                # for hinge loss function
                concat = _concat_arrays([example[i] for example in batch], padding[i])
                concat = np.reshape(concat, (len(batch), ))
            else:
                concat = _concat_arrays([example[i] for example in batch], padding[i])
            result.append(to_device(concat))
        return tuple(result)

    elif isinstance(first_elem, dict):
        print("not impremented")
        assert False 
        exit()
        result = {}
        if not isinstance(padding, dict):
            padding = {key: padding for key in first_elem}

        for key in first_elem:
            result[key] = to_device(_concat_arrays(
                [example[key] for example in batch], padding[key]))
        return result


def _concat_arrays(arrays, padding):
    if padding is not None:
        return _concat_arrays_with_padding(arrays, padding)

    xp = cuda.get_array_module(arrays[0])
    with cuda.get_device_from_array(arrays[0]):
        return xp.concatenate([array[None] for array in arrays])

def _concat_arrays_with_padding(arrays, padding):
    shape = np.array(arrays[0].shape, dtype=int)
    for array in arrays[1:]:
        if np.any(shape != array.shape):
            np.maximum(shape, array.shape, shape)
    shape = tuple(np.insert(shape, 0, len(arrays)))

    xp = cuda.get_array_module(arrays[0])
    with cuda.get_device_from_array(arrays[0]):
        result = xp.full(shape, padding, dtype=arrays[0].dtype)
        for i in six.moves.range(len(arrays)):
            src = arrays[i]
            slices = tuple(slice(dim) for dim in src.shape)
            result[(i,) + slices] = src

    return result

def converter_for_lstm(batch, device):
    if device is None:
        def to_device(x):
            return x
    elif device < 0:
        to_device = cuda.to_cpu
    else:
        def to_device(x):
            return cuda.to_gpu(x, device, cuda.Stream.null)

    def to_device_batch(batch):
        # batch = [array,...]
        if device is None:
            return batch
        elif device < 0:
            return [to_device(x) for x in batch]
        else:
            xp = cuda.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = np.cumsum([len(x) for x in batch[:-1]], dtype='i')
            concat_dev = to_device(concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    if device < 0:
        xp = np
    else:
        xp = cuda.cupy
    return tuple([to_device_batch([x for x, _, _, _ in batch]),
                     to_device_batch([y for _, y, _ ,_ in batch]),
                     to_device_batch([z for _, _, z, _ in batch]),
                     xp.reshape(xp.array(to_device_batch([w for _, _, _, w in batch]), dtype=xp.int32), (len(batch), ))])

    