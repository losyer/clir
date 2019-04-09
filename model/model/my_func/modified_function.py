# from chainer.functions.connection import embed_id
from chainer.initializers import normal
from chainer import link
from chainer import variable

class EmbedID_minus_pad(link.Link):

    """Efficient linear layer for one-hot input.
    This is a link that wraps the :func:`~chainer.functions.embed_id` function.
    This link holds the ID (word) embedding matrix ``W`` as a parameter.
    Args:
        in_size (int): Number of different identifiers (a.k.a. vocabulary
            size).
        out_size (int): Size of embedding vector.
        initialW (2-D array): Initial weight value. If ``None``, then the
            matrix is initialized from the standard normal distribution.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
        ignore_label (int or None): If ``ignore_label`` is an int value,
            ``i``-th column of return value is filled with ``0``.
    .. seealso:: :func:`chainer.functions.embed_id`
    Attributes:
        W (~chainer.Variable): Embedding parameter matrix.
    """

    ignore_label = None

    def __init__(self, in_size, out_size, initialW=None, ignore_label=None):
        super(EmbedID_minus_pad, self).__init__()
        self.ignore_label = ignore_label

        with self.init_scope():
            if initialW is None:
                initialW = normal.Normal(1.0)
            self.W = variable.Parameter(initialW, (in_size, out_size))

    def __call__(self, x):
        """Extracts the word embedding of given IDs.
        Args:
            x (~chainer.Variable): Batch vectors of IDs.
        Returns:
            ~chainer.Variable: Batch of corresponding embeddings.
        """
        return embed_id(x, self.W, ignore_label=self.ignore_label)

import numpy
import six

import chainer
from chainer import cuda
from chainer import function
from chainer.utils import type_check


class EmbedIDFunction(function.Function):

    def __init__(self, ignore_label=None):
        self.ignore_label = ignore_label

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, w_type = in_types
        type_check.expect(
            x_type.dtype == numpy.int32,
            x_type.ndim >= 1,
        )
        type_check.expect(
            w_type.dtype == numpy.float32,
            w_type.ndim == 2
        )

    def forward(self, inputs):
        x, W = inputs

        if not type_check.same_types(*inputs):
            raise ValueError('numpy and cupy must not be used together\n'
                             'type(W): {0}, type(x): {1}'
                             .format(type(W), type(x)))

        xp = cuda.get_array_module(*inputs)
        if chainer.is_debug():
            valid_x = xp.logical_and(0 <= x, x < len(W))
            if self.ignore_label is not None:
                valid_x = xp.logical_or(valid_x, x == self.ignore_label)
            if not valid_x.all():
                raise ValueError('Each not ignored `x` value need to satisfy'
                                 '`0 <= x < len(W)`')

        if self.ignore_label is not None:
            mask = (x == self.ignore_label)
            return xp.where(
                mask[..., None], -1000000.0, W.take(xp.where(mask, 0, x), axis=0)),

        return W.take(x, axis=0),

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        x, W = inputs
        gy = grad_outputs[0]
        gW = xp.zeros_like(W)

        if xp is numpy:
            # It is equivalent to `numpy.add.at(gW, x, gy)` but ufunc.at is
            # too slow.
            for ix, igy in six.moves.zip(x.ravel(),
                                         gy.reshape(x.size, -1)):
                if ix == self.ignore_label:
                    continue
                gW[ix] += igy
        else:
            if self.ignore_label is None:
                cuda.elementwise(
                    'T gy, int32 x, int32 n_out', 'raw T gW',
                    'int w_ind[] = {x, i % n_out}; atomicAdd(&gW[w_ind], gy)',
                    'embed_id_bwd')(
                        gy, xp.expand_dims(x, -1), gW.shape[1], gW)
            else:
                cuda.elementwise(
                    'T gy, int32 x, int32 n_out, int32 ignore', 'raw T gW',
                    '''
                    if (x != ignore) {
                      int w_ind[] = {x, i % n_out};
                      atomicAdd(&gW[w_ind], gy);
                    }
                    ''',
                    'embed_id_bwd_ignore_label')(
                        gy, xp.expand_dims(x, -1), gW.shape[1],
                        self.ignore_label, gW)
        return None, gW


def embed_id(x, W, ignore_label=None):
    """Efficient linear function for one-hot input.
    This function implements so called *word embedding*. It takes two
    arguments: a set of IDs (words) ``x`` in :math:`B` dimensional integer
    vector, and a set of all ID (word) embeddings ``W`` in :math:`V \\times d`
    float32 matrix. It outputs :math:`B \\times d` matrix whose ``i``-th
    column is the ``x[i]``-th column of ``W``.
    This function is only differentiable on the input ``W``.
    Args:
        x (~chainer.Variable): Batch vectors of IDs.
        W (~chainer.Variable): Representation of each ID (a.k.a.
            word embeddings).
        ignore_label (int or None): If ``ignore_label`` is an int value,
            ``i``-th column of return value is filled with ``0``.
    Returns:
        ~chainer.Variable: Output variable.
    .. seealso:: :class:`~chainer.links.EmbedID`
    """
    return EmbedIDFunction(ignore_label=ignore_label)(x, W)