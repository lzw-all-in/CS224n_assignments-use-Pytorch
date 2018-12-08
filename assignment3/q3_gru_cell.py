#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q3(d): Grooving with GRUs
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys
from torch.nn.parameter import Parameter
import torch as t
import torch.nn as nn
import numpy as np

logger = logging.getLogger("hw3.q3.1")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class GRUCell(t.nn.Module):
    """Wrapper around our GRU cell implementation that allows us to play
    nicely with Pytorch.
    """
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_z = Parameter(t.Tensor(input_size, hidden_size))
        self.U_z = Parameter(t.Tensor(hidden_size, hidden_size))
        self.b_z = Parameter(t.Tensor(hidden_size,))
        self.W_r = Parameter(t.Tensor(input_size, hidden_size))
        self.U_r = Parameter(t.Tensor(hidden_size, hidden_size))
        self.b_r = Parameter(t.Tensor(hidden_size,))
        self.W_h = Parameter(t.Tensor(input_size, hidden_size))
        self.U_h = Parameter(t.Tensor(hidden_size, hidden_size))
        self.b_h = Parameter(t.Tensor(hidden_size,))
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化参数，两个W按照xavier的方式初始化，而bias初始化为0
        for i, w in enumerate(self.parameters()):
            if i % 3 == 2:
                w.data = t.zeros((w.size(0),), dtype=t.float32)
            else:
                w.data = nn.init.xavier_uniform_(t.zeros((w.size(0), w.size(1)), dtype=t.float32))

    def forward(self, inputs, hx):
        z_t = t.sigmoid(t.mm(inputs, self.W_z) + t.mm(hx, self.U_z) + self.b_z)
        r_t = t.sigmoid(t.mm(inputs, self.W_r) + t.mm(hx, self.U_r) + self.b_r)
        o_t = t.tanh(t.mm(inputs, self.W_h) + t.mm(r_t * hx, self.U_h) + self.b_h)
        hx = z_t * hx + (1 - z_t) * o_t

        output = hx
        return output, hx
    # @property
    # def state_size(self):
    #     return self._state_size

    # @property
    # def output_size(self):
    #     return self._state_size

    # def __call__(self, inputs, state, scope=None):
    #     """Updates the state using the previous @state and @inputs.
    #     Remember the GRU equations are:

    #     z_t = sigmoid(x_t W_z + h_{t-1} U_z + b_z)
    #     r_t = sigmoid(x_t W_r + h_{t-1} U_r + b_r)
    #     o_t = tanh(x_t W_o + r_t * h_{t-1} U_o + b_o)
    #     h_t = z_t * h_{t-1} + (1 - z_t) * o_t

    #     TODO: In the code below, implement an GRU cell using @inputs
    #     (x_t above) and the state (h_{t-1} above).
    #         - Define U_r, W_r, b_r, U_z, W_z, b_z and U_o, W_o, b_o to
    #           be variables of the apporiate shape using the
    #           `tf.get_variable' functions.
    #         - Compute z, r, o and @new_state (h_t) defined above
    #     Tips:
    #         - Remember to initialize your matrices using the xavier
    #           initialization as before.
    #     Args:
    #         inputs: is the input vector of size [None, self.input_size]
    #         state: is the previous state vector of size [None, self.state_size]
    #         scope: is the name of the scope to be used when defining the variables inside.
    #     Returns:
    #         a pair of the output vector and the new state vector.
    #     """
    #     scope = scope or type(self).__name__

    #     # It's always a good idea to scope variables in functions lest they
    #     # be defined elsewhere!
    #     with tf.variable_scope(scope):
    #         ### YOUR CODE HERE (~20-30 lines)
    #         pass
    #         ### END YOUR CODE ###
    #     # For a GRU, the output and state are the same (N.B. this isn't true
    #     # for an LSTM, though we aren't using one of those in our
    #     # assignment)
    #     output = new_state
    #     return output, new_state

# 当我用pytorch改写了上述函数后，就不能使用这个测试函数去测试了
# 因为tensorflow和Pytorch初始化W的随机的方式不一样
# 所以就不能使用ht和ht_去比较大小了
def test_gru_cell():

    cell = GRUCell(3, 2)

    x = t.tensor([
        [0.4, 0.5, 0.6],
        [0.3, -0.2, -0.1]], dtype=t.float32)
    h = t.tensor([
        [0.2, 0.5],
        [-0.3, -0.3]], dtype=t.float32)
    y = t.tensor([
        [ 0.320, 0.555],
        [-0.006, 0.020]], dtype=t.float32)
    ht = y

    y_, ht_ = cell(x, h)
    print("y_ = " + str(y_))
    print("ht_ = " + str(ht_))

    assert np.allclose(y_.detach().numpy(), ht_.detach().numpy()), "output and state should be equal."
    assert np.allclose(ht.detach().numpy(), ht_.detach().numpy(), atol=1e-2), "new state vector does not seem to be correct."

def do_test(_):
    logger.info("Testing gru_cell")
    test_gru_cell()
    logger.info("Passed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tests the GRU cell implemented as part of Q3 of Homework 3')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('test', help='')
    command_parser.set_defaults(func=do_test)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
