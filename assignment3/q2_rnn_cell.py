#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q2(c): Recurrent neural nets for NER
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

logger = logging.getLogger("hw3.q2.1")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

# 这里我改写这个函数，继承的Pytorch的Module
class RNNCell(t.nn.Module):
    """Wrapper around our RNN cell implementation that allows us to play
    nicely with Pytorch.
    """
    def __init__(self, input_size, hidden_size):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_x = Parameter(t.Tensor(input_size, hidden_size))
        self.W_h = Parameter(t.Tensor(hidden_size, hidden_size))
        self.bias = Parameter(t.Tensor(hidden_size,))
        self.reset_parameters()
        
    def reset_parameters(self):
        # 初始化参数，两个W按照xavier的方式初始化，而bias初始化为0
        for i, w in enumerate(self.parameters()):
            if i != 2:
                w.data = nn.init.xavier_uniform_(t.zeros((w.size(0), w.size(1)), dtype=t.float32))
            else:
                w.data = t.zeros((w.size(0),), dtype=t.float32)

    def forward(self, inputs, hx):

        hx = t.sigmoid(t.mm(inputs, self.W_x) + t.mm(hx, self.W_h) + self.bias)
        output = hx

        return output, hx
    # # Python内置的@property装饰器就是负责把一个方法变成属性调用的
    # # 下面两个方法都只是get方法
    # @property
    # def state_size(self):
    #     return self._state_size

    # @property
    # def output_size(self):
    #     return self._state_size

    # def __call__(self, inputs, state, scope=None):
    #     """Updates the state using the previous @state and @inputs.
    #     Remember the RNN equations are:

# 当我用pytorch改写了上述函数后，就不能使用这个测试函数去测试了
# 因为tensorflow和Pytorch初始化W的随机的方式不一样
# 所以就不能使用ht和ht_去比较大小了
def test_rnn_cell():

    x = t.tensor([
        [0.4, 0.5, 0.6],
        [0.3, -0.2, -0.1]], dtype=t.float32)
    h = t.tensor([
        [0.2, 0.5],
        [-0.3, -0.3]], dtype=t.float32)
    y = t.tensor([
        [0.832, 0.881],
        [0.731, 0.622]], dtype=t.float32)

    cell = RNNCell(3, 2)
    y_, ht_ = cell(x, h)
    ht = y

    print("y_ = " + str(y_))
    print("ht_ = " + str(ht_))

    assert np.allclose(y_.detach().numpy(), ht_.detach().numpy()), "output and state should be equal."
    assert np.allclose(ht.detach().numpy(), ht_.detach().numpy(), atol=1e-2), "new state vector does not seem to be correct."

def do_test(_):
    logger.info("Testing rnn_cell")
    test_rnn_cell()
    logger.info("Passed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tests the RNN cell implemented as part of Q2 of Homework 3')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('test', help='')
    command_parser.set_defaults(func=do_test)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
