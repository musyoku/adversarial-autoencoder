# -*- coding: utf-8 -*-
import sys, os
from chainer import cuda, gradient_check
sys.path.append(os.path.split(os.getcwd())[0])
from util import Adder

xp = cuda.cupy
x = xp.random.uniform(-1.0, 1.0, (2, 256)).astype(xp.float32)
label = xp.zeros((2, 10)).astype(xp.float32)
label[0,3] = 1
label[1,6] = 1

y_grad = xp.ones((2, 256 + 10)).astype(xp.float32)
gradient_check.check_backward(Adder(), (x, label), y_grad, eps=1e-2)