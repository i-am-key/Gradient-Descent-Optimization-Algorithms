import numpy as np
from utils import *

        
class Conv2d(object):
    def __init__(self, in_channels, out_channels, kernel_size, lr, stride=1, padding=0, momentum=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernal_size = kernel_size
        self.w = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) / np.sqrt(out_channels / 2.)
        self.vdw = np.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.stride = stride
        self.padding = padding
        self.dw = None
        self.x = None
        self.col_x = None
        self.col_w = None
        self.col = None
        self.col_w = None
        self.lr = lr
        self.momentum = momentum
        # decaying averages of past
        self.m_w = np.zeros((out_channels, in_channels, kernel_size, kernel_size))
        # past squared gradients 
        self.v_w = np.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.g_w = np.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.e_w = np.zeros((out_channels, in_channels, kernel_size, kernel_size))


    def forward(self, x):
        self.x = x
        FN, C, FH, FW = self.w.shape
        N, C, H, W = x.shape
        H_out = 1 + int((H - FH + 2 * self.padding) / self.stride)
        W_out = 1 + int((W - FW + 2 * self.padding) / self.stride)
        col_x = im2col(x, FH, FW, self.stride, self.padding)
        col_w = self.w.reshape(FN, -1).T
        self.col_x = col_x
        self.col_w = col_w
        out = np.dot(col_x, col_w)
        out = out.reshape(N, H_out, W_out, -1)
        out = out.transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        FN, C, FH, FW = self.w.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)
        self.dw = np.dot(self.col_x.T, dout)
        self.dw = self.dw.transpose(1, 0).reshape(self.w.shape)
        dcol_x = np.dot(dout, self.col_w.T)
        dx = col2im(dcol_x, self.x.shape, FH, FW, self.stride, self.padding)
        return dx

    def weight_matrix(self):
        return self.w

    def zero_grads(self):
        self.dw = np.zeros_like(self.w)

    def SGD(self):
        if not self.momentum:
            self.w -= self.lr * self.dw
        else:
            self.vdw = -self.lr * self.dw + self.momentum * self.vdw
            self.w += self.vdw

    def adam(self, t):
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        self.m_w = beta1 * self.m_w + (1. - beta1) * self.dw
        self.v_w = beta2 * self.v_w + (1. - beta2) * (self.dw**2)
        self.mt_w = self.m_w / (1. - beta1**(t))
        self.vt_w = self.v_w / (1. - beta2**(t))
        self.w -= self.lr * self.mt_w / (np.sqrt(self.vt_w) + eps)

    def adagrad(self):
        eps = 1e-10
        self.g_w += self.dw**2
        self.w -= self.lr * self.dw / (np.sqrt(self.g_w) + eps)

    def RMSprop(self):
        eps = 1e-4
        self.e_w = 0.9 * self.e_w + 0.1 * self.dw**2
        self.w -= self.lr * self.dw / (np.sqrt(self.e_w) + eps) 
        


class MaxPool2d(object):
    def __init__(self, kernel_size, stride, padding=0):
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.index = None
        self.x = None
        # kernel_size = 2

    def forward(self, x):
        N, C, H, W = x.shape
        self.x = x
        H_out = int(1 + (H - self.kernel_size) / self.stride)
        W_out = int(1 + (W - self.kernel_size) / self.stride)
        col_x = im2col(x, self.kernel_size, self.kernel_size, self.stride, self.padding)
        col_x = col_x.reshape(-1, self.kernel_size * self.kernel_size)
        self.index = np.argmax(col_x, axis=1)
        out = np.max(col_x, axis=1)
        out = out.reshape(N, H_out, W_out, C).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        pool_size = self.kernel_size * self.kernel_size
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.index.shape[0]), self.index] = dout.flatten()
        dmax = dmax.reshape(dout.shape[0], dout.shape[1], dout.shape[2], dout.shape[3], pool_size)
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.kernel_size, self.kernel_size, self.stride, self.padding)
        return dx


class FullyConnected(object):
    def __init__(self, input_size, output_size, lr, momentum=0):
        self.lr = lr
        self.w = np.random.randn(input_size, output_size) / np.sqrt(input_size / 2.)
        self.vdw = np.zeros((input_size, output_size))
        self.b = np.zeros((1, output_size))
        self.vdb = np.zeros((1, output_size))
        self.momentum = momentum
        # decaying averages of past
        self.m_w = np.zeros((input_size, output_size))
        self.m_b = np.zeros((1, output_size))
        # past squared gradients 
        self.v_w = np.zeros((input_size, output_size))
        self.v_b = np.zeros((1, output_size))
        self.g_w = np.zeros((input_size, output_size))
        self.g_b = np.zeros((1, output_size))
        self.e_w = np.zeros((input_size, output_size))
        self.e_b = np.zeros((1, output_size))


    def forward(self, x):
        self.x = x
        out = np.dot(self.x, self.w) + self.b
        return out

    def backward(self, dout):
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = np.dot(dout, self.w.T)
        return dx

    def zero_grads(self):
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)

    def SGD(self):
        if not self.momentum:
            self.w -= self.lr * self.dw
            self.b -= self.lr * self.db
        else:
            self.vdw = -self.lr * self.dw + self.momentum * self.vdw
            self.w += self.vdw
            self.vdb = -self.lr * self.db + self.momentum * self.vdb
            self.b += self.vdb
    
    def adam(self, t):
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        self.m_w = beta1 * self.m_w + (1. - beta1) * self.dw
        self.m_b = beta1 * self.m_b + (1. - beta1) * self.db
        self.v_w = beta2 * self.v_w + (1. - beta2) * (self.dw**2)
        self.v_b = beta2 * self.v_b + (1. - beta2) * (self.db**2)
        self.mt_w = self.m_w / (1. - beta1**(t))
        self.vt_w = self.v_w / (1. - beta2**(t))
        self.mt_b = self.m_b / (1. - beta1**(t))
        self.vt_b = self.v_b / (1. - beta2**(t))
        self.w -= self.lr * self.mt_w / (np.sqrt(self.vt_w) + eps)
        self.b -= self.lr * self.mt_b / (np.sqrt(self.vt_b) + eps)

    def adagrad(self):
        eps = 1e-10
        self.g_w += self.dw**2
        self.w  -= self.lr * self.dw / (np.sqrt(self.g_w) + eps)
        self.g_b += self.db**2
        self.b -= self.lr * self.db / (np.sqrt(self.g_b) + eps)

    def RMSprop(self):
        eps = 1e-4
        self.e_w = 0.9 * self.e_w + 0.1 * self.dw**2
        self.w -= self.lr * self.dw / (np.sqrt(self.e_w) + eps) 
        self.e_b = 0.9 * self.e_b + 0.1 * self.db**2
        self.b -= self.lr * self.db / (np.sqrt(self.e_b) + eps)



class Sigmoid(object):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        h = 1 / (1 + np.exp(-x))
        self.state = h
        return h

    def derivative(self):
        h_q = np.multiply(self.state, 1 - self.state)
        return h_q


class Tanh(object):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        x1 = np.exp(x)
        x2 = np.exp(-x)
        h = (x1 - x2) / (x1 + x2)
        self.state = h
        return h

    def derivative(self):
        h_q = 1 - np.square(self.state)
        return h_q


class ReLU(object):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        res = np.maximum(x, 0)
        self.state = res
        return res

    def derivative(self):
        res = np.zeros_like(self.state)
        res[self.state > 0] = 1
        return res

class SoftmaxCrossEntropy(object):
    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()
        self.sm = None

    def forward(self, x, y):
        self.logits = x
        self.labels = y
        exps = np.exp(x)
        p = exps / (np.sum(exps, axis=1)).reshape(-1, 1)
        self.sm = p
        L = - np.sum(y * np.log(p), axis=1)
        return L

    def backward(self):
        return self.sm - self.labels
