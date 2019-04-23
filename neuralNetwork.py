##  Python edition used: python 3
##  Name: Jingzhi Wang
##  Andrew ID: jingzhiw
import numpy as np
import os
import time
import matplotlib.pyplot as plt


class Activation(object):
    """
    Interface for activation functions (non-linearities).

    The base class of Identity, Sigmoid, Tanh and ReLu class.
    """
    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):
    """
    Identity function
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):
    """
    Sigmoid non-linearity
    """

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        h = 1 / (1 + np.exp(-x))
        self.state = h
        return h

    def derivative(self):
        h_q = np.multiply(self.state, 1 - self.state)
        return h_q


class Tanh(Activation):
    """
    Tanh non-linearity
    """
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


class ReLU(Activation):

    """
    ReLU non-linearity
    """

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


class Criterion(object):
    """
    Interface for loss functions.
    """

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class SoftmaxCrossEntropy(Criterion):
    """
    Softmax loss
    """

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

    def derivative(self):
        return self.sm - self.labels


class BatchNorm(object):

    def __init__(self, fan_in, alpha=0.9):

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        self.var = np.ones((1, fan_in))
        self.mean = np.zeros((1, fan_in))

        self.gamma = np.ones((1, fan_in))
        self.dgamma = np.zeros((1, fan_in))

        self.beta = np.zeros((1, fan_in))
        self.dbeta = np.zeros((1, fan_in))

        # inference parameters
        self.running_mean = np.zeros((1, fan_in))
        self.running_var = np.ones((1, fan_in))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        if eval:
            self.x = x
            self.norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            self.out = self.norm * self.gamma + self.beta
            return

        self.x = x
        self.mean = np.mean(x, axis=0)
        self.var = np.var(x, axis=0)
        self.norm = (x - self.mean)/np.sqrt(self.var + self.eps)
        self.out = self.norm * self.gamma + self.beta

        # update running batch statistics
        self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * self.mean
        self.running_var = self.alpha * self.running_var + (1 - self.alpha) * self.var

    def backward(self, delta):
        m = delta.shape[0]
        dxhat = delta * self.gamma
        dxhat_dvar = -0.5 * (self.x - self.mean) * (self.var + self.eps) ** -1.5
        dvar = np.sum(delta * dxhat_dvar, axis=0)
        dx1 = dxhat / np.sqrt(self.var + self.eps)
        dx2 = dvar * 2 * (self.x - self.mean) / m
        dmiu1 = -1 * np.sum(dx1, axis=0)
        dmiu2 = dvar * -2 * np.sum(self.x - self.mean, axis=0) / m
        dmiu = dmiu1 + dmiu2
        dx3 = dmiu / m
        dx = dx1 + dx2 + dx3
        self.dbeta = np.sum(delta, axis=0)
        self.dgamma = np.sum(delta * self.norm, axis=0)
        return dx


def random_normal_weight_init(d0, d1):
    x = np.sqrt(6 / float(d0 + d1))
    w = np.random.uniform(-x, x, (d0, d1))
    return w


def zeros_bias_init(d):
    b = np.zeros(d)
    return b
        

class MLP(object):
    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn, bias_init_fn, criterion, lr,
                 momentum=0.0, num_bn_layers=0, weight_decay=0.0, apply_dropout=False, dropout_ratio=0.5):


        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.apply_dropout = apply_dropout
        self.dropout_ratio = dropout_ratio

        self.input = None
        self.loss = None
        self.mask = None
        self.error = 0
        self.W = []
        self.b = []
        self.dW = []
        self.db = []
        self.bn_layers = []
        self.vdw = []
        self.vdb = []
        if self.nlayers == 1:
            self.W.append(weight_init_fn(self.input_size, self.output_size))
            self.dW.append(weight_init_fn(self.input_size, self.output_size))
            self.b.append(bias_init_fn(self.output_size))
            self.db.append(bias_init_fn(self.output_size))
            if self.momentum:
                self.vdw.append(np.zeros((self.input_size, self.output_size)))
                self.vdb.append(np.zeros(self.output_size))
        else:
            for i in range(self.nlayers):
                if i == 0:
                    self.W.append(weight_init_fn(self.input_size, hiddens[0]))
                    self.dW.append(np.zeros((self.input_size, hiddens[0])))
                    self.b.append(bias_init_fn(hiddens[0]))
                    self.db.append(np.zeros(hiddens[0]))
                    if self.momentum:
                        self.vdw.append(np.zeros((self.input_size, hiddens[0])))
                        self.vdb.append(np.zeros(hiddens[0]))
                elif i == self.nlayers - 1:
                    self.W.append(weight_init_fn(hiddens[i - 1], self.output_size))
                    self.dW.append(np.zeros((hiddens[i - 1], self.output_size)))
                    self.b.append(bias_init_fn(self.output_size))
                    self.db.append(np.zeros(self.output_size))
                    if self.momentum:
                        self.vdw.append(np.zeros((hiddens[i - 1], self.output_size)))
                        self.vdb.append(np.zeros(self.output_size))

                else:
                    self.W.append(weight_init_fn(hiddens[i - 1], hiddens[i]))
                    self.dW.append(np.zeros((hiddens[i - 1], hiddens[i])))
                    self.b.append(bias_init_fn(hiddens[i]))
                    self.db.append(np.zeros(hiddens[i]))
                    if self.momentum:
                        self.vdw.append(np.zeros((hiddens[i - 1], hiddens[i])))
                        self.vdb.append(np.zeros(hiddens[i]))


        # HINT: self.foo = [ bar(???) for ?? in ? ]
        # if batch norm, add batch norm parameters
        if self.bn:
            for i in range(self.num_bn_layers):
                self.bn_layers.append(BatchNorm(hiddens[i]))

    def forward(self, x):
        current_state = x
        self.input = x
        for i in range(self.nlayers):
            q = np.dot(current_state, self.W[i]) + self.b[i]
            if i in range(self.num_bn_layers):
                self.bn_layers[i].forward(q, not self.train_mode)
                q = self.bn_layers[i].out
            if i == 0:
                if self.apply_dropout:
                    if self.train_mode:
                        self.mask = np.random.binomial(1, self.dropout_ratio, size=q.shape)
                    else:
                        self.mask = 1 - self.dropout_ratio
                    q *= self.mask
            h = self.activations[i].forward(q)
            current_state = h
        return current_state

    def zero_grads(self):
        for i in range(self.nlayers):
            self.dW[i] = np.zeros_like(self.dW[i])
            self.db[i] = np.zeros_like(self.db[i])
        for i in range(self.num_bn_layers):
            self.bn_layers[i].dgamma = np.zeros_like(self.bn_layers[i].dgamma)
            self.bn_layers[i].dbeta = np.zeros_like(self.bn_layers[i].dbeta)

    def step(self):
        for i in range(self.num_bn_layers):
            self.bn_layers[i].gamma -= self.lr * self.bn_layers[i].dgamma
            self.bn_layers[i].beta -= self.lr * self.bn_layers[i].dbeta
        if not self.momentum:
            for i in range(self.nlayers):
                self.W[i] -= self.lr * self.dW[i] + self.lr * self.weight_decay * self.W[i]
                self.b[i] -= self.lr * self.db[i]
        else:
            for i in range(self.nlayers):
                self.vdw[i] = -self.lr * self.dW[i] + self.momentum * self.vdw[i]
                self.W[i] += self.vdw[i] - self.lr * self.weight_decay * self.W[i]
                self.vdb[i] = -self.lr * self.db[i] + self.momentum * self.vdb[i]
                self.b[i] += self.vdb[i]

    def backward(self, labels):
        if self.weight_decay:
            sum_of_square = 0
            for i in range(len(self.W)):
                sum_of_square += np.sum(np.square(self.W[i]))
        self.loss = self.criterion.forward(self.activations[-1].state, labels).mean()
        if self.weight_decay:
            self.loss += 0.5 * self.weight_decay * sum_of_square / len(labels[0])
        batch_size = float(self.activations[-1].state.shape[0])
        predict_label = np.argmax(self.activations[-1].state, axis=1)
        original_label = np.argmax(labels, axis=1)
        self.error = 0
        for i in range(len(predict_label)):
            if predict_label[i] != original_label[i]:
                self.error += 1
        self.error = float(self.error) / batch_size
        if not self.train_mode:
            return
        der = self.criterion.derivative()
        for i in range(self.nlayers - 1, -1, -1):
            if self.apply_dropout:
                if i == 0:
                    der *= self.mask       
            der = der * self.activations[i].derivative()
            if i < self.num_bn_layers:
                der = self.bn_layers[i].backward(der)
            if i == 0:
                self.dW[i] = np.dot(np.transpose(self.input), der) / batch_size
            else:
                self.dW[i] = np.dot(np.transpose(self.activations[i - 1].state), der) / batch_size
            self.db[i] = np.mean(der, axis=0)
            der = np.dot(der, np.transpose(self.W[i]))
        return der

    def W_matrix(self):
        return self.dW[0]

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False


def early_stopping(min_loss, current_loss, early_stop_index):
    if current_loss < min_loss:
        min_loss = current_loss
        early_stop_index = 0
    else:
        early_stop_index += 1
    if early_stop_index == 10:
        return True, min_loss, early_stop_index
    return False, min_loss, early_stop_index


def get_training_stats(mlp, dset, nepochs, batch_size, apply_early_stop=False):
    train, val, test = dset
    trainx, trainy = train
    valx, valy = val
    testx, testy = test

    idxs = np.arange(len(trainx))

    training_losses = []
    training_errors = []
    validation_losses = []
    validation_errors = []

    early_stop_index = 0
    min_loss = 100

    for e in range(nepochs):
        # Per epoch setup ...
        mlp.train()
        train_loss = []
        train_error = []
        for b in range(0, len(trainx), batch_size):
            mlp.zero_grads()
            mlp.forward(trainx[b:b + batch_size])
            mlp.backward(trainy[b:b + batch_size])
            mlp.step()
            train_loss.append(mlp.loss)
            train_error.append(mlp.error)
            if b + batch_size >= len(trainx):
                w_matrix = mlp.W_matrix()

        mlp.eval()
        val_loss = []
        val_error = []
        for b in range(0, len(valx), batch_size):
            mlp.forward(valx[b:b + batch_size])
            mlp.backward(valy[b:b + batch_size])
            val_loss.append(mlp.loss)
            val_error.append(mlp.error)
        if apply_early_stop:
            if e > 0:
                [is_early_stop, min_loss, early_stop_index] = \
                    early_stopping(min_loss, np.mean(val_loss), early_stop_index)
                if is_early_stop:
                    break
        # Accumulate data...
        training_losses.append(np.mean(train_loss))
        training_errors.append(np.mean(train_error))
        validation_losses.append(np.mean(val_loss))
        validation_errors.append(np.mean(val_error))

        # Shuffle
        np.random.shuffle(idxs)
        trainx, trainy = trainx[idxs], trainy[idxs]

    # Cleanup ...
    mlp.eval()
    test_loss = []
    test_error = []
    for b in range(0, len(testx), batch_size):
        mlp.forward(testx[b:b + batch_size])
        mlp.backward(testy[b:b + batch_size])
        test_loss.append(mlp.loss)
        test_error.append(mlp.error)
    test_losses = np.mean(test_loss)
    test_errors = np.mean(test_error)

    return training_losses, training_errors, validation_losses, validation_errors,\
    test_losses, test_errors, w_matrix, e+1


def get_data(fr, k):
    data = [inst.strip().split(',')for inst in fr.readlines()]
    data_label = [a[-1] for a in data]
    data = [data[i][:(len(data[0])-1)] for i in range(len(data))]
    data = np.array(data)
    data = data.astype(float)
    data_label = np.array(data_label)
    data_label = data_label.reshape((len(data_label),1))
    data_label = data_label.astype(float)
    label = []
    for value in data_label:
        p = [0 for _ in range(k)]
        p[int(value)] = 1
        label.append(p)
    label = np.array(label)
    label = label.astype(float)
    return data, label


# if __name__ == '__main__':
#     k = 19
#     train_fr = open('train.txt')
#     val_fr = open('val.txt')
#     test_fr = open('test.txt')
#     [train_data, train_label] = get_data(train_fr, k)
#     [val_data, val_label] = get_data(val_fr, k)
#     [test_data, test_label] = get_data(test_fr, k)


    # ############## best performing single-layer network ##############
    # start = time.clock()
    # mlp_single = MLP(1568, 19, [400], [Sigmoid(), Identity()],
    #           random_normal_weight_init, zeros_bias_init,
    #           SoftmaxCrossEntropy(), lr=0.1, momentum=0.9,
    #           num_bn_layers=0, weight_decay=0.0001,
    #           apply_dropout=False, dropout_ratio=0.5)
    # [training_losses, training_errors, validation_losses,
    #  validation_errors, test_losses, test_errors, w_matrix, epoch] \
    #  = get_training_stats(mlp_single, ((train_data, train_label),
    #     (val_data, val_label), (test_data, test_label)),
    #     nepochs=150, batch_size=32, apply_early_stop=True)
    # end = time.clock()


    # ############## best performing 2-layer network ##############
    # start = time.clock()
    # mlp = MLP(1568, 19, [200, 100], [Sigmoid(), Sigmoid(), Identity()],
    #           random_normal_weight_init, zeros_bias_init,
    #           SoftmaxCrossEntropy(), lr=0.1, momentum=0.9,
    #           num_bn_layers=0, weight_decay=0.0001,
    #           apply_dropout=False, dropout_ratio=0.5)
    # [training_losses, training_errors, validation_losses,
    #  validation_errors, test_losses, test_errors, w_matrix, epoch] \
    #  = get_training_stats(mlp, ((train_data, train_label),
    #     (val_data, val_label),(test_data, test_label)),
    #     nepochs=150, batch_size=32, apply_early_stop=True)
    # end = time.clock()
    # print(end - start)
    # print(epoch)
    # print(training_losses)
    # print(training_errors)
    # print(validation_losses)
    # print(validation_errors)
    # print(test_losses)
    # print(test_errors)


