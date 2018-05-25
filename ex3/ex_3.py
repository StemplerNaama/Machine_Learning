import random

import numpy as np
def sigmoid(x):
    return 1 / (1+np.exp(-x))

def tanh(x):
    return  np.tanh(x)

def tanhDerative(x):
    return 1.0 -np.tanh(x)**2

def reLu(x):
    return np.maximum(0,x)

def reLuDerative(x):
    x[x<=0] =0
    x[x>0] = 1
    return x

#update the weights
def update_weights_sgd(params, gradients, lr):
    db1, dW1, db2, dW2 = [gradients[key] for key in ('b1', 'W1', 'b2', 'W2')]
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    return {'W1': w1, 'b1': b1, 'W2': w2, 'b2': b2}

def backPropagatoin(fprop_cache):
  x, y, z1, h1, z2, h2, loss = [fprop_cache[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'h2', 'loss')]
  eyeMat = np.eye(10)[int(y)]
  dz2 = (h2- eyeMat)
  dW2 = np.dot(np.array([dz2]).T, np.array([h1]))
  db2 = dz2
  dz1 = np.dot(fprop_cache['W2'].T,dz2) * h1 * (1-h1)
  dW1 = np.dot(np.array([dz1]).T, np.array([x]))
  db1 = dz1
  return {'b1': db1, 'W1': dW1, 'b2': db2, 'W2': dW2}


def softmax(x):
    op= np.exp(x-np.amax(x))
    return op/op.sum()


def Loss_fun(out,y):
    ret = np.matrix(-1*np.log(out[int(y)]))
    return ret[0,0]

#the model is learning
def forward(params, x,y):
  W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
  z1 = np.dot(W1, x) + b1
  h1 = sigmoid(z1)
  z2 = np.dot(W2, h1) + b2
  # h2 its our yTag
  h2 = softmax(z2)
  loss = Loss_fun(h2,y)
  ret = {'x': x, 'y': y, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'loss': loss}
  for key in params:
    ret[key] = params[key]
  return ret


def train(params, epochs, lr, arrZip, dev_x, dev_y):
    #for every epoch
    for i in xrange(epochs):
        sum_loss = 0.0
        #shuffle train examples
        np.random.shuffle(arrZip)
        for x, y in arrZip:
            retForward = forward(params, x, y)
            sum_loss += retForward['loss']
            gradients = backPropagatoin(retForward)
            params = update_weights_sgd(params, gradients, lr)
        dev_loss, acc = predict_on_dev(params, dev_x, dev_y)
        checkX, chaeckY = zip(*list(arrZip))
        # print the outcome of the validation data
        print i, sum_loss / np.matrix(checkX).shape[0], dev_loss, "{}%".format(acc * 100)
    return params



def predict_on_dev(params, dev_x, dev_y):
    #good will counts how many times gettig correct tag
    sum_loss = good = 0.0
    arrZip = zip(dev_x, dev_y)
    for x, y in (arrZip):
        out = forward(params, x, y)
        sum_loss += out['loss']
        if out['h2'].argmax() == y:
            good += 1
    acc = good /  np.matrix(dev_x).shape[0]
    # the avg of loss
    avg_loss = sum_loss / np.matrix(dev_x).shape[0]
    return avg_loss, acc

#forward without compute loss function
def testForward(params, x):
  W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
  z1 = np.dot(W1, x) + b1
  h1 = sigmoid(z1)
  z2 = np.dot(W2, h1) + b2
  h2 = softmax(z2)
  return h2

def cheackTest(params):
    f = open("test.pred", "w")
    for x in test_x:
       out = testForward(params, x)
       f.write(str(out.argmax())+'\n')
    f.close()

hiddenSize = 110
train_x = np.loadtxt("train_x")
train_y = np.loadtxt("train_y")
test_x = np.loadtxt("test_x")
#normelize the data
train_x = [x/225.0 for x in train_x]
arrZip = list(zip(train_x, train_y))
random.shuffle(arrZip)
train_x, train_y = zip(*arrZip)
dev_size = len(train_y)*0.8
dev_size = int(dev_size)
dev_x, dev_y = train_x[dev_size :], train_y[dev_size :]
train_x, train_y = train_x[: dev_size], train_y[: dev_size]
arrZip = zip(train_x, train_y)
w1 = (np.random.rand(hiddenSize,784) - .5) * .1
w2 = (np.random.rand(10,hiddenSize) - .5) * .1
b1 = (np.random.rand(hiddenSize) - .5) * .1
b2 = (np.random.rand(10) - .5) * .1
params = {'W1':w1, 'b1' : b1, 'W2':w2, 'b2' : b2}
epochs = 70
updateParams = train(params,epochs, 0.01, arrZip,dev_x,dev_y)
cheackTest(updateParams)