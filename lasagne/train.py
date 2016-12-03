import argparse
import cPickle
import time
import os
import logging
import numpy as np
import theano as th
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import lasagne
import lasagne.layers as ll
from lasagne.layers import dnn
import nn
logging.basicConfig(level=logging.INFO)

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--norm_type', default='no_norm', type=str)
parser.add_argument('--learning_rate', default=0.001, type=float)
args = parser.parse_args()
logging.info(args)
        
# fixed random seeds
rng = np.random.RandomState(args.seed)
theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))

# setup output
time_str = time.strftime("%m-%d-%H-%M", time.gmtime())
exp_dir = args.norm_type + "_" + time_str + "_" + "{}".format(args.learning_rate).replace(".", "p")
try:
    os.stat(exp_dir)
except:
    os.makedirs(exp_dir)
logging.info("OPENING " + exp_dir + '/results.csv')
results_file = open(exp_dir + '/results.csv', 'w')
results_file.write('epoch, time, train_error, test_error\n')
results_file.flush()

# load CIFAR-10 data
def unpickle(file):
    fo = open(file, 'rb')
    d = cPickle.load(fo)
    fo.close()
    return {'x': np.cast[th.config.floatX]((-127.5 + d['data'].reshape((10000,3,32,32)))/128.), 'y': np.array(d['labels']).astype(np.uint8)}

train_data = [unpickle('/home/ubuntu/data/cifar-10-python/cifar-10-batches-py/data_batch_' + str(i)) for i in range(1,6)]
trainx = np.concatenate([d['x'] for d in train_data],axis=0)
trainy = np.concatenate([d['y'] for d in train_data])
test_data = unpickle('/home/ubuntu/data/cifar-10-python/cifar-10-batches-py/test_batch')
testx = test_data['x']
testy = test_data['y']
nr_batches_train = int(trainx.shape[0]/args.batch_size)
nr_batches_test = int(testx.shape[0]/args.batch_size)

# whitening
whitener = nn.ZCA(x=trainx)
trainx_white = whitener.apply(trainx)
testx_white = whitener.apply(testx)

# specify model
if args.norm_type=='weight_norm':
    normalizer = lambda l: nn.weight_norm(l)
elif args.norm_type=='batch_norm':
    normalizer = lambda l: nn.batch_norm(l)
elif args.norm_type=='mean_only_bn':
    normalizer = lambda l: nn.mean_only_bn(l)
elif args.norm_type=='no_norm':
    normalizer = lambda l: nn.no_norm(l)
else:
    raise NotImplementedError('incorrect norm type')

layers = [ll.InputLayer(shape=(None, 3, 32, 32))]
layers.append(ll.GaussianNoiseLayer(layers[-1], sigma=0.15))
layers.append(normalizer(dnn.Conv2DDNNLayer(layers[-1], 96, (3,3), pad=1, nonlinearity=nn.lrelu)))
layers.append(normalizer(dnn.Conv2DDNNLayer(layers[-1], 96, (3,3), pad=1, nonlinearity=nn.lrelu)))
layers.append(normalizer(dnn.Conv2DDNNLayer(layers[-1], 96, (3,3), pad=1, nonlinearity=nn.lrelu)))
layers.append(ll.MaxPool2DLayer(layers[-1], 2))
layers.append(ll.DropoutLayer(layers[-1], p=0.5))
layers.append(normalizer(dnn.Conv2DDNNLayer(layers[-1], 192, (3,3), pad=1, nonlinearity=nn.lrelu)))
layers.append(normalizer(dnn.Conv2DDNNLayer(layers[-1], 192, (3,3), pad=1, nonlinearity=nn.lrelu)))
layers.append(normalizer(dnn.Conv2DDNNLayer(layers[-1], 192, (3,3), pad=1, nonlinearity=nn.lrelu)))
layers.append(ll.MaxPool2DLayer(layers[-1], 2))
layers.append(ll.DropoutLayer(layers[-1], p=0.5))
layers.append(normalizer(dnn.Conv2DDNNLayer(layers[-1], 192, (3,3), pad=0, nonlinearity=nn.lrelu)))
layers.append(normalizer(ll.NINLayer(layers[-1], num_units=192, nonlinearity=nn.lrelu)))
layers.append(normalizer(ll.NINLayer(layers[-1], num_units=192, nonlinearity=nn.lrelu)))
layers.append(nn.GlobalAvgLayer(layers[-1]))
layers.append(normalizer(ll.DenseLayer(layers[-1], num_units=10, nonlinearity=None)))

# initialization
x = T.tensor4()
temp = ll.get_output(layers[-1], x, init=True)
init_updates = [u for l in layers for u in getattr(l,'init_updates',[])]

# discriminative cost & updates
output_before_softmax = ll.get_output(layers[-1], x)
bn_updates = [u for l in layers for u in getattr(l,'bn_updates',[])]
y = T.ivector()
cost = nn.softmax_loss(y, output_before_softmax)
train_err = T.mean(T.neq(T.argmax(output_before_softmax,axis=1),y))
params = ll.get_all_params(layers, trainable=True)
lr = T.scalar()
mom1 = T.scalar()
param_updates = nn.adam_updates(params, cost, lr=lr, mom1=mom1)

test_output_before_softmax = ll.get_output(layers[-1], x, deterministic=True)
test_err = T.mean(T.neq(T.argmax(test_output_before_softmax,axis=1),y))

# compile Theano functions
train_batch = th.function(inputs=[x,y,lr,mom1], outputs=train_err, updates=param_updates+bn_updates)
test_batch = th.function(inputs=[x,y], outputs=test_err)
initfun = th.function(inputs=[x], outputs=None, updates=init_updates, on_unused_input='ignore')

# //////////// perform training //////////////
begin_all = time.time()
for epoch in range(200):
    begin_epoch = time.time()
    lr = np.cast[th.config.floatX](args.learning_rate * np.minimum(2. - epoch/100., 1.))
    if epoch<100:
        mom1 = 0.9
    else:
        mom1 = 0.5
    
    # permute the training data
    inds = rng.permutation(trainx_white.shape[0])
    trainx_white = trainx_white[inds]
    trainy = trainy[inds]
    
    # init params if first epoch
    if epoch==0:
        initfun(trainx_white[:500])

    # train
    train_err = 0.
    for t in range(nr_batches_train):
        train_err += train_batch(trainx_white[t*args.batch_size:(t+1)*args.batch_size],
                                             trainy[t*args.batch_size:(t+1)*args.batch_size],lr,mom1)
    train_err /= nr_batches_train
    
    # test
    test_err = 0.
    for t in range(nr_batches_test):
        test_err += test_batch(testx_white[t*args.batch_size:(t+1)*args.batch_size],testy[t*args.batch_size:(t+1)*args.batch_size])
    test_err /= nr_batches_test
    
    logging.info('Iteration %d, time = %ds, train_err = %.4f, test_err = %.4f' % (epoch, time.time()-begin_epoch, train_err, test_err))
    results_file.write('%d, %d, %.4f, %.4f\n' % (epoch, time.time()-begin_all, train_err, test_err))
    results_file.flush()

