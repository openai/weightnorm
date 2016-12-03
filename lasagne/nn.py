import numpy as np
import theano as th
import theano.tensor as T
from scipy import linalg
import lasagne

class ZCA(object):
    def __init__(self, regularization=1e-5, x=None):
        self.regularization = regularization
        if x is not None:
            self.fit(x)

    def fit(self, x):
        s = x.shape
        x = x.copy().reshape((s[0],np.prod(s[1:])))
        m = np.mean(x, axis=0)
        x -= m
        sigma = np.dot(x.T,x) / x.shape[0]
        U, S, V = linalg.svd(sigma)
        tmp = np.dot(U, np.diag(1./np.sqrt(S+self.regularization)))
        tmp2 = np.dot(U, np.diag(np.sqrt(S+self.regularization)))
        self.ZCA_mat = th.shared(np.dot(tmp, U.T).astype(th.config.floatX))
        self.inv_ZCA_mat = th.shared(np.dot(tmp2, U.T).astype(th.config.floatX))
        self.mean = th.shared(m.astype(th.config.floatX))

    def apply(self, x):
        s = x.shape
        if isinstance(x, np.ndarray):
            return np.dot(x.reshape((s[0],np.prod(s[1:]))) - self.mean.get_value(), self.ZCA_mat.get_value()).reshape(s)
        elif isinstance(x, T.TensorVariable):
            return T.dot(x.flatten(2) - self.mean.dimshuffle('x',0), self.ZCA_mat).reshape(s)
        else:
            raise NotImplementedError("Whitening only implemented for numpy arrays or Theano TensorVariables")
            
    def invert(self, x):
        s = x.shape
        if isinstance(x, np.ndarray):
            return (np.dot(x.reshape((s[0],np.prod(s[1:]))), self.inv_ZCA_mat.get_value()) + self.mean.get_value()).reshape(s)
        elif isinstance(x, T.TensorVariable):
            return (T.dot(x.flatten(2), self.inv_ZCA_mat) + self.mean.dimshuffle('x',0)).reshape(s)
        else:
            raise NotImplementedError("Whitening only implemented for numpy arrays or Theano TensorVariables")

# T.nnet.relu has some issues with very large inputs, this is more stable
def relu(x):
    return T.maximum(x, 0)
    
def lrelu(x, a=0.1):
    return T.maximum(x, a*x)
    
def log_sum_exp(x, axis=1):
    m = T.max(x, axis=axis)
    return m+T.log(T.sum(T.exp(x-m.dimshuffle(0,'x')), axis=axis))

def adamax_updates(params, cost, lr=0.001, mom1=0.9, mom2=0.999):
    updates = []
    grads = T.grad(cost, params)
    for p, g in zip(params, grads):
        mg = th.shared(np.cast[th.config.floatX](p.get_value() * 0.))
        v = th.shared(np.cast[th.config.floatX](p.get_value() * 0.))
        if mom1>0:
            v_t = mom1*v + (1. - mom1)*g
            updates.append((v,v_t))
        else:
            v_t = g
        mg_t = T.maximum(mom2*mg, abs(g))
        g_t = v_t / (mg_t + 1e-6)
        p_t = p - lr * g_t
        updates.append((mg, mg_t))
        updates.append((p, p_t))
    return updates

def adam_updates(params, cost, lr=0.001, mom1=0.9, mom2=0.999):
    updates = []
    grads = T.grad(cost, params)
    t = th.shared(np.cast[th.config.floatX](1.))
    for p, g in zip(params, grads):
        v = th.shared(np.cast[th.config.floatX](p.get_value() * 0.))
        mg = th.shared(np.cast[th.config.floatX](p.get_value() * 0.))
        v_t = mom1*v + (1. - mom1)*g
        mg_t = mom2*mg + (1. - mom2)*T.square(g)
        v_hat = v_t / (1. - mom1 ** t)
        mg_hat = mg_t / (1. - mom2 ** t)
        g_t = v_hat / T.sqrt(mg_hat + 1e-8)
        p_t = p - lr * g_t
        updates.append((v, v_t))
        updates.append((mg, mg_t))
        updates.append((p, p_t))
    updates.append((t, t+1))
    return updates
    
def softmax_loss(p_true, output_before_softmax):
    output_before_softmax -= T.max(output_before_softmax, axis=1, keepdims=True)
    if p_true.ndim==2:
        return T.mean(T.log(T.sum(T.exp(output_before_softmax),axis=1)) - T.sum(p_true*output_before_softmax, axis=1))
    else:
        return T.mean(T.log(T.sum(T.exp(output_before_softmax),axis=1)) - output_before_softmax[T.arange(p_true.shape[0]),p_true])

class BatchNormLayer(lasagne.layers.Layer):
    def __init__(self, incoming, b=lasagne.init.Constant(0.), g=lasagne.init.Constant(1.), 
                 W=lasagne.init.Normal(0.05), nonlinearity=relu, **kwargs):
        super(BatchNormLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = nonlinearity
        k = self.input_shape[1]
        if b is not None:
            self.b = self.add_param(b, (k,), name="b", regularizable=False)
        if g is not None:
            self.g = self.add_param(g, (k,), name="g")
        self.avg_batch_mean = self.add_param(lasagne.init.Constant(0.), (k,), name="avg_batch_mean", regularizable=False, trainable=False)
        self.avg_batch_var = self.add_param(lasagne.init.Constant(1.), (k,), name="avg_batch_var", regularizable=False, trainable=False)
        incoming.W.set_value(W.sample(incoming.W.get_value().shape))
        if len(self.input_shape)==4:
            self.axes_to_sum = (0,2,3)
            self.dimshuffle_args = ['x',0,'x','x']
        else:
            self.axes_to_sum = 0
            self.dimshuffle_args = ['x',0]

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic:
            norm_features = (input-self.avg_batch_mean.dimshuffle(*self.dimshuffle_args)) / T.sqrt(1e-6 + self.avg_batch_var).dimshuffle(*self.dimshuffle_args)
        else:
            batch_mean = T.mean(input,axis=self.axes_to_sum).flatten()
            centered_input = input-batch_mean.dimshuffle(*self.dimshuffle_args)
            batch_var = T.mean(T.square(centered_input),axis=self.axes_to_sum).flatten()
            batch_stdv = T.sqrt(1e-6 + batch_var)
            norm_features = centered_input / batch_stdv.dimshuffle(*self.dimshuffle_args)
            
            # BN updates
            new_m = 0.9*self.avg_batch_mean + 0.1*batch_mean
            new_v = 0.9*self.avg_batch_var + T.cast((0.1*input.shape[0])/(input.shape[0]-1.), th.config.floatX)*batch_var
            self.bn_updates = [(self.avg_batch_mean, new_m), (self.avg_batch_var, new_v)]
            
        if hasattr(self, 'g'):
            activation = norm_features*self.g.dimshuffle(*self.dimshuffle_args)
        else:
            activation = norm_features
        if hasattr(self, 'b'):
            activation += self.b.dimshuffle(*self.dimshuffle_args)
            
        return self.nonlinearity(activation)
        
def batch_norm(layer, b=lasagne.init.Constant(0.), g=lasagne.init.Constant(1.), **kwargs):
    """
    adapted from https://gist.github.com/f0k/f1a6bd3c8585c400c190
    """
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = lasagne.nonlinearities.identity
    if hasattr(layer, 'b'):
        del layer.params[layer.b]
        layer.b = None
    return BatchNormLayer(layer, b, g, nonlinearity=nonlinearity, **kwargs)

class GlobalAvgLayer(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(GlobalAvgLayer, self).__init__(incoming, **kwargs)
    def get_output_for(self, input, **kwargs):
        return T.mean(input, axis=(2,3))
    def get_output_shape_for(self, input_shape):
        return input_shape[:2]
        
class MeanOnlyBNLayer(lasagne.layers.Layer):
    def __init__(self, incoming, b=lasagne.init.Constant(0.), g=lasagne.init.Constant(1.),
                 W=lasagne.init.Normal(0.05), nonlinearity=relu, **kwargs):
        super(MeanOnlyBNLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = nonlinearity
        k = self.input_shape[1]
        if b is not None:
            self.b = self.add_param(b, (k,), name="b", regularizable=False)
        if g is not None:
            self.g = self.add_param(g, (k,), name="g")
        self.avg_batch_mean = self.add_param(lasagne.init.Constant(0.), (k,), name="avg_batch_mean", regularizable=False, trainable=False)
        if len(self.input_shape)==4:
            self.axes_to_sum = (0,2,3)
            self.dimshuffle_args = ['x',0,'x','x']
        else:
            self.axes_to_sum = 0
            self.dimshuffle_args = ['x',0]
        
        # scale weights in layer below
        incoming.W_param = incoming.W
        incoming.W_param.set_value(W.sample(incoming.W_param.get_value().shape))
        if incoming.W_param.ndim==4:
            W_axes_to_sum = (1,2,3)
            W_dimshuffle_args = [0,'x','x','x']
        else:
            W_axes_to_sum = 0
            W_dimshuffle_args = ['x',0]
        if g is not None:
            incoming.W = incoming.W_param * (self.g/T.sqrt(T.sum(T.square(incoming.W_param),axis=W_axes_to_sum))).dimshuffle(*W_dimshuffle_args)
        else:
            incoming.W = incoming.W_param / T.sqrt(T.sum(T.square(incoming.W_param),axis=W_axes_to_sum,keepdims=True))        

    def get_output_for(self, input, deterministic=False, init=False, **kwargs):
        if deterministic:
            activation = input - self.avg_batch_mean.dimshuffle(*self.dimshuffle_args)
        else:
            m = T.mean(input,axis=self.axes_to_sum)
            activation = input - m.dimshuffle(*self.dimshuffle_args)
            self.bn_updates = [(self.avg_batch_mean, 0.9*self.avg_batch_mean + 0.1*m)]
            if init:
                stdv = T.sqrt(T.mean(T.square(activation),axis=self.axes_to_sum))
                activation /= stdv.dimshuffle(*self.dimshuffle_args)
                self.init_updates = [(self.g, self.g/stdv)]                
        if hasattr(self, 'b'):
            activation += self.b.dimshuffle(*self.dimshuffle_args)
            
        return self.nonlinearity(activation)
        
def mean_only_bn(layer, **kwargs):
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = lasagne.nonlinearities.identity
    if hasattr(layer, 'b'):
        del layer.params[layer.b]
        layer.b = None
    return MeanOnlyBNLayer(layer, nonlinearity=nonlinearity, **kwargs)

class WeightNormLayer(lasagne.layers.Layer):
    def __init__(self, incoming, b=lasagne.init.Constant(0.), g=lasagne.init.Constant(1.),
                 W=lasagne.init.Normal(0.05), nonlinearity=relu, **kwargs):
        super(WeightNormLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = nonlinearity
        k = self.input_shape[1]
        if b is not None:
            self.b = self.add_param(b, (k,), name="b", regularizable=False)
        if g is not None:
            self.g = self.add_param(g, (k,), name="g")
        if len(self.input_shape)==4:
            self.axes_to_sum = (0,2,3)
            self.dimshuffle_args = ['x',0,'x','x']
        else:
            self.axes_to_sum = 0
            self.dimshuffle_args = ['x',0]
        
        # scale weights in layer below
        incoming.W_param = incoming.W
        incoming.W_param.set_value(W.sample(incoming.W_param.get_value().shape))
        if incoming.W_param.ndim==4:
            W_axes_to_sum = (1,2,3)
            W_dimshuffle_args = [0,'x','x','x']
        else:
            W_axes_to_sum = 0
            W_dimshuffle_args = ['x',0]
        if g is not None:
            incoming.W = incoming.W_param * (self.g/T.sqrt(T.sum(T.square(incoming.W_param),axis=W_axes_to_sum))).dimshuffle(*W_dimshuffle_args)
        else:
            incoming.W = incoming.W_param / T.sqrt(T.sum(T.square(incoming.W_param),axis=W_axes_to_sum,keepdims=True))        

    def get_output_for(self, input, init=False, **kwargs):
        if init:
            m = T.mean(input, self.axes_to_sum)
            input -= m.dimshuffle(*self.dimshuffle_args)
            stdv = T.sqrt(T.mean(T.square(input),axis=self.axes_to_sum))
            input /= stdv.dimshuffle(*self.dimshuffle_args)
            self.init_updates = [(self.b, -m/stdv), (self.g, self.g/stdv)]
        elif hasattr(self,'b'):
            input += self.b.dimshuffle(*self.dimshuffle_args)
            
        return self.nonlinearity(input)
        
def weight_norm(layer, **kwargs):
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = lasagne.nonlinearities.identity
    if hasattr(layer, 'b'):
        del layer.params[layer.b]
        layer.b = None
    return WeightNormLayer(layer, nonlinearity=nonlinearity, **kwargs)
    
class InitLayer(lasagne.layers.Layer):
    def __init__(self, incoming, b=lasagne.init.Constant(0.), g=lasagne.init.Constant(1.), nonlinearity=relu, **kwargs):
        super(InitLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = nonlinearity
        k = self.input_shape[1]
        if b is not None:
            self.b = self.add_param(b, (k,), name="b", regularizable=False)
        if g is not None:
            self.g = self.add_param(g, (k,), name="g", regularizable=False, trainable=False)
        if len(self.input_shape)==4:
            self.axes_to_sum = (0,2,3)
            self.dimshuffle_args = ['x',0,'x','x']
        else:
            self.axes_to_sum = 0
            self.dimshuffle_args = ['x',0]
        
        # scale weights in layer below
        incoming.W_param = incoming.W
        if incoming.W_param.ndim==4:
            W_dimshuffle_args = [0,'x','x','x']
        else:
            W_dimshuffle_args = ['x',0]
        incoming.W = self.g.dimshuffle(*W_dimshuffle_args) * incoming.W_param

    def get_output_for(self, input, init=False, **kwargs):
        if init:
            m = T.mean(input, self.axes_to_sum)
            input -= m.dimshuffle(*self.dimshuffle_args)
            stdv = T.sqrt(T.mean(T.square(input),axis=self.axes_to_sum))
            input /= stdv.dimshuffle(*self.dimshuffle_args)
            self.init_updates = [(self.b, -m/stdv), (self.g, self.g/stdv)]
        elif hasattr(self,'b'):
            input += self.b.dimshuffle(*self.dimshuffle_args)
            
        return self.nonlinearity(input)
        
def no_norm(layer, **kwargs):
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = lasagne.nonlinearities.identity
    if hasattr(layer, 'b'):
        del layer.params[layer.b]
        layer.b = None
    return InitLayer(layer, nonlinearity=nonlinearity, **kwargs)
