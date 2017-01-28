from keras import backend as K
from keras.optimizers import SGD,Adam
import tensorflow as tf

# adapted from keras.optimizers.SGD
class SGDWithWeightnorm(SGD):
    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = []

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))
            self.updates .append(K.update_add(self.iterations, 1))

        # momentum
        shapes = [K.get_variable_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):

            # if a weight tensor (len > 1) use weight normalized parameterization
            ps = K.get_variable_shape(p)
            if len(ps) > 1:

                # get weight normalization parameters
                V, V_norm, V_scaler, g_param, grad_g, grad_V = get_weightnorm_params_and_grads(p, g)

                # momentum container for the 'g' parameter
                V_scaler_shape = K.get_variable_shape(V_scaler)
                m_g = K.zeros(V_scaler_shape)

                # update g parameters
                v_g = self.momentum * m_g - lr * grad_g  # velocity
                self.updates.append(K.update(m_g, v_g))
                if self.nesterov:
                    new_g_param = g_param + self.momentum * v_g - lr * grad_g
                else:
                    new_g_param = g_param + v_g

                # update V parameters
                v_v = self.momentum * m - lr * grad_V  # velocity
                self.updates.append(K.update(m, v_v))
                if self.nesterov:
                    new_V_param = V + self.momentum * v_v - lr * grad_V
                else:
                    new_V_param = V + v_v

                # if there are constraints we apply them to V, not W
                if p in constraints:
                    c = constraints[p]
                    new_V_param = c(new_V_param)

                # wn param updates --> W updates
                add_weightnorm_param_updates(self.updates, new_V_param, new_g_param, p, V_scaler)

            else: # normal SGD with momentum
                v = self.momentum * m - lr * g  # velocity
                self.updates.append(K.update(m, v))

                if self.nesterov:
                    new_p = p + self.momentum * v - lr * g
                else:
                    new_p = p + v

                # apply constraints
                if p in constraints:
                    c = constraints[p]
                    new_p = c(new_p)

                self.updates.append(K.update(p, new_p))
        return self.updates

# adapted from keras.optimizers.Adam
class AdamWithWeightnorm(Adam):
    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        t = self.iterations + 1
        lr_t = lr * K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t))

        shapes = [K.get_variable_shape(p) for p in params]
        ms = [K.zeros(shape) for shape in shapes]
        vs = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):

            # if a weight tensor (len > 1) use weight normalized parameterization
            # this is the only part changed w.r.t. keras.optimizers.Adam
            ps = K.get_variable_shape(p)
            if len(ps)>1:

                # get weight normalization parameters
                V, V_norm, V_scaler, g_param, grad_g, grad_V = get_weightnorm_params_and_grads(p, g)

                # Adam containers for the 'g' parameter
                V_scaler_shape = K.get_variable_shape(V_scaler)
                m_g = K.zeros(V_scaler_shape)
                v_g = K.zeros(V_scaler_shape)

                # update g parameters
                m_g_t = (self.beta_1 * m_g) + (1. - self.beta_1) * grad_g
                v_g_t = (self.beta_2 * v_g) + (1. - self.beta_2) * K.square(grad_g)
                new_g_param = g_param - lr_t * m_g_t / (K.sqrt(v_g_t) + self.epsilon)
                self.updates.append(K.update(m_g, m_g_t))
                self.updates.append(K.update(v_g, v_g_t))

                # update V parameters
                m_t = (self.beta_1 * m) + (1. - self.beta_1) * grad_V
                v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(grad_V)
                new_V_param = V - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)
                self.updates.append(K.update(m, m_t))
                self.updates.append(K.update(v, v_t))

                # if there are constraints we apply them to V, not W
                if p in constraints:
                    c = constraints[p]
                    new_V_param = c(new_V_param)

                # wn param updates --> W updates
                add_weightnorm_param_updates(self.updates, new_V_param, new_g_param, p, V_scaler)

            else: # do optimization normally
                m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
                v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

                self.updates.append(K.update(m, m_t))
                self.updates.append(K.update(v, v_t))

                new_p = p_t
                # apply constraints
                if p in constraints:
                    c = constraints[p]
                    new_p = c(new_p)
                self.updates.append(K.update(p, new_p))
        return self.updates


def get_weightnorm_params_and_grads(p, g):
    ps = K.get_variable_shape(p)

    # construct weight scaler: V_scaler = g/||V||
    V_scaler_shape = (ps[-1],)  # assumes we're using tensorflow!
    V_scaler = K.ones(V_scaler_shape)  # init to ones, so effective parameters don't change

    # get V parameters = ||V||/g * W
    norm_axes = [i for i in range(len(ps) - 1)]
    V = p / tf.reshape(V_scaler, [1] * len(norm_axes) + [-1])

    # split V_scaler into ||V|| and g parameters
    V_norm = tf.sqrt(tf.reduce_sum(tf.square(V), norm_axes))
    g_param = V_scaler * V_norm

    # get grad in V,g parameters
    grad_g = tf.reduce_sum(g * V, norm_axes) / V_norm
    grad_V = tf.reshape(V_scaler, [1] * len(norm_axes) + [-1]) * \
             (g - tf.reshape(grad_g / V_norm, [1] * len(norm_axes) + [-1]) * V)

    return V, V_norm, V_scaler, g_param, grad_g, grad_V


def add_weightnorm_param_updates(updates, new_V_param, new_g_param, W, V_scaler):
    ps = K.get_variable_shape(new_V_param)
    norm_axes = [i for i in range(len(ps) - 1)]

    # update W and V_scaler
    new_V_norm = tf.sqrt(tf.reduce_sum(tf.square(new_V_param), norm_axes))
    new_V_scaler = new_g_param / new_V_norm
    new_W = tf.reshape(new_V_scaler, [1] * len(norm_axes) + [-1]) * new_V_param
    updates.append(K.update(W, new_W))
    updates.append(K.update(V_scaler, new_V_scaler))


# data based initialization for a given Keras model
def data_based_init(model, input):

    # input can be dict, numpy array, or list of numpy arrays
    if type(input) is dict:
        feed_dict = input
    elif type(input) is list:
        feed_dict = {tf_inp: np_inp for tf_inp,np_inp in zip(model.inputs,input)}
    else:
        feed_dict = {model.inputs[0]: input}

    # add learning phase if required
    if model.uses_learning_phase and K.learning_phase() not in feed_dict:
        feed_dict.update({K.learning_phase(): 1})

    # get all layer name, output, weight, bias tuples
    layer_output_weight_bias = []
    for l in model.layers:
        if hasattr(l, 'W') and hasattr(l, 'b'):
            assert(l.built)
            layer_output_weight_bias.append( (l.name,l.get_output_at(0),l.W,l.b) ) # if more than one node, only use the first

    # iterate over our list and do data dependent init
    sess = K.get_session()
    for l,o,W,b in layer_output_weight_bias:
        print('Performing data dependent initialization for layer ' + l)
        m,v = tf.nn.moments(o, [i for i in range(len(o.get_shape())-1)])
        s = tf.sqrt(v + 1e-10)
        updates = tf.group(W.assign(W/tf.reshape(s,[1]*(len(W.get_shape())-1)+[-1])), b.assign((b-m)/s))
        sess.run(updates, feed_dict)
