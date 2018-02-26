from mpi4py import MPI
import tensorflow as tf
import numpy as np

def create_network():
    value_in = tf.placeholder(tf.float32, [None, 32])
    value_hidden = tf.layers.dense(value_in, 32)
    value_out = tf.layers.dense(value_hidden, 1)
    return value_in, value_out, value_hidden

def var_shape(x):
    out = x.get_shape().as_list()
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out

def numel(x):
    return intprod(var_shape(x))

def intprod(x):
    return int(np.prod(x))


class GetFlat(object):
    def __init__(self, var_list, sess):
        self.op = tf.concat(axis=0, values=[tf.reshape(v, [numel(v)]) for v in var_list])
        self.sess = sess

    def __call__(self):
        return self.sess.run(self.op)

if __name__ == '__main__':
    v_in, v_out, v_hidden = create_network()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    all_variable = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
    all_variable = GetFlat(all_variable, sess)()
    rank = MPI.COMM_WORLD.Get_rank()
    print('before')
    print(rank, all_variable[0])
    MPI.COMM_WORLD.Bcast(all_variable, root=0)
    print('after')
    print(rank, all_variable[0])

