from mpi_running_mean_std import RunningMeanStd
import tf_util as U
import tensorflow as tf
import gym
from distributions import make_pdtype
import numpy as np


class MlpDistTwolayersPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, control_space, hid_size, num_hid_layers, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        c_in = U.get_placeholder(name="c_in", dtype=tf.float32, shape=[sequence_length, control_space])
     
        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "vf1fc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        vpred1 = U.dense(last_out, 1, "vf1final", weight_init=U.normc_initializer(1.0))[:,0]
        
        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "pol1fc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean1 = U.dense(last_out, pdtype.param_shape()[0]//2, "pol1final", U.normc_initializer(0.01))   
            print np.shape(mean1)         
            # logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
            # pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)
        else:
            pdparam = U.dense(last_out, pdtype.param_shape()[0], "pol1final", U.normc_initializer(0.01))
        ######

        # Above, first layer
        # Below, second layer

        ######
        last_out = U.concatenate([obz, c_in], axis = 1) 
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "vf2fc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        vpred2 = U.dense(last_out, 1, "vf2final", weight_init=U.normc_initializer(1.0))[:,0]
        
        last_out = U.concatenate([obz, c_in], axis = 1) 
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "pol2fc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean2 = U.dense(last_out, pdtype.param_shape()[0]//2, "pol2final", U.normc_initializer(0.01))    
            print np.shape(mean2)         
            # logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
            # pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)
        else:
            pdparam = U.dense(last_out, pdtype.param_shape()[0], "pol2final", U.normc_initializer(0.01))

        self.vpred = U.sum([vpred1,vpred2], axis = 0)
        print np.shape(vpred1)
        print np.shape(vpred2)
        # print np.shape(self.vpred1)
        mean        = U.sum([mean1, mean2], axis = 0)
        print np.shape(mean)
        logstd      = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
        pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(name = "stochastic", dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        # self.opname = ac.name

        self._act = U.function([stochastic, ob, c_in], [ac, self.vpred])

    def act(self, stochastic, ob, c_in):
        ac1, vpred1 =  self._act(stochastic, ob[None], c_in[None])
        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

