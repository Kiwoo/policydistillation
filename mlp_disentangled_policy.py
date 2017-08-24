from mpi_running_mean_std import RunningMeanStd
import tf_util as U
import tensorflow as tf
import gym
from distributions import make_pdtype
import numpy as np

class MlpDisentangledPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ctrl_sz, ac_space, hid_size, num_hid_layers, beta = 4, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)
        self._ob_space  = ob_space
        self._ac_space  = ac_space
        self._n_control = ctrl_sz
        self.pdtype     = pdtype = make_pdtype(ac_space)
        self.beta       = beta

        print np.shape(self._ob_space)
        print np.shape(self._ac_space)
        print ctrl_sz
        print beta


        self._create_network()

    def _create_network(self):
        n_hidden_encode   = [32, 32]
        n_hidden_v        = [10, 10]
        n_hidden_w        = [16, 16]
        n_input           = self._ob_space.shape
        n_z               = 20
        n_w               = 12         # Need to test with value, 
        n_v               = n_z - n_w # Conditionally independent variables, from beta-VAE

        sequence_length = None
        self.ob     = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(self._ob_space.shape))
        self.c_in   = U.get_placeholder(name="c_in", dtype = tf.float32, shape=[sequence_length, self._n_control])

        # need something related to observation normalization
        self.obz    = self.ob
        
        self.z_mean, self.z_log_sigma_sq = \
                                        self._create_encoder_network(hidden_layers = n_hidden_encode, latent_sz = n_z)

        self.latent_loss = self.beta * -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
                                                  - tf.square(self.z_mean)
                                                  - tf.exp(self.z_log_sigma_sq), 1)
        # batch_sz = tf.shape(self.x)[0]
        batch_sz    = tf.shape(self.ob)[0]
        eps_shape   = tf.stack([batch_sz, n_z])

        # mean=0.0, stddev=1.0    
        eps = tf.random_normal( eps_shape, 0, 1, dtype=tf.float32 )
        
        # z = mu + sigma * epsilon
        self.z      = tf.add(self.z_mean, 
                            tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
        self.z_w    = self.z[:, :n_w]
        self.z_v    = self.z[:, n_w:]

        self.vpred, self.pd = self._create_policy_network(distil_layers = n_hidden_w, task_layers = n_hidden_v, pdtype = self.pdtype)

        stochastic = tf.placeholder(name = "stochastic", dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode()) 

        self._act = U.function([stochastic, self.ob, self.c_in], [ac, self.vpred])

 
    def _create_encoder_network(self, hidden_layers, latent_sz):
        last_out = self.obz
        for (i, hid_size) in enumerate(hidden_layers):
            last_out    = tf.nn.relu(U.dense(last_out, hid_size, "enc%i"%(i+1), weight_init=tf.contrib.layers.xavier_initializer()))

        z_mean          = U.dense(last_out, latent_sz, "enc_mean", weight_init=tf.contrib.layers.xavier_initializer())
        z_log_sigma_sq  = U.dense(last_out, latent_sz, "enc_sigma", weight_init=tf.contrib.layers.xavier_initializer())

        return (z_mean, z_log_sigma_sq)

    def _create_policy_network(self, distil_layers, task_layers, pdtype):
        # Distil layers
        last_out = self.z_w
        for (i, hid_size) in enumerate(distil_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "distpol_fc%i"%(i+1), weight_init=U.normc_initializer(1.0)))

        dist_mean = U.dense(last_out, pdtype.param_shape()[0]//2, "distpolfinal", weight_init=U.normc_initializer(0.01)) 

        # Task layers
        last_out = self.z_v
        for (i, hid_size) in enumerate(task_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "taskpol_fc%i"%(i+1), weight_init=U.normc_initializer(1.0)))

        task_mean = U.dense(last_out, pdtype.param_shape()[0]//2, "taskpolfinal", weight_init = U.normc_initializer(0.01)) 

        mean = U.sum([dist_mean, task_mean], axis = 0, name = "polfinal")
        logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
        pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)

        pd = pdtype.pdfromflat(pdparam) 

        # Distil layers vprediction
        last_out = self.z_w
        for (i, hid_size) in enumerate(distil_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "distvf_fc%i"%(i+1), weight_init=U.normc_initializer(1.0)))

        dist_vf = U.dense(last_out, 1, "distvffinal", weight_init=U.normc_initializer(1.0))[:,0]

        # Task layers vprediction
        last_out = self.z_v
        for (i, hid_size) in enumerate(task_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "taskvf_fc%i"%(i+1), weight_init=U.normc_initializer(1.0)))

        task_vf = U.dense(last_out, 1, "taskvffinal", weight_init=U.normc_initializer(1.0))[:,0]

        vpred = U.sum([dist_vf, task_vf], axis = 0, name = "vffinal")
        return (vpred, pd) 

    def act(self, stochastic, ob, c_in):
        ac1, vpred1 =  self._act(stochastic, ob[None], c_in[None])
        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

