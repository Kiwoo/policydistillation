#!/usr/bin/env python
from misc_util import set_global_seeds
import tf_util as U
import benchmarks
# import monitor
import os.path as osp
import gym, logging
# import logger
import sys
import trpo_mpi
from mlp_policy import MlpPolicy
import argparse
import tensorflow as tf

num_cpu=1


def sim(env_id, num_timesteps, seed):
    # whoami  = mpi_fork(num_cpu)
    # if whoami == "parent":
    #     return
    import tf_util as U
    # logger.session().__enter__()
    sess = U.single_threaded_session()
    sess.__enter__()

    # U.initialize()
    saver = tf.train.import_meta_graph('my_test_model-11.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    graph = tf.get_default_graph()
    # oplist = graph.get_all_collection_keys()

    ob_s = graph.get_tensor_by_name("pi/ob:0")
    ac_s = graph.get_tensor_by_name("pi/cond/Merge:0")
    stochastic_s = graph.get_tensor_by_name("pi/stochastic:0")

    pi = U.function([ob_s, stochastic_s], ac_s)

    env = gym.make(env_id)
    ob = env.reset()
    i = 0

    for _ in range(10000):
        env.render()
        i = i+1
        # action = env.action_space.sample()  # your agent here (this takes random actions)
        # feed_dict ={ob_s:ob[None], stochastic_s:True}
        ac = pi(ob[None], True)
        # print ac
        ob, reward, done, info = env.step(ac)
        if done:
            break
            # print "Done: {}".format(i)
            # i = 0
            # env.reset()


def main():
    sim('HopperC1-v1', num_timesteps=1e7, seed=0)

if __name__ == '__main__':
    main()
