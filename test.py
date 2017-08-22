#!/usr/bin/env python
from misc_util import set_global_seeds
import tf_util as U
import benchmarks
from misc_util import zipsame, header, warn, failure, filesave, mkdir_p, get_cur_dir
import os.path as osp
import gym, logging
# import logger
import sys
import trpo_mpi
from mlp_policy import MlpPolicy
import argparse
import tensorflow as tf
import h5py
import pandas as pd
import numpy as np
import os
num_cpu=1

def load_checkpoints(saver, checkpoint_dir = get_cur_dir()):
    saver = tf.train.Saver(max_to_keep = None)
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(U.get_session(), checkpoint.model_checkpoint_path)
        header("loaded checkpoint: {0}".format(checkpoint.model_checkpoint_path))
    else:
        header("Could not find old checkpoint")

def sim(env_id, num_timesteps, seed):
    # whoami  = mpi_fork(num_cpu)
    # if whoami == "parent":
    #     return
    import tf_util as U
    # logger.session().__enter__()
    sess = U.single_threaded_session()
    sess.__enter__()

    cur_dir = get_cur_dir()
    save_dir = os.path.join(cur_dir, env_id)
    meta_dir = os.path.join(save_dir, "checkpoint-11.meta")

    saver = tf.train.import_meta_graph(meta_dir)
    load_checkpoints(saver, checkpoint_dir = save_dir)
    # saver.restore(sess, tf.train.latest_checkpoint('./'))

    graph = tf.get_default_graph()
    # oplist = graph.get_all_collection_keys()

    ob_s = graph.get_tensor_by_name("pi/ob:0")
    ac_s = graph.get_tensor_by_name("pi/cond/Merge:0")
    stochastic_s = graph.get_tensor_by_name("pi/stochastic:0")

    pi = U.function([ob_s, stochastic_s], ac_s)

    env = gym.make(env_id)
    ob = env.reset()

    iter_log    = []
    vel_log     = []

    i = 0
    for _ in range(1000):
        env.render()
        i = i+1
        if i%100 == 0:
            print "Step : {}".format(i)
        # action = env.action_space.sample()  # your agent here (this takes random actions)
        # feed_dict ={ob_s:ob[None], stochastic_s:True}
        ac = pi(ob[None], True)
        # print ac
        ob, reward, done, info = env.step(ac)        
        vel = env.getvel()
        # print vel
        iter_log.append(i)
        vel_log.append(vel)

        if done:
            env.reset()

    iter_log_d = pd.DataFrame(iter_log)
    velo_log_d = pd.DataFrame(vel_log)
    save_file = "vel_log.h5"
    with pd.HDFStore(save_file, 'w') as outf:
        outf['iter_log'] = iter_log_d
        outf['vel_log']  = velo_log_d
    print "Average Velocity : {}".format(np.mean(vel_log))

def main():
    sim('Hopper-v1', num_timesteps=1e7, seed=0)

if __name__ == '__main__':
    main()
