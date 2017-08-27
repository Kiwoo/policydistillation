#!/usr/bin/env python
from misc_util import set_global_seeds
import tf_util as U
import benchmarks
# import monitor
import os.path as osp
import gym, logging
# import logger
import sys
import trpo_dist_twolayers
from mlp_dist_twolayers_policy import MlpDistTwolayersPolicy
from misc_util import set_global_seeds
num_cpu=1


# This is to test two layers alternative update

def train(env_id_list, num_timesteps, seed, save_name):
    # whoami  = mpi_fork(num_cpu)
    # if whoami == "parent":
    #     return
    import tf_util as U
    # logger.session().__enter__()
    sess = U.single_threaded_session()
    sess.__enter__()

    # rank = MPI.COMM_WORLD.Get_rank()
    # if rank != 0:
    #     logger.set_level(logger.DISABLED)
    # workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(seed)
    env_list = []
    for _, env_id in enumerate(env_id_list):
        env = gym.make(env_id)
        env.seed(seed)
        env_list.append(env)

    def policy_fn(name, ob_space, ac_space):
        return MlpDistTwolayersPolicy(name=name, ob_space=env_list[0].observation_space, ac_space=env_list[0].action_space, control_space = len(env_list),
            hid_size=64, num_hid_layers=4)
    # env = bench.Monitor(env, osp.join(logger.get_dir(), "%i.monitor.json" % rank))
    
    # gym.logger.setLevel(logging.WARN)
    # for i in range(len(env_id_list)):
    #     env = env_list[i]
    #     env_id = env_id_list[i]
    trpo_dist_twolayers.learn(env_list = env_list, save_name = save_name, policy_func = policy_fn, timesteps_per_batch=20000, max_kl=0.01, cg_iters=10, cg_damping=0.1,
        max_timesteps=0, gamma=0.99, lam=0.98, lr = 1e-4, vf_batch_size = 64, vf_iters=5, vf_stepsize=3e-4, render_freq = 10)
    env.close()
        

def main():
    env_id_list = ['Hopper-v1']
    save_name = 'Dist_Twolayers'
    train(env_id_list, num_timesteps=1e7, seed=0, save_name= save_name)

if __name__ == '__main__':
    main()
