#!/usr/bin/env python
from misc_util import set_global_seeds
import tf_util as U
import benchmarks
# import monitor
import os.path as osp
import gym, logging
# import logger
import sys
import trpo_dist_controls
from mlp_disentangled_policy import MlpDisentangledPolicy
from misc_util import set_global_seeds
num_cpu=1

def train(env_id_list, num_timesteps, seed):
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
    for i in range(len(env_id_list)):
        env = gym.make(env_id_list[i])
        env.seed(seed)
        env_list.append(env)

    def policy_fn(name, ob_space, ac_space):
        return MlpDisentangledPolicy(name=name, ob_space=env.observation_space, ctrl_sz = len(env_list), ac_space=env.action_space,
            hid_size=32, num_hid_layers=2)
    # env = bench.Monitor(env, osp.join(logger.get_dir(), "%i.monitor.json" % rank))
    
    trpo_dist_controls.learn(env_list = env_list, env_id_list = env_id_list, policy_func = policy_fn, timesteps_per_batch=5000, max_kl=0.025, cg_iters=10, cg_damping=0.1,
        max_timesteps=0, gamma=0.99, lam=0.98, lr = 1e-4, vf_batch_size = 64, vf_iters=5, vf_stepsize=3e-4, render_freq = 10)
    for i in range(len(env_id_list)):
        env_list[i].close()

def main():
    env_id_list = ['Hopper-v1', 'HopperC1-v1', 'HopperC2-v1', 'HopperC3-v1', 'HopperC4-v1']#, 'HopperC5-v1']
    train(env_id_list, num_timesteps=1e7, seed=0)

if __name__ == '__main__':
    main()
