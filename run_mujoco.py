#!/usr/bin/env python
from misc_util import set_global_seeds
import tf_util as U
import benchmarks
import monitor
import os.path as osp
import gym, logging
# import logger
import sys
import trpo_mpi
from mlp_policy import MlpPolicy

num_cpu=1

def train(env_id, num_timesteps, seed):
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
    # set_global_seeds(workerseed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space,
            hid_size=32, num_hid_layers=2)
    # env = bench.Monitor(env, osp.join(logger.get_dir(), "%i.monitor.json" % rank))
    # env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    trpo_mpi.learn(env, policy_fn, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
        max_timesteps=num_timesteps, gamma=0.99, lam=0.98, lr = 1e-4, vf_batch_size = 64, vf_iters=5, vf_stepsize=1e-3)
    env.close()

def main():
    train('Hopper-v1', num_timesteps=1e7, seed=0)

if __name__ == '__main__':
    main()
