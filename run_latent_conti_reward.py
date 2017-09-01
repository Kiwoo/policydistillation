#!/usr/bin/env python
from misc_util import set_global_seeds
import tf_util as U
import benchmarks
# import monitor
import os.path as osp
import gym, logging
# import logger
import sys
import trpo_latent_conti_reward
from mlp_latent_conti_reward import MlpLatentContinuousRewardPolicy
from misc_util import set_global_seeds
num_cpu=1


# This is to test two layers alternative update

def train(env_id, ctrl_range, num_timesteps, seed, save_name):
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
    env = gym.make(env_id)
    env.seed(seed)

    def policy_fn(name, ob_space, ac_space):
        return MlpLatentContinuousRewardPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space, control_space = 1,
            hid_size=64, num_hid_layers=4)
    # env = bench.Monitor(env, osp.join(logger.get_dir(), "%i.monitor.json" % rank))
    
    # gym.logger.setLevel(logging.WARN)
    # for i in range(len(env_id_list)):
    #     env = env_list[i]
    #     env_id = env_id_list[i]
    trpo_latent_conti_reward.learn(env = env, save_name = save_name, ctrl_range = ctrl_range, policy_func = policy_fn, timesteps_per_batch=20000, max_kl=0.01, cg_iters=10, cg_damping=0.1,
        max_timesteps=0, gamma=0.99, lam=0.98, lr = 1e-4, vf_batch_size = 64, vf_iters=5, vf_stepsize=3e-4, render_freq = 10)
    env.close()
        

def main():
    env_id = 'Hopper-v1'
    save_name = 'Latent_Conti_Reward'
    ctrl_range = {"min" : 0.5, "max" : 1.5}
    train(env_id, ctrl_range, num_timesteps=1e7, seed=0, save_name= save_name)

if __name__ == '__main__':
    main()
