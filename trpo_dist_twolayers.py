from math_util import explained_variance 
from misc_util import zipsame, header, warn, failure, filesave, mkdir_p, get_cur_dir
import dataset
# from baselines import logger
import tf_util as U
import tensorflow as tf, numpy as np
import time
from console_util import colorize
from mpi4py import MPI
from collections import deque
# from baselines.common.mpi_adam import MpiAdam
import cg
from contextlib import contextmanager
import h5py
import pandas as pd
import os

def traj_segment_generator(pi, env_list, horizon, stochastic):
    # Initialize state variables
    t = 0
    env_index = 0
    num_env = len(env_list)
    env = env_list[env_index]
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    ob = env.reset()
    c_in = np.zeros(num_env)
    part_horizon = np.arange(num_env+1) * horizon / num_env

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    c_ins = np.array([c_in for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    c_in[env_index] = 1.0

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob, c_in)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            return {"ob" : obs, "c_in" : c_ins, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            _, vpred = pi.act(stochastic, ob)            
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        c_ins[i] = c_in
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        # vel = env.get_vel()
        # print vel

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0

            if i > part_horizon[env_index+1]: # env index update to +1 so every environment contains similar number of trajectories.
                c_in[env_index] = 0.0
                env_index = (env_index + 1) % num_env
                env = env_list[env_index]
                c_in[env_index] = 1.0

            ob = env.reset()
        t += 1

def sample_rew_vel(vel_min, vel_max, num_sample = 1):
    print "Sampling reward velocity constant"
    s = np.random.uniform(vel_min, vel_max, num_sample)
    return s

def set_env_vel(env, vel):
    print "Set environment velocity"
    env.setvel(vel)

def 



def add_vtarg_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def load_checkpoints(load_requested = True, checkpoint_dir = get_cur_dir()):
    saver = tf.train.Saver(max_to_keep = None)
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkpoint and checkpoint.model_checkpoint_path and load_requested == True:
        saver.restore(U.get_session(), checkpoint.model_checkpoint_path)
        header("loaded checkpoint: {0}".format(checkpoint.model_checkpoint_path))
    else:
        header("Could not find old checkpoint")
        if not os.path.exists(checkpoint_dir):
            mkdir_p(checkpoint_dir)
    return saver    

def learn(env_list, save_name, policy_func, 
        timesteps_per_batch, # what to train on
        max_kl, cg_iters,
        gamma, lam, # advantage estimation
        lr = 1e-4,
        vf_batch_size = 64,
        entcoeff=0.0,
        cg_damping=1e-2,
        vf_stepsize=3e-4,
        vf_iters =3,
        max_timesteps=0, max_episodes=0, max_iters=502,  # time constraint
        callback=None,
        save_data_freq = 10,
        save_model_freq = 10,
        render_freq = 10
        ):
    # nworkers = MPI.COMM_WORLD.Get_size()
    # rank = MPI.COMM_WORLD.Get_rank()
    # np.set_printoptions(precision=3)    
    # Setup losses and stuff
    # ----------------------------------------
    ob_space    = env_list[0].observation_space
    ac_space    = env_list[0].action_space
    pi          = policy_func("pi", ob_space, ac_space)
    oldpi       = policy_func("oldpi", ob_space, ac_space)
    atarg       = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret         = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    ob      = U.get_placeholder_cached(name="ob")
    c_in    = U.get_placeholder_cached(name="c_in")
    ac      = pi.pdtype.sample_placeholder([None])

    kloldnew    = oldpi.pd.kl(pi.pd)
    ent         = pi.pd.entropy()
    meankl      = U.mean(kloldnew)
    meanent     = U.mean(ent)
    entbonus    = entcoeff * meanent

    vferr    = U.mean(tf.square(pi.vpred - ret))

    ratio    = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # advantage * pnew / pold
    surrgain = U.mean(ratio * atarg)

    optimgain   = surrgain + entbonus
    losses      = [optimgain, meankl, entbonus, surrgain, meanent]
    loss_names  = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

    dist = meankl

    all_var_list    = pi.get_trainable_variables()
    var_list        = [v for v in all_var_list if v.name.split("/")[1].startswith("pol1")]
    vf_var_list     = [v for v in all_var_list if v.name.split("/")[1].startswith("vf1")]
 
    var_list_task   = [v for v in all_var_list if v.name.split("/")[1].startswith("pol2")]
    vf_var_list_task= [v for v in all_var_list if v.name.split("/")[1].startswith("vf2")]


    #### CHECK !!!!! ######



    optimizer=tf.train.AdamOptimizer(learning_rate=lr, epsilon = 0.01/vf_batch_size)

    # vfadam = MpiAdam(vf_var_list)




    get_flat        = U.GetFlat(var_list)
    set_from_flat   = U.SetFromFlat(var_list)
    klgrads         = tf.gradients(dist, var_list)
    flat_tangent    = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    shapes          = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
        start += sz
    gvp = tf.add_n([U.sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)]) #pylint: disable=E1111
    fvp = U.flatgrad(gvp, var_list)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, c_in, atarg], losses)
    compute_lossandgrad = U.function([ob, ac, c_in, atarg], losses + [U.flatgrad(optimgain, var_list)])
    compute_fvp = U.function([flat_tangent, ob, ac, c_in, atarg], fvp)
    compute_vflossandgrad = U.function([ob, c_in, ret], U.flatgrad(vferr, vf_var_list))
    # compute_vfloss = U.function([ob, ret], vferr)

    vf_optimize_expr = optimizer.minimize(vferr, var_list=vf_var_list)
    vf_train = U.function([ob, c_in, ret], vferr, updates = [vf_optimize_expr])



    get_flat_task       = U.GetFlat(var_list_task)
    set_from_flat_task  = U.SetFromFlat(var_list_task)
    klgrads_task        = tf.gradients(dist, var_list_task)
    flat_tangent_task   = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    shapes              = [var.get_shape().as_list() for var in var_list_task]
    start = 0
    tangents_task = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents_task.append(tf.reshape(flat_tangent_task[start:start+sz], shape))
        start += sz
    gvp_task = tf.add_n([U.sum(g*tangent) for (g, tangent) in zipsame(klgrads_task, tangents_task)]) #pylint: disable=E1111
    fvp_task = U.flatgrad(gvp_task, var_list_task)

    # compute_losses = U.function([ob, ac, atarg], losses)
    compute_lossandgrad_task    = U.function([ob, ac, c_in, atarg], losses + [U.flatgrad(optimgain, var_list_task)])
    compute_fvp_task            = U.function([flat_tangent_task, ob, ac, c_in, atarg], fvp_task)
    compute_vflossandgrad_task  = U.function([ob, c_in, ret], U.flatgrad(vferr, vf_var_list_task))
    # compute_vfloss = U.function([ob, ret], vferr)

    vf_optimize_expr_task       = optimizer.minimize(vferr, var_list=vf_var_list_task)
    vf_train_task               = U.function([ob, c_in, ret], vferr, updates = [vf_optimize_expr_task])




    U.initialize()
    th_init = get_flat()
    set_from_flat(th_init)

    th_init_task = get_flat_task()
    set_from_flat_task(th_init_task)



    ##### CHECK !!! #######

    # vfadam.sync()

    # Prepare for rollouts
    # ----------------------------------------
    # seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0])==1

    iter_log = []
    epis_log = []
    timestep_log = []
    ret_mean_log = []
    ret_std_log = []

    # saver = tf.train.Saver(max_to_keep = None)
    cur_dir = get_cur_dir()
    save_dir = os.path.join(cur_dir, save_name)
    print save_dir
    saver = load_checkpoints(load_requested = True, checkpoint_dir = save_dir)

    meta_saved = False


    while True:        
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            print "Max Timestep : {}".format(timesteps_so_far)
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            print "Max Episodes : {}".format(episodes_so_far)
            break
        elif max_iters and iters_so_far >= max_iters:
            print "Max Iter : {}".format(iters_so_far)
            break
        warn("********** Iteration %i ************"%iters_so_far)

        # with timed("sampling"):
        #       seg = seg_gen.__next__()
        seg = traj_segment_generator(pi, env_list, timesteps_per_batch, stochastic=True)
        add_vtarg_and_adv(seg, gamma, lam)


        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, c_in, atarg, tdlamret = seg["ob"], seg["ac"], seg["c_in"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

        if hasattr(pi, "ret_rms"): pi.ret_rms.update(tdlamret)
        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        args = seg["ob"], seg["ac"], seg["c_in"], seg["adv"]
        fvpargs = [arr[::5] for arr in args]

        if iters_so_far % 2 == 0:

            print "Without control"

            def fisher_vector_product(p):
                return compute_fvp(p, *fvpargs) + cg_damping * p

            assign_old_eq_new() # set old parameter values to new parameter values
            surrbefore, _,_,_,_, g = compute_lossandgrad(*args)
            surrbefore = np.array(surrbefore)
            if np.allclose(g, 0):
                print("Got zero gradient. not updating")
            else:
                stepdir = U.conjugate_gradient(fisher_vector_product, g, cg_iters=cg_iters)
                assert np.isfinite(stepdir).all()
                shs = .5*stepdir.dot(fisher_vector_product(stepdir))
                lm = np.sqrt(shs / max_kl)
                fullstep = stepdir / lm
                expectedimprove = g.dot(fullstep)
                stepsize = 1.0
                thbefore = get_flat()
                for _ in range(10):
                    thnew = thbefore + fullstep * stepsize
                    set_from_flat(thnew)
                    meanlosses = surr, kl, _,_,_ = np.array(compute_losses(*args))
                    improve = surr - surrbefore
                    if not np.isfinite(meanlosses).all():
                        print("Got non-finite value of losses -- bad!")
                    elif kl > max_kl * 1.5:
                        print("violated KL constraint. shrinking step.")
                    elif improve < 0:
                        print("surrogate didn't improve. shrinking step.")
                    else:
                        break
                    stepsize *= .5
                else:
                    print("couldn't compute a good step")
                    set_from_flat(thbefore)

            for _ in range(vf_iters):
                for (mbob, mbc, mbret) in dataset.iterbatches((seg["ob"], seg["c_in"], seg["tdlamret"]), 
                include_final_partial_batch=False, batch_size=64):
                    vfloss = vf_train(mbob, mbc, mbret)

        if iters_so_far % 2 == 1:  

            print "With control"
            def fisher_vector_product_task(p):
                return compute_fvp_task(p, *fvpargs) + cg_damping * p

            assign_old_eq_new() # set old parameter values to new parameter values
            surrbefore, _,_,_,_, g = compute_lossandgrad_task(*args)
            surrbefore = np.array(surrbefore)
            if np.allclose(g, 0):
                print("Got zero gradient. not updating")
            else:
                stepdir = U.conjugate_gradient(fisher_vector_product_task, g, cg_iters=cg_iters)
                assert np.isfinite(stepdir).all()
                shs = .5*stepdir.dot(fisher_vector_product_task(stepdir))
                lm = np.sqrt(shs / max_kl)
                fullstep = stepdir / lm
                expectedimprove = g.dot(fullstep)
                stepsize = 1.0
                thbefore = get_flat_task()
                for _ in range(10):
                    thnew = thbefore + fullstep * stepsize
                    set_from_flat_task(thnew)
                    meanlosses = surr, kl, _,_,_ = np.array(compute_losses(*args))
                    improve = surr - surrbefore
                    if not np.isfinite(meanlosses).all():
                        print("Got non-finite value of losses -- bad!")
                    elif kl > max_kl * 1.5:
                        print("violated KL constraint. shrinking step.")
                    elif improve < 0:
                        print("surrogate didn't improve. shrinking step.")
                    else:
                        break
                    stepsize *= .5
                else:
                    print("couldn't compute a good step")
                    set_from_flat_task(thbefore)

            for _ in range(vf_iters):
                for (mbob, mbc, mbret) in dataset.iterbatches((seg["ob"], seg["c_in"], seg["tdlamret"]), 
                include_final_partial_batch=False, batch_size=64):
                    vfloss = vf_train_task(mbob, mbc, mbret)



        episodes_so_far += len(seg["ep_lens"])
        timesteps_so_far += sum(seg["ep_lens"])
        iters_so_far += 1

        mean_ret = np.mean(seg["ep_rets"])
        std_ret = np.std(seg["ep_rets"])

        iter_log.append(iters_so_far)
        epis_log.append(episodes_so_far)
        timestep_log.append(timesteps_so_far)
        ret_mean_log.append(mean_ret)
        ret_std_log.append(std_ret)

        if iters_so_far > 10 and iters_so_far % save_model_freq == 1:
            if meta_saved == True:
                saver.save(U.get_session(), save_dir + '/' + 'checkpoint', global_step = iters_so_far, write_meta_graph = False)
            else:
                print "Save  meta graph"
                saver.save(U.get_session(), save_dir + '/' + 'checkpoint', global_step = iters_so_far, write_meta_graph = True)
                meta_saved = True

        if iters_so_far % save_data_freq == 1:
            iter_log_d = pd.DataFrame(iter_log)
            epis_log_d = pd.DataFrame(epis_log)
            timestep_log_d = pd.DataFrame(timestep_log)
            ret_mean_log_d = pd.DataFrame(ret_mean_log)
            ret_std_log_d = pd.DataFrame(ret_std_log)

            
            log_dir = "{}_log".format(save_name)
            log_dir = os.path.join(cur_dir, log_dir)
            if not os.path.exists(log_dir):
                mkdir_p(log_dir)

            save_file = "iter_{}.h5".format(iters_so_far)
            save_file = os.path.join(log_dir, save_file)

            with pd.HDFStore(save_file, 'w') as outf:
                outf['iter_log'] = iter_log_d
                outf['epis_log'] = epis_log_d
                outf['timestep_log'] = timestep_log_d
                outf['ret_mean_log'] = ret_mean_log_d
                outf['ret_std_log'] = ret_std_log_d
                
            filesave('Wrote {}'.format(save_file))

        header('iters_so_far : {}'.format(iters_so_far))
        header('timesteps_so_far : {}'.format(timesteps_so_far))
        header('mean_ret : {}'.format(mean_ret))
        header('std_ret : {}'.format(std_ret))


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]