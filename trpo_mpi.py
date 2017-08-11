from math_util import explained_variance 
from misc_util import zipsame, header, warn, failure 
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

def traj_segment_generator(pi, env, horizon, stochastic):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            return {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            _, vpred = pi.act(stochastic, ob)            
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

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

def learn(env, policy_func, 
        timesteps_per_batch, # what to train on
        max_kl, cg_iters,
        gamma, lam, # advantage estimation
        lr = 1e-4,
        vf_batch_size = 64,
        entcoeff=0.0,
        cg_damping=1e-2,
        vf_stepsize=3e-4,
        vf_iters =3,
        max_timesteps=0, max_episodes=0, max_iters=0,  # time constraint
        callback=None,
        save_iter_freq = 50
        ):
    # nworkers = MPI.COMM_WORLD.Get_size()
    # rank = MPI.COMM_WORLD.Get_rank()
    # np.set_printoptions(precision=3)    
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space)
    oldpi = policy_func("oldpi", ob_space, ac_space)
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = U.mean(kloldnew)
    meanent = U.mean(ent)
    entbonus = entcoeff * meanent

    vferr = U.mean(tf.square(pi.vpred - ret))

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # advantage * pnew / pold
    surrgain = U.mean(ratio * atarg)

    optimgain = surrgain + entbonus
    losses = [optimgain, meankl, entbonus, surrgain, meanent]
    loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

    dist = meankl

    all_var_list = pi.get_trainable_variables()
    var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
    vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
 

    #### CHECK !!!!! ######



    optimizer=tf.train.AdamOptimizer(learning_rate=lr, epsilon = 0.01/vf_batch_size)

    # vfadam = MpiAdam(vf_var_list)




    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)
    klgrads = tf.gradients(dist, var_list)
    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    shapes = [var.get_shape().as_list() for var in var_list]
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
    compute_losses = U.function([ob, ac, atarg], losses)
    compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
    compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
    compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))
    # compute_vfloss = U.function([ob, ret], vferr)

    vf_optimize_expr = optimizer.minimize(vferr, var_list=vf_var_list)
    vf_train = U.function([ob, ret], vferr, updates = [vf_optimize_expr])


    U.initialize()
    th_init = get_flat()
    set_from_flat(th_init)



    ##### CHECK !!! #######

    # vfadam.sync()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    # tstart = time.time()
    lenbuffer = deque(maxlen=40) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0])==1

    iter_log = []
    epis_log = []
    timestep_log = []
    ret_mean_log = []
    ret_std_log = []


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
        seg = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True)
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

        if hasattr(pi, "ret_rms"): pi.ret_rms.update(tdlamret)
        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        args = seg["ob"], seg["ac"], seg["adv"]
        fvpargs = [arr[::5] for arr in args]
        def fisher_vector_product(p):
            return compute_fvp(p, *fvpargs) + cg_damping * p

        assign_old_eq_new() # set old parameter values to new parameter values
        # with timed("computegrad"):
        surrbefore, _,_,_,_, g = compute_lossandgrad(*args)
        surrbefore = np.array(surrbefore)
        # g = allmean(g)
        if np.allclose(g, 0):
            print("Got zero gradient. not updating")
        else:
            # with timed("cg"):
            stepdir = U.conjugate_gradient(fisher_vector_product, g, cg_iters=cg_iters)
            assert np.isfinite(stepdir).all()
            shs = .5*stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / max_kl)
            # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
            fullstep = stepdir / lm
            expectedimprove = g.dot(fullstep)
            # surrbefore = lossbefore[0]
            stepsize = 1.0
            thbefore = get_flat()
            for _ in range(10):
                thnew = thbefore + fullstep * stepsize
                set_from_flat(thnew)
                meanlosses = surr, kl, _,_,_ = np.array(compute_losses(*args))
                #losses = [optimgain, meankl, entbonus, surrgain, meanent]
                improve = surr - surrbefore
                # print("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
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
            # if nworkers > 1 and iters_so_far % 20 == 0:
                # paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum())) # list of tuples
                # assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

        # for (lossname, lossval) in zip(loss_names, meanlosses):
            # logger.record_tabular(lossname, lossval)

        # with timed("vf"):



        for _ in range(vf_iters):
            for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]), 
            include_final_partial_batch=False, batch_size=64):
                # vfloss = compute_vfloss(mbob, mbret)
                vfloss = vf_train(mbob, mbret)
                #### CHECK !!!! ####
                
                # vfadam.update(g, vf_stepsize)

        # print("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

        # lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        # listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        # lens, rews = map(flatten_lists, zip(*listoflrpairs))
        # lenbuffer.extend(lens)
        # rewbuffer.extend(rews)

        # logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        # print("EpRewMean", np.mean(seg["ep_rets"]))
        # logger.record_tabular("EpThisIter", len(lens))


        episodes_so_far += len(seg["ep_lens"])
        timesteps_so_far += sum(seg["ep_lens"])
        iters_so_far += 1

        mean_ret = np.mean(seg["ep_rets"])
        std_ret = np.std(seg["ep_rets"])

        if iters_so_far % save_iter_freq == 1:
            iter_log.append(iters_so_far)
            epis_log.append(episodes_so_far)
            timestep_log.append(timesteps_so_far)
            ret_mean_log.append(mean_ret)
            ret_std_log.append(std_ret)

            save_file = "test_iter_{}.hdf5".format(iters_so_far)

            with h5py.File(save_file, 'w') as f:
                def write(dsetname, a):
                    f.create_dataset(dsetname, data=a, compression='gzip', compression_opts=9)
                # Right-padded trajectory data using custom RaggedArray class.
                write('iters_so_far', iter_log)
                write('episodes_so_far', epis_log)
                write('timesteps_so_far', timestep_log)
                write('mean_ret', ret_mean_log)
                write('std_ret', ret_std_log)
                
            header('Wrote {}'.format(save_file))
        header('iters_so_far : {}'.format(iters_so_far))
        header('timesteps_so_far : {}'.format(timesteps_so_far))
        header('mean_ret : {}'.format(mean_ret))
        header('std_ret : {}'.format(std_ret))


        # logger.record_tabular("EpisodesSoFar", episodes_so_far)
        # logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        # logger.record_tabular("TimeElapsed", time.time() - tstart)

        # if rank==0:
        #     logger.dump_tabular()

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]