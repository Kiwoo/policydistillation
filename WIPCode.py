def select_traj(seg, upper_percent ,lower_percent):
    # Select traj which has upper_percent % episode reward and has lower_percent % episode reward
    # Since generate trajectories doesnt inlcude last trajectory in their calculation of ep_reward and ep_length
    # We don't need to take into account of nextvpred, it is just 0 since every trajectory has complete episode.

    if upper_percent == None or lower_percent == None:
        print "Error: select_traj"
        return

    ep_reward = seg["ep_rets"]
    ep_len    = seg["ep_lens"]

    print len(ep_reward)

    obs     = []
    c_ins   = []
    rews    = []
    vpreds  = []
    news    = []
    acs     = []

    upper_cutline = np.percentile(ep_reward, upper_percent, interpolation='higher')
    lower_cutline = np.percentile(ep_reward, lower_percent, interpolation='higher')
    start  = 0
    end    = 0
    for i in range(len(ep_reward)):
        start = end
        end   = end + ep_len[i]
        if ep_reward[i] >= lower_cutline and ep_reward[i] <= upper_cutline:
            obs.append(seg["ob"][start:start+ep_len[i]])
            c_ins.append(seg["c_in"][start:start+ep_len[i]])
            rews.append(seg["rew"][start:start+ep_len[i]])
            vpreds.append(seg["vpred"][start:start+ep_len[i]])
            news.append(seg["new"][start:start+ep_len[i]])
            acs.append(seg["ac"][start:start+ep_len[i]])
    return {"ob" : np.concatenate([obs]),
            "c_in" : np.concatenate([c_ins]),
            "rew" : np.concatenate([rews]),
            "vpred" : np.concatenate([vpreds]),
            "new" : np.concatenate([news]),
            "ac" : np.concatenate([acs]),
            "nextvpred" : 0}


def sample_rew_vel(vel_min, vel_max, num_sample = 1):
    print "Sampling reward velocity constant"
    s = np.random.uniform(vel_min, vel_max, num_sample)
    print "Testing Need to change later"
    s[0] = 1.2
    return s

def set_env_vel(env, vel):
    print "Set environment velocity"
    env.setvel(vel)