
import tensorflow as tf, numpy as np


def traj_show(pi, env, horizon, stochastic):
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
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
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
        env.render()
        # time.sleep(0.1)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            print(cur_ep_ret)
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

def test(test_env, policy_func, *,
        timesteps_per_batch, env_name
        ):
    ob_space = test_env.observation_space
    ac_space = test_env.action_space
    pi = policy_func("pi", ob_space, ac_space)

    seg_show = traj_show(pi, test_env, timesteps_per_batch, stochastic=False)
    sess = tf.get_default_session()
    saver = tf.train.Saver()
    saver.restore(sess, './' + env_name + '/' + env_name + '.cptk')
    while True:
        seg_show.__next__()

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]