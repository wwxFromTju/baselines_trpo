from mytrpo.util import explained_variance, zipsame, dataset
# from mytrpo import logger
import tensorflow as tf, numpy as np
import time
from mpi4py import MPI
from collections import deque
from mytrpo.mpi_util.mpi_util import MpiAdam
from mytrpo.util.cg import cg
from contextlib import contextmanager

import mytrpo.tf_util.tf_math as UM
import mytrpo.tf_util.tf_nn as UN
import mytrpo.tf_util.tf_th as UH
import mytrpo.tf_util.tf_type as UT


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]


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


def learn(env, policy_func, *,
        timesteps_per_batch, # what to train on
        max_kl, cg_iters,
        gamma, lam, # advantage estimation
        entcoeff=0.0,
        cg_damping=1e-2,
        vf_stepsize=3e-4,
        vf_iters =3,
        max_timesteps=0, max_episodes=0, max_iters=0,  # time constraint
        # callback=None,
        times=0):

    # number and rank
    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()


    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space

    pi = policy_func("pi", ob_space, ac_space)
    oldpi = policy_func("oldpi", ob_space, ac_space)
    atarg = tf.placeholder(dtype=tf.float32, shape=[None])
    ret = tf.placeholder(dtype=tf.float32, shape=[None])

    ob = UT.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    # 计算两个probability distribution之间的kl散度
    kloldnew = oldpi.pd.kl(pi.pd)
    meankl = UM.mean(kloldnew)

    # 计算新策略的熵---》max entropy
    ent = pi.pd.entropy()
    meanent = UM.mean(ent)
    entbonus = entcoeff * meanent

    # 不同value function上面误差，就是二范数
    vferr = UM.mean(tf.square(pi.vpred - ret))

    # advantage * pnew / pold
    # 就是L，然后通过L来求梯度的方向，然后trpo再算stepsize
    # 就是两个action采取概率的比值，避免数值上的误差，所以先做log，再exp，避免除法上的数值问题
    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))
    surrgain = UM.mean(ratio * atarg)

    # 这里将entropy也考虑上
    optimgain = surrgain + entbonus

    losses = [optimgain, meankl, entbonus, surrgain, meanent]
    # loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

    dist = meankl

    all_var_list = pi.get_trainable_variables()
    # 得到policy network的var
    var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
    # 得到value network的var
    vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
    # 因为value用的是二范数来做监督学习，所以可以直接用adam来优化
    vfadam = MpiAdam(vf_var_list)

    # 就是构造得到policy network的var的value的op
    get_flat = UN.GetFlat(var_list)
    # 构造从flat的value，reshape，同时assign回policy network的op
    set_from_flat = UN.SetFromFlat(var_list)

    klgrads = tf.gradients(dist, var_list)


    # 下面的目的，一直到fvp，目的就是计算fisher
    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    shapes = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents = []

    for shape in shapes:
        sz = UN.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
        start += sz

    # 就是这个意思tf.sum( kl_gradient * v )
    gvp = tf.add_n([UM.sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)])
    fvp = UN.flatgrad(gvp, var_list)

    # 将pi_new 赋值给 pi_old的op
    assign_old_eq_new = UH.function([],[], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])

    # 计算loss的op 与 同时计算loos和grad的op
    compute_losses = UH.function([ob, ac, atarg], losses)
    compute_lossandgrad = UH.function([ob, ac, atarg], losses + [UN.flatgrad(optimgain, var_list)])

    # 计算fisher的op 与 同时计算fisher与grad的op
    compute_fvp = UH.function([flat_tangent, ob, ac, atarg], fvp)
    compute_vflossandgrad = UH.function([ob, ret], UN.flatgrad(vferr, vf_var_list))

    @contextmanager
    def timed(msg):
        if rank == 0:
            tstart = time.time()
            yield
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out

    UN.initialize()
    # 先拿到policy的那个var的数值
    th_init = get_flat()
    # 广播，所以th_init都相同
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    # set到var里main
    set_from_flat(th_init)

    vfadam.sync()
    print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    # lenbuffer = deque(maxlen=40)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40)  # rolling buffer for episode rewards

    while True:
        # if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break

        # with timed("sampling"):
        seg = seg_gen.__next__()


        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std()

        if hasattr(pi, "ret_rms"): pi.ret_rms.update(tdlamret)
        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)

        args = seg["ob"], seg["ac"], atarg
        fvpargs = [arr[::5] for arr in args]

        # MPI
        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p

        assign_old_eq_new() # set old parameter values to new parameter values


        # with timed("computegrad"):
        *lossbefore, g = compute_lossandgrad(*args)



        # MPI
        lossbefore = allmean(np.array(lossbefore))
        g = allmean(g)


        if np.allclose(g, 0):
            pass
        else:
            # stepdir ---> xk，就是用conjugate gradient (CG) algorithm 来算xk
            #with timed("cg"):
            stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank==0)

            assert np.isfinite(stepdir).all()
            # stepdir.dot(fisher_vector_product(stepdir)) ---> xk_Hk_xk
            shs = 0.5 * stepdir.dot(fisher_vector_product(stepdir))
            # max_kl ---> entropy_delta
            lm = np.sqrt(shs / max_kl)
            # fullstep --> parameters_delta
            fullstep = stepdir / lm
            # expectedimprove = g.dot(fullstep)
            surrbefore = lossbefore[0]
            stepsize = 1.0
            thbefore = get_flat()
            for _ in range(10):
                # 赋值
                thnew = thbefore + fullstep * stepsize
                set_from_flat(thnew)

                # MPI
                # 计算当前step之后的各种参数，看看是不是满足trpo的条件，不是的话，重新设置
                meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)))
                improve = surr - surrbefore
                # 就是一旦一个不满足就将step缩小0.5
                if not np.isfinite(meanlosses).all():
                    pass
                elif kl > max_kl * 1.5:
                    pass
                elif improve < 0:
                    pass
                else:
                    break
                stepsize *= .5
            else:
                # 就是10次都不满足条件，重新reset回原来的参数设置
                set_from_flat(thbefore)

            # MPI
            if nworkers > 1 and iters_so_far % 20 == 0:
                paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum())) # list of tuples
                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])


        with timed("vf"):
            for _ in range(vf_iters):
                for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]), include_final_partial_batch=False, batch_size=64):
                    # MPI
                    g = allmean(compute_vflossandgrad(mbob, mbret))
                    vfadam.update(g, vf_stepsize)

        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values

        #MPI
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        # lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        # logger.record_tabular("EpRewMean", np.mean(rewbuffer))

        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
