#!/usr/bin/env python3
import tensorflow as tf
from mpi4py import MPI
from mytrpo.util import set_global_seeds
import os
import sys
import gym
from mytrpo import logger
from mytrpo.trpo_mpi.mpi_policy import MlpPolicy
from mytrpo.trpo_mpi import trpo_mpi

import mytrpo.tf_util.tf_sess as US


def train(env_id, num_timesteps, seed, times):
    sess = US.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                         hid_size=64, num_hid_layers=3)
    env.seed(workerseed)

    trpo_mpi.learn(env, policy_fn, timesteps_per_batch=2048,
                   max_kl=0.01, cg_iters=10, cg_damping=0.1,
        max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3, times=times)
    env.close()

def main():
    print('hehehheheheh')
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Walker2d-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(2e7))
    parser.add_argument('--times', type=int, default=0)
    parser.add_argument('--store-weights', type=bool, default=False)
    args = parser.parse_args()
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, times=args.times)
    if rank == 0 and args.store_weights:
        if not os.path.exists('./' + args.env):
            os.mkdir('./' + args.env)
        sess = tf.get_default_session()
        saver = tf.train.Saver()
        saver.save(sess, './' + args.env + '/' + args.env + '.cptk')


if __name__ == '__main__':
    main()
