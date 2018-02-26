#!/usr/bin/env python3
from mpi4py import MPI
from mytrpo.util import set_global_seeds
import gym
from mytrpo import logger
from mytrpo.trpo_mpi.mpi_policy import MlpPolicy
from mytrpo.trpo_mpi import trpo_mpi_test

import mytrpo.tf_util.tf_sess as US

def policy_fn(name, ob_space, ac_space):
    return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                     hid_size=64, num_hid_layers=3)

def train(env_id, num_timesteps, seed):
    sess = US.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    test_env = gym.make(env_id)
    test_env.seed(workerseed)

    trpo_mpi_test.test(test_env, policy_fn, timesteps_per_batch=1024, env_name=env_id)
    test_env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Walker2d-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(5e6))
    args = parser.parse_args()
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()
