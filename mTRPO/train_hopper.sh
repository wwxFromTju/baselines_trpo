echo "np 8" >> hopper.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Hopper-v1 --num-timesteps 1000000 --times 1) 2>> hopper.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Hopper-v1 --num-timesteps 1000000 --times 2) 2>> hopper.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Hopper-v1 --num-timesteps 1000000 --times 3) 2>> hopper.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Hopper-v1 --num-timesteps 1000000 --times 4) 2>> hopper.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Hopper-v1 --num-timesteps 1000000 --times 5) 2>> hopper.txt

echo "np 4" >> hopper.txt
(time mpirun -np 4 python -m mytrpo.trpo_mpi.run_mujoco --env Hopper-v1 --num-timesteps 1000000 --times 1) 2>> hopper.txt
(time mpirun -np 4 python -m mytrpo.trpo_mpi.run_mujoco --env Hopper-v1 --num-timesteps 1000000 --times 2) 2>> hopper.txt
(time mpirun -np 4 python -m mytrpo.trpo_mpi.run_mujoco --env Hopper-v1 --num-timesteps 1000000 --times 3) 2>> hopper.txt
(time mpirun -np 4 python -m mytrpo.trpo_mpi.run_mujoco --env Hopper-v1 --num-timesteps 1000000 --times 4) 2>> hopper.txt
(time mpirun -np 4 python -m mytrpo.trpo_mpi.run_mujoco --env Hopper-v1 --num-timesteps 1000000 --times 5) 2>> hopper.txt


echo "np 2" >> hopper.txt
(time mpirun -np 2 python -m mytrpo.trpo_mpi.run_mujoco --env Hopper-v1 --num-timesteps 1000000 --times 1) 2>> hopper.txt
(time mpirun -np 2 python -m mytrpo.trpo_mpi.run_mujoco --env Hopper-v1 --num-timesteps 1000000 --times 2) 2>> hopper.txt
(time mpirun -np 2 python -m mytrpo.trpo_mpi.run_mujoco --env Hopper-v1 --num-timesteps 1000000 --times 3) 2>> hopper.txt
(time mpirun -np 2 python -m mytrpo.trpo_mpi.run_mujoco --env Hopper-v1 --num-timesteps 1000000 --times 4) 2>> hopper.txt
(time mpirun -np 2 python -m mytrpo.trpo_mpi.run_mujoco --env Hopper-v1 --num-timesteps 1000000 --times 5) 2>> hopper.txt


echo "np 1" >> hopper.txt
(time mpirun -np 1 python -m mytrpo.trpo_mpi.run_mujoco --env Hopper-v1 --num-timesteps 1000000 --times 1) 2>> hopper.txt
(time mpirun -np 1 python -m mytrpo.trpo_mpi.run_mujoco --env Hopper-v1 --num-timesteps 1000000 --times 2) 2>> hopper.txt
(time mpirun -np 1 python -m mytrpo.trpo_mpi.run_mujoco --env Hopper-v1 --num-timesteps 1000000 --times 3) 2>> hopper.txt
(time mpirun -np 1 python -m mytrpo.trpo_mpi.run_mujoco --env Hopper-v1 --num-timesteps 1000000 --times 4) 2>> hopper.txt
(time mpirun -np 1 python -m mytrpo.trpo_mpi.run_mujoco --env Hopper-v1 --num-timesteps 1000000 --times 5) 2>> hopper.txt

