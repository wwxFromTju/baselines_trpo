echo "timesteps 1000000" >> Walker2d_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Walker2d-v1 --num-timesteps 1000000 --times 1) 2>> Walker2d_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Walker2d-v1 --num-timesteps 1000000 --times 2) 2>> Walker2d_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Walker2d-v1 --num-timesteps 1000000 --times 3) 2>> Walker2d_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Walker2d-v1 --num-timesteps 1000000 --times 4) 2>> Walker2d_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Walker2d-v1 --num-timesteps 1000000 --times 5) 2>> Walker2d_many.txt

echo "timesteps 2000000" >> Walker2d_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Walker2d-v1 --num-timesteps 2000000 --times 1) 2>> Walker2d_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Walker2d-v1 --num-timesteps 2000000 --times 2) 2>> Walker2d_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Walker2d-v1 --num-timesteps 2000000 --times 3) 2>> Walker2d_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Walker2d-v1 --num-timesteps 2000000 --times 4) 2>> Walker2d_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Walker2d-v1 --num-timesteps 2000000 --times 5) 2>> Walker2d_many.txt


echo "timesteps 3000000" >> Walker2d_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Walker2d-v1 --num-timesteps 3000000 --times 1) 2>> Walker2d_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Walker2d-v1 --num-timesteps 3000000 --times 2) 2>> Walker2d_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Walker2d-v1 --num-timesteps 3000000 --times 3) 2>> Walker2d_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Walker2d-v1 --num-timesteps 3000000 --times 4) 2>> Walker2d_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Walker2d-v1 --num-timesteps 3000000 --times 5) 2>> Walker2d_many.txt


echo "timesteps 4000000" >> Walker2d_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Walker2d-v1 --num-timesteps 4000000 --times 1) 2>> Walker2d_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Walker2d-v1 --num-timesteps 4000000 --times 2) 2>> Walker2d_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Walker2d-v1 --num-timesteps 4000000 --times 3) 2>> Walker2d_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Walker2d-v1 --num-timesteps 4000000 --times 4) 2>> Walker2d_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Walker2d-v1 --num-timesteps 4000000 --times 5) 2>> Walker2d_many.txt

