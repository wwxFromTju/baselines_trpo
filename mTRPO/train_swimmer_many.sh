echo "timesteps 1000000" >> Swimmer_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Swimmer-v1 --num-timesteps 1000000 --times 1) 2>> Swimmer_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Swimmer-v1 --num-timesteps 1000000 --times 2) 2>> Swimmer_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Swimmer-v1 --num-timesteps 1000000 --times 3) 2>> Swimmer_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Swimmer-v1 --num-timesteps 1000000 --times 4) 2>> Swimmer_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Swimmer-v1 --num-timesteps 1000000 --times 5) 2>> Swimmer_many.txt

echo "timesteps 2000000" >> Swimmer_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Swimmer-v1 --num-timesteps 2000000 --times 1) 2>> Swimmer_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Swimmer-v1 --num-timesteps 2000000 --times 2) 2>> Swimmer_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Swimmer-v1 --num-timesteps 2000000 --times 3) 2>> Swimmer_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Swimmer-v1 --num-timesteps 2000000 --times 4) 2>> Swimmer_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Swimmer-v1 --num-timesteps 2000000 --times 5) 2>> Swimmer_many.txt


echo "timesteps 3000000" >> Swimmer_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Swimmer-v1 --num-timesteps 3000000 --times 1) 2>> Swimmer_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Swimmer-v1 --num-timesteps 3000000 --times 2) 2>> Swimmer_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Swimmer-v1 --num-timesteps 3000000 --times 3) 2>> Swimmer_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Swimmer-v1 --num-timesteps 3000000 --times 4) 2>> Swimmer_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Swimmer-v1 --num-timesteps 3000000 --times 5) 2>> Swimmer_many.txt


echo "timesteps 4000000" >> Swimmer_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Swimmer-v1 --num-timesteps 4000000 --times 1) 2>> Swimmer_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Swimmer-v1 --num-timesteps 4000000 --times 2) 2>> Swimmer_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Swimmer-v1 --num-timesteps 4000000 --times 3) 2>> Swimmer_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Swimmer-v1 --num-timesteps 4000000 --times 4) 2>> Swimmer_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env Swimmer-v1 --num-timesteps 4000000 --times 5) 2>> Swimmer_many.txt

