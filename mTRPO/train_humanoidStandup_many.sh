echo "timesteps 1000000" >> HumanoidStandup_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env HumanoidStandup-v1 --num-timesteps 1000000 --times 1) 2>> HumanoidStandup_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env HumanoidStandup-v1 --num-timesteps 1000000 --times 2) 2>> HumanoidStandup_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env HumanoidStandup-v1 --num-timesteps 1000000 --times 3) 2>> HumanoidStandup_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env HumanoidStandup-v1 --num-timesteps 1000000 --times 4) 2>> HumanoidStandup_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env HumanoidStandup-v1 --num-timesteps 1000000 --times 5) 2>> HumanoidStandup_many.txt

echo "timesteps 2000000" >> HumanoidStandup_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env HumanoidStandup-v1 --num-timesteps 2000000 --times 1) 2>> HumanoidStandup_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env HumanoidStandup-v1 --num-timesteps 2000000 --times 2) 2>> HumanoidStandup_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env HumanoidStandup-v1 --num-timesteps 2000000 --times 3) 2>> HumanoidStandup_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env HumanoidStandup-v1 --num-timesteps 2000000 --times 4) 2>> HumanoidStandup_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env HumanoidStandup-v1 --num-timesteps 2000000 --times 5) 2>> HumanoidStandup_many.txt


echo "timesteps 3000000" >> HumanoidStandup_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env HumanoidStandup-v1 --num-timesteps 3000000 --times 1) 2>> HumanoidStandup_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env HumanoidStandup-v1 --num-timesteps 3000000 --times 2) 2>> HumanoidStandup_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env HumanoidStandup-v1 --num-timesteps 3000000 --times 3) 2>> HumanoidStandup_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env HumanoidStandup-v1 --num-timesteps 3000000 --times 4) 2>> HumanoidStandup_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env HumanoidStandup-v1 --num-timesteps 3000000 --times 5) 2>> HumanoidStandup_many.txt


echo "timesteps 4000000" >> HumanoidStandup_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env HumanoidStandup-v1 --num-timesteps 4000000 --times 1) 2>> HumanoidStandup_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env HumanoidStandup-v1 --num-timesteps 4000000 --times 2) 2>> HumanoidStandup_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env HumanoidStandup-v1 --num-timesteps 4000000 --times 3) 2>> HumanoidStandup_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env HumanoidStandup-v1 --num-timesteps 4000000 --times 4) 2>> HumanoidStandup_many.txt
(time mpirun -np 8 python -m mytrpo.trpo_mpi.run_mujoco --env HumanoidStandup-v1 --num-timesteps 4000000 --times 5) 2>> HumanoidStandup_many.txt

