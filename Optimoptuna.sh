#python optim.py --structure 784 512 100 --Homeo_mode 'SM' --exp_N 1
#python optim.py --structure 784 1024 100 --Homeo_mode 'SM' --exp_N 1
#python optuna_optim.py --structure 4 30 12 --Homeo_mode 'batch' --exp_N 1
python optuna_optim.py --device -1 --structure 4 30 30 --Homeo_mode 'SM' --exp_N 1
python optuna_optim.py --device -1 --structure 4 30 150 --Homeo_mode 'SM' --exp_N 4
python optuna_optim.py --device -1 --structure 4 30 300 --Homeo_mode 'SM' --exp_N 5