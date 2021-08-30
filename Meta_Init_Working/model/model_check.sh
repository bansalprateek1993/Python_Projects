srun -K --ntasks=1 --gpus-per-task=1 --cpus-per-gpu=5 --mem-per-gpu=16G  -p GTX1080Ti \
     --container-mounts=/netscratch:/netscratch,/ds:/ds,`pwd`:`pwd` \
     --container-image=/netscratch/enroot/dlcc_pytorch_20.10.sqsh \
     --container-workdir=`pwd` \
     --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
     python3 renet_bk.py

