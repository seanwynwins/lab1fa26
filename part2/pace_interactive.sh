#!/bin/sh
salloc -Jlab1_gpu_interactive --partition=ice-gpu,coc-gpu -N1 --ntasks-per-node=8 -C"intel&core24&nvidia-gpu" --gres=gpu:1
exit 0
