#!/bin/bash

#SBATCH --job-name=experimento_benchmark
#SBATCH --nodes=1
#SBATCH --output="job_logs/%j.out"
#SBATCH --gpus=1

srun singularity run --nv \
    -B/pgeoprj,/pgeodsv,/pgeo,/tatu,/scr01 \
    /tatu/container_images/basica_experimentacao.sif \
    python segment_seismic.py -a unet -d /pgeoprj/godeep/dados/l1_datasets/f3_silva/amplitude.npy -l /pgeoprj/godeep/dados/l1_datasets/f3_silva/labels.npy -t -v -W -p /pgeoprj/godeep/ewab/seismic-segmentation-benchmark/paper_results_all -b 8 -s