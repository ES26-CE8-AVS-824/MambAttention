srun singularity exec --nv \
     -B /ceph/project/es26-ce8-avs-824/whispers-in-the-storm/extern/MambAttention/mambattention_venv:/scratch/mambattention_venv \
     -B $HOME/.singularity:/scratch/singularity \
     /ceph/container/pytorch/pytorch_26.01.sif \
     /bin/bash -c "export TMPDIR=/scratch/singularity/tmp && \
                   source /scratch/mambattention_venv/bin/activate && \
                   pip install pesq"