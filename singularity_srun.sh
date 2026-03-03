srun singularity exec --nv --writable-tmpfs \
     -B /ceph/project/es26-ce8-avs-824/whispers-in-the-storm/extern/MambAttention/mambattention_venv:/scratch/mambattention_venv \
     -B $HOME/.singularity:/scratch/singularity \
     /ceph/container/pytorch/pytorch_26.01.sif \
     /bin/bash -c "export TMPDIR=/scratch/singularity/tmp && \
                   export TRITON_LIBCUDA_PATH=/.singularity.d/libs && \
                   source /scratch/mambattention_venv/bin/activate && \
                   ls -l /.singularity.d/libs/libcuda.so*"