#!/usr/bin/env bash
# Section 4M M181: memory and time of the layer-checkpointed trainer (GPU 0 of the pilots job, idle).
main() {
  source .venv/bin/activate; export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
  python -I logs/nano4/m181_ckpt_check.py 768 3072 512 8 0 || return 1
  python -I logs/nano4/m181_ckpt_check.py 768 3072 2048 8 4 3 2 0
}
main "$@"; exit $?
