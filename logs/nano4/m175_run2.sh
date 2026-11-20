#!/usr/bin/env bash
main() {
  source .venv/bin/activate; export OMP_NUM_THREADS=12
  python -I logs/nano4/m175_rerank_pilot.py --ckpt /work/u2507115/m161_ckpt/p/ckpt_-1_s0.pt --out results/m161/p/out_-1_s0.npz --size 512 2048 --json m175_rerank_pilot_p.json --save-scores results/m175/p || return 1
  python -I logs/nano4/m161_fused_eval.py --base P,S2y,S4ft,S4Lt --json m175_fused_p.json --addz pmean=results/m175/p_mean.npz --addz pmax=results/m175/p_max.npz --addz plme=results/m175/p_lme30.npz --add h2c=results/m161/h2/out_-1_s0.npz:centres --combo pmean+pmax --combo pmean+plme --combo pmean+h2c --combo pmean+pmax+h2c
}
main "$@"; exit $?
