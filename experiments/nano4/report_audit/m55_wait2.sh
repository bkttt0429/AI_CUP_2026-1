cd /home/u2507115/BK_AICUP_2026-1
st() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
until st 724293 | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED"; do sleep 30; done
echo "== power"; grep -v "^+" logs/nano4/m55pow-724293.out | tail -12
for j in 724181 724182; do until st $j | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED"; do sleep 60; done; echo "pack $j $(st $j)"; grep -v "^+" logs/nano4/m55s0*-$j.out | tail -7; done
TAG=$(ls results/trunk_seq/es/val1/ | grep "^S-pc.*_s0.npz" | sed 's/_s0.npz//' | head -1); echo "S tag: $TAG"
J=$(sbatch --parsable --exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155] -J m55allS -c 12 -t 00:40:00 scripts/nano4/run.sbatch python -I logs/nano4/m55_allin.py --s-tag "$TAG")
until st $J | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED"; do sleep 20; done
echo "== allin with S"; grep -v "^+" logs/nano4/m55allS-$J.out | grep -v "^E1\|^E2A\|^E2B" | tail -20
