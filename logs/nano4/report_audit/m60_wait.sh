cd /home/u2507115/BK_AICUP_2026-1
st() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
for j in 725071 725072 725073 725074; do until st $j | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED"; do sleep 20; done; echo "$j $(st $j)"; done
for v in 1 2 3 4; do f=$(ls -t logs/nano4/m60cp$v-*.out | head -1); echo "== fold $v"; grep -E "^(mixed|pureB|pureW|sanity)|Error|Traceback" $f | cut -c1-260; done
J=$(sbatch --parsable --exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155] -J m60adj -c 12 -t 00:30:00 scripts/nano4/run.sbatch python -I logs/nano4/m60_colour_adjust.py)
until st $J | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED"; do sleep 15; done
echo "== adjust"; grep -v "^+" logs/nano4/m60adj-$J.out
