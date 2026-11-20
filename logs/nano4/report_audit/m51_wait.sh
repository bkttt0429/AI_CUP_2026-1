cd ~/BK_AICUP_2026-1
until [ -f logs/nano4/m51hd-719044.out ] && grep -q "exit [0-9]" logs/nano4/m51hd-719044.out; do sleep 15; done
echo "== m51hd"; grep -v "^+" logs/nano4/m51hd-719044.out
until [ -f logs/nano4/m51agg-719043.out ] && grep -q "exit [0-9]" logs/nano4/m51agg-719043.out; do sleep 20; done
echo "== m51agg"; grep -v "^+" logs/nano4/m51agg-719043.out
