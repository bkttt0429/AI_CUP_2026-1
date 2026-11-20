cd ~/BK_AICUP_2026-1
until [ -f logs/nano4/m51b1-719078.out ] && grep -q "exit [0-9]" logs/nano4/m51b1-719078.out; do sleep 10; done
echo "== check 1"; grep -v "^+" logs/nano4/m51b1-719078.out
until [ -f logs/nano4/m51bagg-719083.out ] && grep -q "exit [0-9]" logs/nano4/m51bagg-719083.out; do sleep 15; done
echo "== check 2"; grep -v "^+" logs/nano4/m51bagg-719083.out
for v in 1 2 3 4; do f=$(ls logs/nano4/m51bh$v-*.out | tail -1); grep -h "Traceback\|Error" "$f" | head -3; done
