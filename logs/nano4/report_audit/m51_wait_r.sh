cd ~/BK_AICUP_2026-1
until [ -f logs/nano4/m51ragg-719050.out ] && grep -q "exit [0-9]" logs/nano4/m51ragg-719050.out; do sleep 20; done
echo "== m51ragg"; grep -v "^+" logs/nano4/m51ragg-719050.out
