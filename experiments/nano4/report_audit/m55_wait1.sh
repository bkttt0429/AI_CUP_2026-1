cd /home/u2507115/BK_AICUP_2026-1
until [ -f logs/nano4/m55all-724191.out ] && grep -q "exit [0-9]" logs/nano4/m55all-724191.out; do sleep 15; done
echo "== allin"; grep -v "^+" logs/nano4/m55all-724191.out | tail -30
until [ -f logs/nano4/m55scan-724192.out ] && grep -q "exit [0-9]" logs/nano4/m55scan-724192.out; do sleep 15; done
echo "== scan"; grep -v "^+" logs/nano4/m55scan-724192.out | tail -15
