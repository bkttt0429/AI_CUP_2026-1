#!/bin/bash
main() {
  echo "=== m183_time2_fix"; python -I logs/nano4/m183_time2_fix.py || echo "exit $?"
  echo "=== m187_headroom_fix"; python -I logs/nano4/m187_headroom_fix.py || echo "exit $?"
  echo "ALL DONE"
}
main "$@"; exit $?
