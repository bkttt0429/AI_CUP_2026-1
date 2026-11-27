#!/usr/bin/env bash
# M129 preparation: the m15 identifier files of the other voters' identifiers (SP, S2, S3c, S3); SP2e's exist (M126).
set -uo pipefail
main() {
  python -I scripts/m_series/m127_conf_family.py --stage fuse --tags SP S2 S3c S3 || return 1
  python -I scripts/m_series/m127_conf_family.py --stage write --tags SP S2 S3c S3 --write m15
}
main "$@"; exit $?
