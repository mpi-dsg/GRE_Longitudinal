#!/bin/bash
set -uo pipefail
cd /home/GRE_Longitudinal
echo "==== LIBIO RIS $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
./experiments/17_ris_longitudinal_libio_st.sh || true
echo "==== PLANET RIS $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
./experiments/17_ris_longitudinal_planet_st.sh || true
echo "==== ALL RIS DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
