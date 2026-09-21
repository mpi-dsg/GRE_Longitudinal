#!/bin/bash
set -euo pipefail
cd /home/GRE_Longitudinal
echo "==== LIBIO RIUD $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
./experiments/15_riud_longitudinal_libio_st.sh
echo "==== PLANET RIUD $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
./experiments/15_riud_longitudinal_planet_st.sh
echo "==== ALL RIUD DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
