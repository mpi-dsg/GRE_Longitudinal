#!/bin/sh
# Build the wave benchmark against GRE's vendored index sources.
#
#   GRE=/path/to/GRE_Longitudinal sh build.sh
#
# Copies the competitor headers, applies the three patches in patches/ to the COPY, and
# compiles. The GRE checkout is never modified.
#
# Requirements: g++ with OpenMP, oneTBB headers and libtbb. oneTBB removed tbb/mutex.h and
# tbb/reader_writer_lock.h in 2021; shim/tbb supplies three-line replacements. Both are
# named only by typedefs these indexes never instantiate, which we verified.
set -e
# Living inside the GRE tree, default to the enclosing checkout.
GRE="${GRE:-$(cd "$(dirname "$0")/.." && pwd)}"
SRC="$GRE/src/competitor"
[ -d "$SRC" ] || { echo "no $SRC - is GRE set correctly?"; exit 1; }

# Only alexol is vendored in-tree. lippol, sali, btreeolc and artsync are submodules,
# so a fresh clone has the wrapper headers but none of the implementations.
missing=""
for d in lippol sali btreeolc artsync; do
  if [ -z "$(find "$SRC/$d/src" -name '*.h' 2>/dev/null | head -1)" ]; then
    missing="$missing $d"
  fi
done
if [ -n "$missing" ]; then
  echo "Missing sources for:$missing"
  echo
  echo "These are git submodules. In $GRE run:"
  echo "    git submodule update --init --recursive src/competitor/lippol/src \\"
  echo "        src/competitor/sali/src src/competitor/btreeolc/src src/competitor/artsync/src"
  exit 1
fi

# Optional:
#   XF=1          also build XIndex and FINEdex (-DWITH_XF). They need MKL's LAPACKE_dgels;
#                 shim_mkl/ supplies a least-squares stand-in.
#   SIDEALWAYS=1  also apply alexol-sidealways.patch (-DSIDEALWAYS), the ablation with a
#                 permanent per-node buffer. With the flag off it behaves as without the patch.
rm -rf _c && mkdir -p _c
for d in alexol lippol sali btreeolc artsync; do cp -R "$SRC/$d" _c/; done
if [ "${XF:-0}" = 1 ]; then for d in xindex finedex; do cp -R "$SRC/$d" _c/; done; fi
cp "$SRC/indexInterface.h" _c/

# In order: randomized bulk-load density (off unless `stagger`), contention counters
# (off unless `lockstats`), side buffer and background expansion (off unless `side` / `bg`).
# With no flags the binary behaves as unmodified ALEX-OL.
patch -p0 -d _c/alexol/src -i "$PWD/patches/alexol-stagger.patch"
patch -p1 -d _c/alexol/src -i "$PWD/patches/alexol-contention-stats.patch"
patch -p0 -d _c/alexol/src -i "$PWD/patches/alexol-sidebuf.patch"
DEFS="-DLOCK_STATS"
if [ "${SIDEALWAYS:-0}" = 1 ]; then
  patch -p0 -d _c/alexol/src -i "$PWD/patches/alexol-sidealways.patch"
  DEFS="$DEFS -DSIDEALWAYS"
fi
XFINC=""
if [ "${XF:-0}" = 1 ]; then
  DEFS="$DEFS -DWITH_XF"
  XFINC="-Ishim_mkl -I_c/xindex -I_c/xindex/src -I_c/finedex -I_c/finedex/src"
fi

g++ -O3 -std=c++17 -march=native -fopenmp $DEFS \
    -Ishim -I_c -I_c/alexol -I_c/alexol/src -I_c/lippol -I_c/lippol/src \
    -I_c/sali -I_c/sali/src -I_c/btreeolc -I_c/artsync $XFINC -I. \
    -o bench bench.cpp -lpthread -ltbb

echo "built ./bench"
echo
echo "smoke test (should print VERIFY ... 100.0000%):"
./bench alexol 200000 400000 50000 1866 4 smoke 2>&1 >/dev/null | grep VERIFY
