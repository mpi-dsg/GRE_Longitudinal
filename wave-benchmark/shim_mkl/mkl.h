// Stand-in for Intel MKL. XIndex and FINEdex (as vendored in GRE) use exactly one MKL routine,
// LAPACKE_dgels, to fit their linear models. The test servers have no MKL and we have no root,
// so mkl_lapacke.h provides that routine with LAPACK's contract.
#pragma once
#include "mkl_lapacke.h"
