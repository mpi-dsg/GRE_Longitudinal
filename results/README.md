# Results catalog

Where experiment outputs live. Runners are under `../experiments/`; binaries and caches are not listed here.

## Layout

| Path | Contents |
|------|----------|
| `longbench/longitudinal/` | Main longitudinal CSVs (paper / VLDB runs) |
| `gre_stock/growth_rho025/` | Stock GRE growth baselines at ρ=0.25 |
| `exploratory/` | Ablations, smoke tests, competitor prototypes — not paper tables |
| `ops_cache/` | Shared op-sequence binaries (`.bin`) |
| `logs/` | Per-run console logs |

## Filename conventions (`longbench/longitudinal/`)

Pattern: `{dataset}_{threads}_{workload}[_variant]_seed{S}.csv`

| Token | Meaning |
|-------|---------|
| `libio` / `planet` / `covid` / `osm` / `uniform` / `books_800m` | Dataset |
| `st` | Single-threaded (`thread_num=1`) |
| `mt` | Multi-threaded concurrent indexes |
| `t8` / `t16` / `t24` / `t32` | Thread count (MT only) |
| `half` | Balanced R/I = 0.5/0.5 |
| `readheavy` | R/I = 0.8/0.2 |
| `writeheavy` | R/I = 0.2/0.8 |
| `rho010` / `rho050` | `init_table_ratio` 0.10 / 0.50 (default half ≡ ρ=0.25) |
| `m25m` / `m100m` | Ops per batch (default half ≡ M=50M) |
| `rebuild_every2` | Periodic rebuild every 2 batches |
| `rebuild_every2_v2` | Same protocol; rebuild log splits snapshot/sort/kv/bulk |
| `seed1866` / `seed5` / `seed72` | RNG seed |

Default protocol unless a variant says otherwise: N=200M keys, ρ=0.25, M=50M ops/batch, never-reset between batches.

Mix tags in RID/RIU/… names: `r40i40d20` = 40% read / 40% insert / 20% delete (same idea for `u`=update, `s`=scan).

---

## `longbench/longitudinal/` — by experiment

### Growth-only baselines (HotStorage-style)

| Files | What |
|-------|------|
| `{ds}_st_half_seed*.csv` | Default ST balanced trajectory (ρ=0.25, M=50M, B=6) |
| `{ds}_st_readheavy_seed*.csv` | ST read-heavy spectrum |
| `{ds}_st_writeheavy_seed*.csv` | ST write-heavy spectrum |
| `libio_st_half_0.5_rerun.csv` | One-off libio half rerun |

Datasets with full half/RH/WH: libio, planet, covid, osm, uniform.

### Concurrent (MT)

| Files | What |
|-------|------|
| `{ds}_mt_half_t{8,16,24,32}_seed*.csv` | Concurrent cast (alexol, sali, finedex, artolc, btreeolc), balanced, 200M |

Datasets: libio, planet, covid, osm. Some FINEdex rows omitted where OpCheck failed.

### Batch size (M)

| Files | What |
|-------|------|
| `libio_st_half_m25m_seed*.csv` | M=25M ops/batch (B=12) |
| `libio_st_half_m100m_seed*.csv` | M=100M ops/batch (B=3) |

### Bulk-load fraction (ρ)

| Files | What |
|-------|------|
| `{libio,planet}_st_rho010_seed*.csv` | ρ=0.10 (B=7) |
| `{libio,planet}_st_rho050_seed*.csv` | ρ=0.50, GRE-matched (B=4) |

Use `*_st_half_*` for ρ=0.25.

### Rebuild

| Files | What |
|-------|------|
| `{libio,planet}_st_half_rebuild_every2_seed*.csv` | Throughput with rebuild every 2 batches |
| `rebuild_log_*_every2_seed*.csv` | Rebuild wall time (bulk only) |
| `*_rebuild_every2_v2_*` + `rebuild_log_*_v2_*` | Same runs; log has snapshot/sort/kv/bulk/total |
| `rebuild_oracle_libio.csv` | Matched-size fresh vs aged oracle point |

### 800M scale

| Files | What |
|-------|------|
| `longitudinal_cross_hardness_800m_st_{books,osm}_800m_t1_seed*.csv` | ST growth @ 800M (books vs osm) |
| `longitudinal_growth_800m_mt_{books,osm}_800m_t8_seed*.csv` | MT t=8 @ 800M |
| `books_800m_st_readheavy.csv` | Partial 800M read-heavy (libio-style name on books) |

### Mixed-op longitudinal (audit-limited casts)

| Files | What |
|-------|------|
| `rid_*_r40i40d20_seed*.csv` | Read/insert/delete |
| `riu_longitudinal_*_r40i40u20_seed*.csv` | Read/insert/update |
| `riud_nomem_lipp_dili_*_r30i40u20d10_seed*.csv` | R/I/U/D without `--memory` (lipp+dili) |
| `longbench_crud_libio*.csv` | Early CRUD longbench on libio |

---

## `gre_stock/`

| Files | What |
|-------|------|
| `growth_rho025/{libio,planet}_st_rho025_seed*.csv` | Stock GRE single-shot-style growth at ρ=0.25 |

---

## `exploratory/`

Not for paper tables. Includes:

- `alex_*`, `alex_rls_*`, `blade_*`, `glade_*`, `rail_*`, `hyper_*`, `beat_alex_*`, `dytis_*`, `lipp_vs_alex_*` — index prototypes / bakeoffs (often 5M smoke)
- `*_SMOKE8M.csv` — short 800M-protocol smokes
- `ris_*_r40i40s20_*` — read/insert/scan mixes (not yet promoted to `longbench/`)
- Loose `riud_nomem_*` / duplicate RID copies from early runs

Prefer `longbench/longitudinal/` when both exist.

---

## Scripts → CSV map

| Script | Writes |
|--------|--------|
| `experiments/18_spectrum_growth_st.sh` | `{ds}_st_{half,readheavy,writeheavy}_*` |
| `experiments/19_spectrum_800m_st.sh` | 800M spectrum |
| `experiments/20_spectrum_covid_osm_uniform_st.sh` | covid/osm/uniform spectrum |
| `experiments/21_batchsize_sweep_libio_st.sh` | `libio_st_half_m{25,100}m_*` |
| `experiments/22_rho_init_sweep_st.sh` | `*_st_rho{010,050}_*` |
| `experiments/12_*` / `11_*` | 800M ST / MT |
| `experiments/14`–`17_*` | RIU / RIUD / RID / RIS |
| concurrent MT runner (ad hoc) | `*_mt_half_t*_seed*` |
