# DuckDB, tuned for on-chain trade data

This fork of [DuckDB](https://github.com/duckdb/duckdb) v1.5.6 is the query engine behind
a research pipeline that records Solana memecoin trades. It holds two pieces of work:

1. **An engine change.** `sum()` and `avg()` over unsigned 64-bit integers (`UBIGINT`) run
   **5.9-6.8x faster** and stay exact. Lamport amounts are unsigned 64-bit, so this is the
   arithmetic our trade queries do most.
2. **A storage change.** The trade tape moved from JSON lines to mint-sorted Parquet queried
   through DuckDB. Pulling one day's trades for a set of mints went from **2.24 s to 28 ms**,
   and the data shrank **5.6x**, with every row verified against its source.

The second change is the larger win. The first one is the engine fix we found along the way.

Branch: `trenches/v1.5.6`, based on upstream `v1.5.6` (`069cc9f9b`).

## 1. What we changed in DuckDB

### The problem

Profiling a real query (one capture day of 4.13M trades, summing `v_sol`) showed that
`sum(v_sol)` cost twice as much as `sum(slot)`, although both columns hold 8-byte integers
stored the same way. The difference is the type. `v_sol` is `UBIGINT`, so lamports stay
exact, and DuckDB had **no `sum` kernel for `UBIGINT`**. The binder inserted
`CAST(u AS HUGEINT)` and ran the general 128-bit add on every value, overflow checks
included. Those checks can never fire: a sum of unsigned 64-bit values needs 2^63 rows to
leave the 128-bit range.

### The math

Signed sums already use a fast 64-into-128-bit accumulator (Gubner et al.), which needs a
sign test because a negative input borrows from the upper word. An unsigned input can only
carry, so the sign test and its branch drop out and the carry is the comparison itself:

```
lower += v
upper += (lower < v)        // 1 when the low word wrapped, else 0
```

The second improvement is a statistics bound. When the column statistics prove

```
max(v) * rows < 2^64
```

the sum cannot wrap a 64-bit accumulator, so the planner switches to a plain `uint64_t` add
with no carry handling at all. Signed sums already had this check; unsigned sums never
reached it.

### The changes

| File | Change |
|---|---|
| `extension/core_functions/include/core_functions/aggregate/sum_helpers.hpp` | `UnsignedAddToHugeint`: the branch-free unsigned carry, with a constant-vector fast path |
| `extension/core_functions/aggregate/distributive/sum.cpp` | `sum(UBIGINT)` overload; `SumPropagateStats` handles `UINT64`; `sum_no_overflow` gains a `UBIGINT` variant |
| `extension/core_functions/aggregate/algebraic/avg.cpp` | `avg(UBIGINT)` on the same accumulator; the signed and unsigned average share one template |
| `test/sql/aggregate/aggregates/test_sum_ubigint.test` | Exactness tests: carries at 2^64-1, constant vectors, NULLs, empty input, 4 threads, groups, windows, the statistics path |

Return types are unchanged: `sum(UBIGINT)` still returns `HUGEINT`, and `avg(UBIGINT)` still
returns `DOUBLE`.

### Measured

20M rows, 1 thread, same commit built with and without the change:

| Query | Before | After | Speed-up |
|---|---|---|---|
| `sum(UBIGINT)`, statistics prove no wrap | 197 ms | 29 ms | **6.8x** |
| `sum(UBIGINT)`, values near 2^64 (carry on most rows) | 195 ms | 33 ms | **5.9x** |
| One real capture day, `sum(v_sol)` from Parquet | 126 ms | 94 ms | 1.34x |
| One real capture day, `sum(v_sol)` grouped by mint | 167 ms | 137 ms | 1.22x |

On real data the gain is smaller because decoding the Parquet column becomes the larger
cost. Every result equals `sum(u::HUGEINT)` exactly.

### Tried and not shipped

Each was measured against the official v1.5.6 binary from the same commit:

- **`-march=native` build: no gain** (0.85-1.07x, within noise) on a Xeon E3-1230 V2. That
  CPU has AVX but no AVX2, so there are no wider integer instructions to use. Worth
  re-testing on AVX2 hardware.
- **Base58 keys stored as binary:** dedupe 22% faster, joins unchanged. Not worth a new type
  that every string-expecting reader would have to learn.
- **Deferred carries (hi/lo split).** Write `v = hi * 2^32 + lo` and keep two plain 64-bit
  sums; neither can wrap before 2^32 additions, and one combine at the end restores the
  exact total. With no carry dependency the loop vectorises: standalone it ran **1.7x
  faster than the shipped unsigned kernel and 4.6x faster than DuckDB's signed sum**. Inside
  DuckDB it loses, because the aggregate API calls the kernel once per value and offers no
  end-of-vector hook to fold the halves; the flush counter that replaces the hook blocks
  vectorisation. It needs that hook in the aggregate API first.
- **Join Bloom filter.** DuckDB builds a Bloom filter for a hash join only when the build
  side carries a filter, a rule about query shape rather than selectivity. Forcing it on our
  500-mint join cut the rows passed to the join from 4.11M to 52k, a 0.57% false-positive
  rate, but saved no time: column decoding dominated the scan.
- **Profile-guided build:** the instrumented build exists; the A/B is not measured yet.

## 2. What we did with the data

### Before

The capture pipeline wrote every trade as one JSON line, about 2-3 GB per day, plus a
second stream of raw websocket frames and a cache of 1.46M small RPC response files.
Every reader scanned the files line by line in Python, usually to pull out the trades of a
set of mints.

### After

| Data | Before | After |
|---|---|---|
| Trade events, 11 days (30.6M rows) | 21.6 GB JSONL | **3.8 GB** Parquet (5.6x) |
| Operator observations, 815k rows of nested JSON | 4.2 GB JSONL | **125 MB** Parquet (34x) |
| RPC response cache | 1,458,371 files | 256 Parquet files, original bytes inside |

On one capture day:

| Read | Time |
|---|---|
| Python byte scanner over JSONL, 500 mints | 2.24 s |
| DuckDB over the same JSONL | 0.61 s (3.7x, the engine alone) |
| DuckDB over mint-sorted Parquet | **28 ms** (22x more, the layout) |
| One mint's whole tape | 6.5 ms |

The layout is where most of the gain comes from:

- **Columnar files.** A query reads only the columns it names. The transaction signature,
  54% of the bytes, is never touched by a mint query.
- **Sorted row groups.** Files are sorted by `mint, tau_ms`, so each 65,536-row group covers
  a narrow mint range. A lookup reads the footer's min/max per group and skips the rest.
- **Batched execution.** DuckDB processes values 2,048 at a time as machine arrays. On the
  same file and columns that is 84x faster than a Python loop over rows.
- **One catalog.** A small DuckDB file holds only view definitions over the Parquet files,
  so any number of processes can open it read-only and query every capture by name.

### Lossless by construction

- **Exact amounts.** One JSON key mixes integer lamports (up to 1.8e19, past int64) and UI
  floats, switched by `amount_units`. Each key splits into an exact `UBIGINT` column and a
  `DOUBLE` column, chosen by the JSON token's own type. A schema left to inference would
  have rounded the integers.
- **Row-by-row verification.** Every converted day is compared with the standard-library
  JSON parse of its source: values, types, order and unknown keys. `line_no` is the
  physical source line, so each row traces back to its original bytes.
- **A trusted fast path.** DuckDB's JSON reader is ~20x faster than a raw-line parse but
  skips blank lines and, with `ignore_errors`, turns a torn line into an all-NULL row.
  Verification caught that on the first test. The fast reader is trusted only when every
  physical line is accounted for; otherwise the exact reader redoes the file.
- **Originals kept.** The JSONL sources stay as the record. Recompressed archives replace a
  container only after the decompressed content's sha256 matches.

## Build and test

```
GEN=ninja BUILD_EXTENSIONS='core_functions;json;parquet' make
build/release/test/unittest test/sql/aggregate/aggregates/test_sum_ubigint.test
```

---

*The upstream DuckDB README follows.*

