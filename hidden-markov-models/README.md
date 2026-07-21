# Hidden Markov Models — CpG Island Detection

Implementation of the core HMM algorithms (Viterbi, Forward, Backward, and
soft/posterior decoding) built from first principles, applied to detecting
CpG islands in real genomic DNA. Baum-Welch parameter learning is planned as
a follow-up and is not included yet.

## Files

- **`hmm.py`** — the `HiddenMarkovModel` class only. Supports any number of
  states, with both linear-probability and numerically stable log-space
  versions of every algorithm.
- **`cpg_cli.py`** — the application layer: FASTA/parameter file parsing,
  the command-line interface, and the interactive plot. Imports
  `HiddenMarkovModel` from `hmm.py` rather than duplicating it.

## Why broadcasting

The naive implementation of each recurrence is a double loop over "current
state" and "previous state" at every position. Since the recurrence is just
`previous_column * transition_matrix`, reshaping the previous column into a
`(states, 1)` vector lets numpy broadcast it against the full transition
matrix in one operation, replacing both loops with a single matrix multiply
and a `max`/`sum` reduction. Viterbi, Forward, and Backward all share this
structure, differing only in which reduction (`max` vs `sum`) is applied.

Genomic sequences are long enough that repeated multiplication of
probabilities underflows to zero, so every algorithm also has a log-space
version (`viterbi_log`, `forward_log`, `backward_log`) that replaces
multiplication with addition and uses `scipy.special.logsumexp` in place of
a plain sum, avoiding underflow without changing the math.

## Datasets

Both are real genomic sequences fetched from NCBI (GRCh38), covering the
promoter regions of well-characterized CpG-island-containing genes:

- **`tp53_promoter.fasta`** — chromosome 17, `chr17:7,686,000-7,689,000`
  (minus strand), the TP53 promoter, ~3001bp.
- **`gapdh_promoter.fasta`** — chromosome 12, `chr12:6,533,500-6,536,500`,
  the GAPDH promoter, ~3001bp.

## The 8-state model

A simple 2-state model (island vs. background), where each state emits
A/C/G/T according to its own frequency distribution, only captures overall
GC-richness — not the specific CpG dinucleotide enrichment that actually
defines a real island.

`cpg_hmm_parameters_8state.txt` fixes this by using one state per
nucleotide per region (`A+, C+, G+, T+` for island; `A-, C-, G-, T-` for
background). Each state deterministically emits its own letter, so all the
real signal moves into the transition matrix — `P(C+ → G+)` directly
encodes how often a C is followed by a G inside an island, versus
`P(C- → G-)` outside one. This lets the model detect real CpG enrichment
that a marginal-GC-content model misses entirely.

## Example output

```bash
python3 cpg_cli.py tp53_promoter.fasta cpg_hmm_parameters_8state.txt
```

produces `cpg_results.txt` (tab-separated: site, base, Viterbi call, and
soft-decoding P(CpG island)):

```
site    base    viterbi_call    p_cpg_island
536     A       N               0.6790
537     C       N               0.7451
538     C       I               0.8573
539     G       I               0.9332
540     C       I               0.9536
```

and `cpg_results.html`, an interactive Plotly chart — hover any point to see
its site, base, Viterbi call, and probability.

## Running it

```bash
python3 cpg_cli.py <fasta_file> <param_file>
```

Both `hmm.py` and `cpg_cli.py` need to be in the same directory. Requires
`numpy`, `scipy`, and `plotly`.
