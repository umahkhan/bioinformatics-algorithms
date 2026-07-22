# Hidden Markov Models — CpG Island Detection

Implementation of the core HMM algorithms — Viterbi, Forward, Backward,
soft/posterior decoding, and Baum-Welch parameter estimation — built from
first principles, applied to detecting CpG islands in real genomic DNA.

## Files

- **`hmm.py`** — the `HiddenMarkovModel` class only. Supports any number of
  states, with both linear-probability and numerically stable log-space
  versions of every algorithm, including Baum-Welch.
- **`cpg_cli.py`** — the application layer: FASTA/parameter file parsing,
  the command-line interface, parameter optimization, and the interactive
  before/after report. Imports `HiddenMarkovModel` from `hmm.py` rather than
  duplicating it.

## Algorithms implemented

| Method | What it does |
|---|---|
| `viterbi` / `viterbi_log` | Most likely single hidden-state path (hard decoding) |
| `forward` / `forward_log` | Total probability of the sequence, P(x) |
| `backward` / `backward_log` | Mirror of Forward, run right-to-left; combined with Forward for soft decoding and Baum-Welch |
| `soft_decoding` / `soft_decoding_log` | Per-site posterior probability of each state (soft decoding) |
| `baum_welch_learning` / `baum_welch_learning_log` | Re-estimates the transition and emission matrices directly from the sequence, with no labeled training data |

## Why broadcasting

The naive implementation of each recurrence is a double loop over "current
state" and "previous state" at every position. Since the recurrence is just
`previous_column * transition_matrix`, reshaping the previous column into a
`(states, 1)` vector lets numpy broadcast it against the full transition
matrix in one operation, replacing both loops with a single matrix multiply
and a `max`/`sum` reduction. Viterbi, Forward, and Backward all share this
structure, differing only in which reduction is applied.

Genomic sequences are long enough that repeated multiplication of
probabilities underflows to zero, so every algorithm also has a log-space
version that replaces multiplication with addition and uses
`scipy.special.logsumexp` in place of a plain sum, avoiding underflow
without changing the math. Baum-Welch's transition update sums over every
position in the sequence, so its log-space version applies `logsumexp`
across positions as well, not just across states.

## Baum-Welch: learning parameters instead of guessing them

Rather than hand-picking transition/emission probabilities, Baum-Welch
iteratively refines a starting guess directly from the sequence itself,
using expectation-maximization:

1. **E-step** — run Forward/Backward with the current parameters, and use
   them to compute how plausible every transition and emission is, at
   every position in the sequence.
2. **M-step** — re-estimate the transition and emission matrices from those
   plausibility scores (a *weighted* count, since the true hidden path is
   unknown — unlike supervised parameter estimation, which counts a known
   path directly).
3. Repeat for a fixed number of iterations, feeding each round's updated
   parameters into the next round's E-step.

`cpg_cli.py` runs CpG island detection **twice** per invocation — once with
the starting parameter guess ("before"), and once after Baum-Welch has
re-estimated the parameters from the actual input sequence ("after") — so
you can directly see how much the initial guess mattered.

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

`cpg_cli.py` supports both automatically — it looks for a state literally
named `"I"`, and falls back to treating any state ending in `+` as an
island state otherwise.

## Datasets

Both are real genomic sequences fetched from NCBI (GRCh38), covering the
promoter regions of well-characterized CpG-island-containing genes:

- **`tp53_promoter.fasta`** — chromosome 17, `chr17:7,686,000-7,689,000`
  (minus strand), the TP53 promoter, ~3001bp.
- **`gapdh_promoter.fasta`** — chromosome 12, `chr12:6,533,500-6,536,500`,
  the GAPDH promoter, ~3001bp.
- **`cpg_test_sequence.fasta`** — a small 231bp synthetic sequence with a
  deliberately embedded CpG-rich region, used for quick sanity checks.

## Running it

```bash
python3 cpg_cli.py <fasta_file> <param_file> [-n ITERATIONS] [-o OUTPUT_NAME]
```

- `-n` / `--baum-welch-iterations` — number of Baum-Welch iterations to run
  (default: 10).
- `-o` / `--output-name` — base name for the output files, producing
  `<name>.txt` and `<name>.html`. Defaults to the input FASTA filename
  (without extension), so running on different sequences won't overwrite
  each other's results.

Example:
```bash
python3 cpg_cli.py tp53_promoter.fasta cpg_hmm_parameters_8state.txt -n 10
```

produces `tp53_promoter.txt` (final, post-Baum-Welch results) and
`tp53_promoter.html` — an interactive report showing, for both before and
after parameter optimization: the transition and emission matrices used,
and a hoverable plot of Viterbi calls plus soft-decoding P(CpG island) per
site.

Output table columns: site, base, Viterbi call, and soft-decoding
P(CpG island):

```
site    base    viterbi_call    p_cpg_island
536     A       N               0.6790
537     C       N               0.7451
538     C       I               0.8573
539     G       I               0.9332
540     C       I               0.9536
```

Both `hmm.py` and `cpg_cli.py` need to be in the same directory. Requires
`numpy`, `scipy`, and `plotly`.
