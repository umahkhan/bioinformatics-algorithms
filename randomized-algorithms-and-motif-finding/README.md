# Randomized Algorithms & Motif Finding

Implementations of two classic motif-discovery algorithms — **Randomized
Motif Search** and **Gibbs Sampler** — both with pseudocounts and multiple
random restarts, unified behind a single command-line tool. Applied to
discovering conserved regulatory motifs (e.g. CRISPR promoter/BRE regions)
upstream of a set of related genomic sequences.

## Files

- **`motifSearch.py`** — the `RandomizedMotifSearch` class and CLI. Supports
  both algorithms, selected with a single flag.
- **`motifLogo.py`** — builds a sequence logo image from the recovered
  motif instances (feed it the `-m` output from `motifSearch.py`).

## The two algorithms

Both start from random k-mers and iteratively refine a motif set toward
one that's more conserved, but differ in *how* each refinement step picks
a new candidate motif:

- **Randomized Motif Search** (`randomizedMotifSearch`) — at each step,
  rebuilds a profile from the current best motif set and replaces
  *every* sequence's motif with its single most probable k-mer under that
  profile (deterministic argmax). Repeats until no further improvement is
  found, then restarts from a new random starting point.
- **Gibbs Sampler** (`runSingleGibbsSampler`) — at each step, picks *one*
  random sequence, builds a profile from all the *other* sequences'
  current motifs, then replaces that one sequence's motif with a
  **profile-weighted random draw** rather than the single best k-mer —
  so lower-probability k-mers still have a real (if smaller) chance of
  being picked. This makes Gibbs Sampler less prone to a single early
  step locking in a bad motif and never escaping it.

Both share the exact same outer loop — run one full trajectory, then keep
repeating and tracking whichever attempt scored best — implemented once in
`run()`, which dispatches to the right single-trajectory method based on
the selected algorithm.

## Scoring

Motif sets are scored by **Shannon entropy** of the profile: for each
position, `-Σ p·log2(p)` across the four bases, summed across all
positions. Lower score = more conserved / less uncertain profile = better
motif set.

## Usage

```bash
python3 motifSearch.py -a=randomized -i=100000 -k=13 <input.fa >output.fa
python3 motifSearch.py -a=gibbs -i=100 -k=13 <input.fa >output.fa
python3 motifSearch.py -a=gibbs -i=100 -k=13 -m=motifs.txt <input.fa >output.fa
```

- `-a` — algorithm: `randomized` or `gibbs` (default: `randomized`)
- `-i` — for `randomized`: number of random restarts. For `gibbs`: N
  (substitution steps per single Gibbs run) **and** the number of random
  restarts of the full Gibbs run — one value drives both.
- `-k` — k-mer (motif) length to search for
- `-m` — optional path to write the recovered best motif instances, one
  per line; feed this into `motifLogo.py` to generate a sequence logo

Input is FASTA, read from stdin. Output prints the consensus motif and its
score to stdout.

## Example

```bash
python3 motifSearch.py -a=gibbs -i=100 -k=8 -m=motifs.txt <input.fa
python3 motifLogo.py -o=logo.png -t "Motif" <motifs.txt
```

```
>consensus
tctcgggg
algorithm=gibbs
score=13.3762
```
