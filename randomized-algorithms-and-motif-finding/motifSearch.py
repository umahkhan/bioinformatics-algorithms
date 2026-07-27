#!/usr/bin/env python3
"""
motifSearch.py

Finds conserved motifs across a set of DNA sequences using either the
Randomized Motif Search algorithm or the Gibbs Sampler algorithm (both
with pseudocounts), each run over multiple random restarts to reduce the
risk of converging on a poor local optimum. Select the algorithm with -a.

Usage:
    motifSearch.py -a=randomized -i=100000 -k=13 <somefile.fa >someOutputFile.fa
    motifSearch.py -a=gibbs -i=100 -k=13 <somefile.fa >someOutputFile.fa
    motifSearch.py -a=gibbs -i=100 -k=13 -m=motifs.txt <somefile.fa >someOutputFile.fa

Arguments:
    -a  Algorithm to use: "randomized" or "gibbs" (default: randomized)
    -i  For randomized: number of random-restart iterations.
        For gibbs: N, the number of substitution steps per single Gibbs
        run, AND the number of random restarts of the full Gibbs run.
    -k  k-mer (motif) length to search for.
    -m  Optional output path to write the recovered best motif instances,
        one per line. Feed this into motifLogo.py to generate a sequence
        logo. If omitted, motif instances are not saved.

Reports the consensus sequence of the best motif set found and its
entropy score (lower is better).
"""

import sys
import argparse
import random
import math
from functools import reduce


class FastaReader:
    """Reads FASTA-formatted sequences from a file or stdin."""

    def __init__(self, fileName=''):
        self.fileName = fileName

    def doOpen(self):
        if self.fileName == '':
            return sys.stdin
        else:
            return open(self.fileName)

    def readFasta(self):
        header = ''
        sequence = ''
        with self.doOpen() as fileHandle:
            header = ''
            sequence = ''
            # skip to first fasta header
            line = fileHandle.readline()
            while not line.startswith('>'):
                line = fileHandle.readline()
            header = line[1:].rstrip()
            for line in fileHandle:
                if line.startswith('>'):
                    yield header, sequence
                    header = line[1:].rstrip()
                    sequence = ''
                else:
                    sequence += ''.join(line.rstrip().split()).upper()
        yield header, sequence


class RandomizedMotifSearch:
    """
    Encapsulates both the Randomized Motif Search algorithm and the Gibbs
    Sampler algorithm, each with pseudocounts and multiple random restarts
    to reduce the risk of converging on a local minimum.
    """

    BASES = ("a", "c", "g", "t")

    def __init__(self, sequences, kmerSize, iterations, algorithm="randomized"):
        self.sequences = [s.lower() for s in sequences]
        self.kmerSize = kmerSize
        self.iterations = iterations  # randomized: # restarts. gibbs: N (steps per run) AND # restarts
        self.algorithm = algorithm
        self.sequenceLength = len(self.sequences[0])

    # ---------- core subroutines ----------

    def getRandomKmer(self, sequence):
        randomStart = random.randint(0, len(sequence) - self.kmerSize)
        return sequence[randomStart:randomStart + self.kmerSize]

    def buildProfile(self, motifList):
        """Builds a profile matrix (list of dicts) with pseudocounts of 1."""
        profileMatrix = []
        for position in range(len(motifList[0])):
            countDict = {base: 1 for base in self.BASES}
            for motif in motifList:
                countDict[motif[position]] += 1
            countSum = sum(countDict.values())
            profile = {base: count / countSum for base, count in countDict.items()}
            profileMatrix.append(profile)
        return profileMatrix

    def getMostProbableKmer(self, sequence, profileMatrix):
        """Deterministic argmax pick — used by plain Randomized Motif Search."""
        kmerScores = {}
        for i in range(len(sequence) - self.kmerSize + 1):
            kmer = sequence[i:i + self.kmerSize]
            kmerScore = reduce(
                lambda x, y: x * y,
                [profileMatrix[position][base] for position, base in enumerate(kmer)]
            )
            kmerScores[kmer] = kmerScore
        return max(kmerScores, key=kmerScores.get)

    def getRandomProfileKmer(self, sequence, profileMatrix):
        """Profile-weighted random pick — used by Gibbs Sampler, so that
        lower-probability k-mers still have a real (if smaller) chance of
        being selected, rather than always taking the single best one."""
        kmerScores = {}
        for i in range(len(sequence) - self.kmerSize + 1):
            kmer = sequence[i:i + self.kmerSize]
            kmerScore = reduce(
                lambda x, y: x * y,
                [profileMatrix[position][base] for position, base in enumerate(kmer)]
            )
            kmerScores[kmer] = kmerScore
        kmers = list(kmerScores.keys())
        weights = list(kmerScores.values())
        return random.choices(kmers, weights=weights, k=1)[0]

    def scoreProfile(self, profileMatrix):
        """Entropy-based score: sum of -sum(p*log2(p)) across all positions.
        Lower score = more conserved / better motif set."""
        totalScore = 0
        for position in profileMatrix:
            positionScore = 0
            for base, probability in position.items():
                positionScore += probability * math.log2(probability)
            totalScore += -positionScore
        return totalScore

    def getConsensus(self, profileMatrix):
        """Builds the consensus sequence: the most probable base at each position."""
        consensus = ""
        for position in profileMatrix:
            consensus += max(position, key=position.get)
        return consensus

    # ---------- single-trajectory implementations ----------

    def randomizedMotifSearch(self):
        """Runs one full randomized-restart-to-convergence trajectory."""
        motifs = [self.getRandomKmer(sequence) for sequence in self.sequences]
        bestMotifs = motifs

        while True:
            profile = self.buildProfile(bestMotifs)
            motifs = [self.getMostProbableKmer(sequence, profile) for sequence in self.sequences]
            newProfile = self.buildProfile(motifs)
            newScore = self.scoreProfile(newProfile)

            if newScore < self.scoreProfile(profile):
                bestMotifs = motifs
            else:
                return bestMotifs, self.scoreProfile(profile)

    def runSingleGibbsSampler(self):
        """Runs one full Gibbs Sampler trajectory: self.iterations (N)
        substitution steps, each replacing one randomly-chosen sequence's
        motif with a profile-weighted random pick built from all the
        other sequences' current motifs."""
        T = len(self.sequences)
        motifs = [self.getRandomKmer(sequence) for sequence in self.sequences]
        bestMotifs = motifs.copy()

        for _ in range(self.iterations):
            randomIndex = random.randint(0, T - 1)
            profile = self.buildProfile(motifs[:randomIndex] + motifs[randomIndex + 1:])
            newKmer = self.getRandomProfileKmer(self.sequences[randomIndex], profile)
            motifs[randomIndex] = newKmer

            if self.scoreProfile(self.buildProfile(motifs)) < self.scoreProfile(self.buildProfile(bestMotifs)):
                bestMotifs = motifs.copy()

        return bestMotifs, self.scoreProfile(self.buildProfile(bestMotifs))

    # ---------- unified run ----------

    def run(self):
        """Runs the selected algorithm's single-trajectory method (chosen
        via self.algorithm) over self.iterations random restarts, keeping
        the best (lowest-scoring) motif set found across all of them. Both
        algorithms share this exact outer restart-and-keep-best structure —
        only the single-trajectory method differs; self.iterations serves
        as the restart count either way. Returns (consensus, score, bestMotifs)."""
        if self.algorithm == "gibbs":
            singleTrajectory = self.runSingleGibbsSampler
        else:
            singleTrajectory = self.randomizedMotifSearch
        restartCount = self.iterations

        bestMotifs, bestScore = singleTrajectory()

        for _ in range(restartCount):
            motifs, score = singleTrajectory()
            if score < bestScore:
                bestScore = score
                bestMotifs = motifs

        bestProfile = self.buildProfile(bestMotifs)
        consensus = self.getConsensus(bestProfile)
        return consensus, bestScore, bestMotifs


class CommandLine:
    """Parses command-line arguments for motifSearch.py."""

    def __init__(self, inOpts=None):
        parser = argparse.ArgumentParser(
            description="Randomized Motif Search / Gibbs Sampler with multiple random restarts."
        )
        parser.add_argument(
            "-a", dest="algorithm", choices=["randomized", "gibbs"], default="randomized",
            help="Which algorithm to run: 'randomized' or 'gibbs' (default: randomized)."
        )
        parser.add_argument(
            "-i", dest="iterations", type=int, required=True,
            help="Randomized: number of random-restart iterations. "
                 "Gibbs: N, number of substitution steps per single Gibbs run, "
                 "AND number of random restarts of the full Gibbs run."
        )
        parser.add_argument(
            "-k", dest="kmerSize", type=int, required=True,
            help="k-mer (motif) length to search for."
        )
        parser.add_argument(
            "-m", dest="motifsFile", default=None,
            help="Optional output path to write the recovered best motif instances, "
                 "one per line (e.g. motifs.txt). Feed this into motifLogo.py to "
                 "generate a sequence logo. If omitted, motif instances are not saved."
        )
        self.args = parser.parse_args(inOpts)


def main(inOpts=None):
    commandLine = CommandLine(inOpts)
    kmerSize = commandLine.args.kmerSize
    iterations = commandLine.args.iterations
    algorithm = commandLine.args.algorithm

    fastaReader = FastaReader('')  # reads from stdin
    sequences = [sequence for header, sequence in fastaReader.readFasta()]

    searcher = RandomizedMotifSearch(sequences, kmerSize, iterations, algorithm)
    consensus, score, bestMotifs = searcher.run()

    print(">consensus")
    print(consensus)
    print("algorithm={}".format(algorithm))
    print("score={:.4f}".format(score))

    if commandLine.args.motifsFile:
        with open(commandLine.args.motifsFile, 'w') as motifsOut:
            for motif in bestMotifs:
                motifsOut.write(motif + "\n")
        print("motif instances written to {}".format(commandLine.args.motifsFile), file=sys.stderr)


if __name__ == "__main__":
    main()
