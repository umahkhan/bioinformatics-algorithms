#!/usr/bin/env python3
"""
randomizedMotifSearch.py

Usage:
    randomizedMotifSearch.py -i=100000 -k=13 <somefile.fa >someOutputFile.fa
    randomizedMotifSearch.py -i=100000 -k=13 -m=motifs.txt <somefile.fa >someOutputFile.fa

Runs the Randomized Motif Search algorithm (with pseudocounts) across
many random restarts to avoid local minima, then reports the consensus
sequence of the best motif set found and its entropy score. If -m is
given, also writes the recovered best motif instance per sequence to a
file, one per line — feed that file into motifLogo.py to build a
sequence logo.
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
    Encapsulates the Randomized Motif Search algorithm with pseudocounts,
    run over multiple random restarts to reduce the risk of converging
    on a local minimum.
    """

    BASES = ("a", "c", "g", "t")

    def __init__(self, sequences, kmerSize, iterations):
        self.sequences = [s.lower() for s in sequences]
        self.kmerSize = kmerSize
        self.iterations = iterations
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
        kmerScores = {}
        for i in range(len(sequence) - self.kmerSize + 1):
            kmer = sequence[i:i + self.kmerSize]
            kmerScore = reduce(
                lambda x, y: x * y,
                [profileMatrix[position][base] for position, base in enumerate(kmer)]
            )
            kmerScores[kmer] = kmerScore
        return max(kmerScores, key=kmerScores.get)

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

    # ---------- search ----------

    def runSingleSearch(self):
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

    def run(self):
        """Runs runSingleSearch() over many random restarts and keeps the
        best (lowest-scoring) motif set found across all of them.
        Returns (consensus, score, bestMotifs) — bestMotifs is the actual
        recovered motif instance per input sequence, useful for downstream
        visualization (e.g. a sequence logo) rather than just the consensus."""
        bestMotifs, bestScore = self.runSingleSearch()

        for _ in range(self.iterations):
            motifs, score = self.runSingleSearch()
            if score < bestScore:
                bestScore = score
                bestMotifs = motifs

        bestProfile = self.buildProfile(bestMotifs)
        consensus = self.getConsensus(bestProfile)
        return consensus, bestScore, bestMotifs


class CommandLine:
    """Parses command-line arguments for randomizedMotifSearch.py."""

    def __init__(self, inOpts=None):
        parser = argparse.ArgumentParser(
            description="Randomized Motif Search with multiple random restarts."
        )
        parser.add_argument(
            "-i", dest="iterations", type=int, required=True,
            help="Number of random-restart iterations to run."
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

    fastaReader = FastaReader('')  # reads from stdin
    sequences = [sequence for header, sequence in fastaReader.readFasta()]

    searcher = RandomizedMotifSearch(sequences, kmerSize, iterations)
    consensus, score, bestMotifs = searcher.run()

    print(">consensus")
    print(consensus)
    print("score={:.4f}".format(score))

    if commandLine.args.motifsFile:
        with open(commandLine.args.motifsFile, 'w') as motifsOut:
            for motif in bestMotifs:
                motifsOut.write(motif + "\n")
        print("motif instances written to {}".format(commandLine.args.motifsFile), file=sys.stderr)


if __name__ == "__main__":
    main()
