#!/usr/bin/env python3
"""
motifLogo.py

Builds a sequence logo (information-content-scaled stacked letters) from a
set of recovered motif instances, using the same profile-construction logic
as randomizedMotifSearch.py.

Usage:
    motifLogo.py -o logo.png < recovered_motifs.fa
    (one motif instance per line, or FASTA formatted)
"""

import sys
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Affine2D

BASES = ("a", "c", "g", "t")
COLOR_SCHEME = {"a": "#109648", "c": "#255C99", "g": "#F7B32B", "t": "#D62839"}
FONT = FontProperties(family="monospace", weight="bold")


def buildProfile(motifList):
    """Same profile-construction logic as RandomizedMotifSearch.buildProfile,
    with pseudocounts of 1 so a single base doesn't dominate a short sample."""
    profileMatrix = []
    for position in range(len(motifList[0])):
        countDict = {base: 1 for base in BASES}
        for motif in motifList:
            countDict[motif[position]] += 1
        countSum = sum(countDict.values())
        profile = {base: count / countSum for base, count in countDict.items()}
        profileMatrix.append(profile)
    return profileMatrix


def informationContent(position, numSymbols=4):
    """Bits of information at a position: max entropy (log2(4)=2 bits)
    minus the observed entropy. High IC = highly conserved position."""
    entropy = -sum(p * math.log2(p) for p in position.values() if p > 0)
    return math.log2(numSymbols) - entropy


def letterAt(letter, x, y, height, ax):
    """Draws a single letter glyph scaled to the given height, stacked at (x, y)."""
    textPath = TextPath((0, 0), letter.upper(), size=1, prop=FONT)
    bbox = textPath.get_extents()
    glyphWidth = bbox.width if bbox.width > 0 else 1
    glyphHeight = bbox.height if bbox.height > 0 else 1

    transform = (
        Affine2D()
        .translate(-bbox.x0, -bbox.y0)
        .scale(1.0 / glyphWidth, height / glyphHeight)
        .translate(x, y)
    )
    patch = PathPatch(
        transform.transform_path(textPath),
        facecolor=COLOR_SCHEME.get(letter, "gray"),
        edgecolor="none",
    )
    ax.add_patch(patch)


def plotLogo(profileMatrix, outFile, title="Motif Logo"):
    numPositions = len(profileMatrix)
    fig, ax = plt.subplots(figsize=(max(6, numPositions * 0.7), 3.5))

    maxHeight = math.log2(4)  # 2 bits, tallest possible column
    for pos, position in enumerate(profileMatrix):
        ic = informationContent(position)
        # sort bases by height so the tallest letter is drawn on top
        sortedBases = sorted(position.items(), key=lambda kv: kv[1])
        yOffset = 0
        for base, freq in sortedBases:
            letterHeight = freq * ic
            if letterHeight > 0.01:
                letterAt(base, pos, yOffset, letterHeight, ax)
            yOffset += letterHeight

    ax.set_xlim(-0.5, numPositions - 0.5)
    ax.set_ylim(0, maxHeight)
    ax.set_xticks(range(numPositions))
    ax.set_xticklabels([str(i + 1) for i in range(numPositions)])
    ax.set_ylabel("Bits")
    ax.set_xlabel("Position")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(outFile, dpi=200)
    print(f"Logo saved to {outFile}")


def readMotifs(fileHandle):
    """Accepts either one motif per line, or FASTA-formatted input."""
    motifs = []
    for line in fileHandle:
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        motifs.append(line.lower())
    return motifs


def main(inOpts=None):
    parser = argparse.ArgumentParser(
        description="Generate a sequence logo from a set of motif instances."
    )
    parser.add_argument("-o", dest="outFile", default="motif_logo.png",
                         help="Output image file path.")
    parser.add_argument("-t", dest="title", default="Motif Logo",
                         help="Title for the logo plot.")
    args = parser.parse_args(inOpts)

    motifs = readMotifs(sys.stdin)
    if not motifs:
        sys.exit("No motif instances found on stdin.")

    profileMatrix = buildProfile(motifs)
    plotLogo(profileMatrix, args.outFile, args.title)


if __name__ == "__main__":
    main()
