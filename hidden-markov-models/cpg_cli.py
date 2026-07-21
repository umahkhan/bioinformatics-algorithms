import argparse
import os
import webbrowser
import numpy as np
from hmm import HiddenMarkovModel


def parse_fasta(path):
    entries, header = {}, None
    for line in open(path):
        if line.startswith(">"):
            header = line[1:].strip()
            entries[header] = ""
        elif header:
            entries[header] += line.strip()
    return entries


def parse_hmm_params(path):
    blocks = [b.strip().splitlines() for b in open(path).read().split("--------") if b.strip()]
    tokens = [[line.split() for line in block] for block in blocks]
    return {
        "outcome_states": tokens[0][0],
        "hidden_states": tokens[1][0],
        "transition_matrix": np.array([row[1:] for row in tokens[2][1:]], dtype=float),
        "emission_matrix": np.array([row[1:] for row in tokens[3][1:]], dtype=float),
    }


def get_island_indices(hidden_states):
    """
    Works for both a simple 2-state model (states literally named "I"/"N")
    and an 8-state dinucleotide model (states named "A+","C+",...,"T-"),
    where island states are marked with a trailing "+".
    """
    if "I" in hidden_states:
        return [hidden_states.index("I")]
    island_indices = [i for i, s in enumerate(hidden_states) if s.endswith("+")]
    if not island_indices:
        raise ValueError(f"Could not identify island states in {hidden_states}. "
                          "Expected a state named 'I', or states ending in '+'.")
    return island_indices


def plot_cpg_interactive(outcome, viterbi_call, p_cpg_island):
    import plotly.graph_objects as go

    sites = list(range(len(outcome)))
    customdata = list(zip(outcome, viterbi_call))

    fig = go.Figure()

    blocks, start = [], None
    for i, s in enumerate(viterbi_call):
        if s == "I" and start is None:
            start = i
        if s != "I" and start is not None:
            blocks.append((start, i - 1))
            start = None
    if start is not None:
        blocks.append((start, len(viterbi_call) - 1))

    for s, e in blocks:
        fig.add_vrect(x0=s - 0.5, x1=e + 0.5, fillcolor="#199e70", opacity=0.14, line_width=0)

    fig.add_trace(go.Scatter(
        x=sites, y=p_cpg_island, mode="lines", name="P(CpG island)",
        line=dict(color="#2a78d6", width=2.5, shape="spline", smoothing=0.3),
        fill="tozeroy", fillcolor="rgba(42,120,214,0.08)",
        customdata=customdata,
        hovertemplate="<b>site %{x}</b><br>base: %{customdata[0]}<br>viterbi call: %{customdata[1]}<br>P(island): %{y:.1%}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", name="Viterbi call (island)",
        marker=dict(size=10, color="#199e70", opacity=0.4, symbol="square"),
        hoverinfo="skip",
    ))

    fig.update_layout(
        title=dict(text="CpG island detection", font=dict(size=15, color="#2c2c2a")),
        xaxis_title="Site (bp position)",
        yaxis_title="P(CpG island)",
        yaxis=dict(range=[0, 1.05], tickformat=".0%", gridcolor="#e1e0d9", zeroline=False),
        xaxis=dict(gridcolor="#f1f0ea", zeroline=False),
        template="plotly_white",
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", bordercolor="#e1e0d9", font=dict(size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
        margin=dict(l=50, r=20, t=60, b=40), height=340,
        plot_bgcolor="white",
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description="Detect CpG islands in a DNA sequence using an HMM.")
    parser.add_argument("fasta_path", help="Path to input FASTA file")
    parser.add_argument("param_file", help="Path to HMM parameter file (states + transition + emission matrices)")
    args = parser.parse_args()

    output_path = "cpg_results.txt"
    html_path = "cpg_results.html"

    outcome = list(parse_fasta(args.fasta_path).values())[0]
    params = parse_hmm_params(args.param_file)
    hmm = HiddenMarkovModel(params["outcome_states"], params["hidden_states"],
                             params["emission_matrix"], params["transition_matrix"])

    island_indices = get_island_indices(params["hidden_states"])

    raw_hidden_path = hmm.viterbi_log(outcome)
    viterbi_call = ["I" if params["hidden_states"].index(s) in island_indices else "N" for s in raw_hidden_path]

    log_prob, fwd = hmm.forward_log(outcome)
    bwd = hmm.backward_log(outcome)
    posteriors = hmm.soft_decoding_log(log_prob, fwd, bwd)  # shape (sites, states)
    p_cpg_island = posteriors[:, island_indices].sum(axis=1)

    with open(output_path, "w") as f:
        f.write("site\tbase\tviterbi_call\tp_cpg_island\n")
        for i in range(len(outcome)):
            f.write(f"{i}\t{outcome[i]}\t{viterbi_call[i]}\t{p_cpg_island[i]:.4f}\n")
    print(f"Wrote {len(outcome)} rows to {output_path}")

    fig = plot_cpg_interactive(outcome, viterbi_call, p_cpg_island)
    fig.write_html(html_path)
    print(f"Wrote interactive plot to {html_path}")
    webbrowser.open(f"file://{os.path.abspath(html_path)}")


if __name__ == "__main__":
    main()
