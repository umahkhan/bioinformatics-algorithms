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


def run_detection(hmm, outcome, island_indices):
    """Runs Viterbi + soft decoding for a given (already-parameterized) HMM."""
    raw_hidden_path = hmm.viterbi_log(outcome)
    viterbi_call = ["I" if hmm.hidden_states.index(s) in island_indices else "N" for s in raw_hidden_path]

    log_prob, fwd = hmm.forward_log(outcome)
    bwd = hmm.backward_log(outcome)
    posteriors = hmm.soft_decoding_log(log_prob, fwd, bwd)  # shape (states, sites) -- no transpose in this hmm.py
    p_cpg_island = posteriors[island_indices, :].sum(axis=0)

    return viterbi_call, p_cpg_island


def matrix_to_html_table(matrix, row_labels, col_labels, title):
    """Renders a small numpy matrix as an HTML table, for showing HMM parameters above a plot."""
    header_cells = "".join(f"<th style='padding:4px 10px;border-bottom:1px solid #ccc;'>{c}</th>" for c in col_labels)
    rows_html = ""
    for i, row_label in enumerate(row_labels):
        cells = "".join(f"<td style='padding:4px 10px;'>{matrix[i, j]:.3f}</td>" for j in range(len(col_labels)))
        rows_html += f"<tr><td style='padding:4px 10px;font-weight:bold;'>{row_label}</td>{cells}</tr>"
    return f"""
    <div style="margin-bottom:6px;font-weight:600;color:#2c2c2a;">{title}</div>
    <table style="border-collapse:collapse;font-family:sans-serif;font-size:13px;margin-bottom:18px;">
        <tr><td></td>{header_cells}</tr>
        {rows_html}
    </table>
    """


def plot_cpg_interactive(outcome, viterbi_call, p_cpg_island, title):
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
        title=dict(text=title, font=dict(size=15, color="#2c2c2a")),
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


def build_combined_html(outcome, hidden_states, outcome_states,
                         before_transition, before_emission, before_call, before_prob,
                         after_transition, after_emission, after_call, after_prob,
                         num_iterations, html_path):
    fig_before = plot_cpg_interactive(outcome, before_call, before_prob, "CpG island detection — before parameter optimization")
    fig_after = plot_cpg_interactive(outcome, after_call, after_prob, f"CpG island detection — after Baum-Welch parameter optimization ({num_iterations} iterations)")

    before_transition_table = matrix_to_html_table(before_transition, hidden_states, hidden_states, "Transition matrix (initial guess)")
    before_emission_table = matrix_to_html_table(before_emission, hidden_states, outcome_states, "Emission matrix (initial guess)")
    after_transition_table = matrix_to_html_table(after_transition, hidden_states, hidden_states, f"Transition matrix (learned via Baum-Welch, {num_iterations} iterations)")
    after_emission_table = matrix_to_html_table(after_emission, hidden_states, outcome_states, f"Emission matrix (learned via Baum-Welch, {num_iterations} iterations)")

    plotly_js = "cdn"
    before_div = fig_before.to_html(full_html=False, include_plotlyjs=plotly_js)
    after_div = fig_after.to_html(full_html=False, include_plotlyjs=False)

    html = f"""
    <html>
    <head><meta charset="utf-8"><title>CpG Island Detection — Baum-Welch Parameter Estimation</title></head>
    <body style="font-family:sans-serif;max-width:900px;margin:30px auto;">
        <h1 style="color:#2c2c2a;">CpG Island Detection with Baum-Welch Parameter Estimation</h1>
        <p style="color:#555;">
            This report shows CpG island detection <b>before</b> parameter optimization (using the
            starting parameter guesses) and <b>after</b> running Baum-Welch learning for
            <b>{num_iterations} iteration(s)</b> to re-estimate the transition and emission matrices
            directly from this sequence.
        </p>

        <h2 style="color:#2c2c2a;border-bottom:2px solid #199e70;padding-bottom:4px;">Before parameter optimization</h2>
        {before_transition_table}
        {before_emission_table}
        {before_div}

        <h2 style="color:#2c2c2a;border-bottom:2px solid #2a78d6;padding-bottom:4px;margin-top:40px;">
            After parameter optimization (Baum-Welch, {num_iterations} iterations)
        </h2>
        {after_transition_table}
        {after_emission_table}
        {after_div}
    </body>
    </html>
    """

    with open(html_path, "w") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description="Detect CpG islands in a DNA sequence using an HMM, "
                                                   "with optional Baum-Welch parameter estimation.")
    parser.add_argument("fasta_path", help="Path to input FASTA file")
    parser.add_argument("param_file", help="Path to HMM parameter file (states + transition + emission matrices)")
    parser.add_argument("-n", "--baum-welch-iterations", type=int, default=10,
                         help="Number of Baum-Welch iterations to run for parameter estimation (default: 10)")
    parser.add_argument("-o", "--output-name", default=None,
                         help="Base name for output files (produces <name>.txt and <name>.html). "
                              "Defaults to the input FASTA filename (without extension).")
    args = parser.parse_args()

    output_name = args.output_name or os.path.splitext(os.path.basename(args.fasta_path))[0]
    output_path = f"{output_name}.txt"
    html_path = f"{output_name}.html"

    outcome = list(parse_fasta(args.fasta_path).values())[0]
    params = parse_hmm_params(args.param_file)
    hidden_states = params["hidden_states"]
    outcome_states = params["outcome_states"]
    island_indices = get_island_indices(hidden_states)

    # --- BEFORE parameter optimization ---
    hmm_before = HiddenMarkovModel(outcome_states, hidden_states, params["emission_matrix"], params["transition_matrix"])
    before_call, before_prob = run_detection(hmm_before, outcome, island_indices)

    # --- Baum-Welch parameter estimation ---
    print(f"Running Baum-Welch learning for {args.baum_welch_iterations} iteration(s)...")
    learned_transition, learned_emission = hmm_before.baum_welch_learning_log(args.baum_welch_iterations, outcome)

    # --- AFTER parameter optimization ---
    hmm_after = HiddenMarkovModel(outcome_states, hidden_states, learned_emission, learned_transition)
    after_call, after_prob = run_detection(hmm_after, outcome, island_indices)

    # --- write text table (final, post-optimization results) ---
    with open(output_path, "w") as f:
        f.write("site\tbase\tviterbi_call\tp_cpg_island\n")
        for i in range(len(outcome)):
            f.write(f"{i}\t{outcome[i]}\t{after_call[i]}\t{after_prob[i]:.4f}\n")
    print(f"Wrote {len(outcome)} rows to {output_path} (post-Baum-Welch results)")

    # --- write combined before/after HTML report ---
    build_combined_html(
        outcome, hidden_states, outcome_states,
        params["transition_matrix"], params["emission_matrix"], before_call, before_prob,
        learned_transition, learned_emission, after_call, after_prob,
        args.baum_welch_iterations, html_path,
    )
    print(f"Wrote interactive before/after report to {html_path}")
    webbrowser.open(f"file://{os.path.abspath(html_path)}")


if __name__ == "__main__":
    main()
