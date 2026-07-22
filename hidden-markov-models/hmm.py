import numpy as np
from scipy.special import logsumexp


class HiddenMarkovModel:
    """
    A Hidden Markov Model supporting Viterbi decoding, Forward/Backward
    evaluation, soft (posterior) decoding, and Baum-Welch parameter
    estimation — in both linear-probability and numerically stable
    log-probability forms. Supports any number of states.

    Parameters
    ----------
    outcome_states : list
        The emission alphabet, e.g. ["A", "C", "G", "T"].
    hidden_states : list
        The hidden state labels, e.g. ["I", "N"] or ["A+", "C+", "G+", "T+", "A-", "C-", "G-", "T-"].
    emission_matrix : np.ndarray, shape (n_states, n_symbols)
        emission_matrix[k, s] = P(emit symbol s | state k).
    transition_matrix : np.ndarray, shape (n_states, n_states)
        transition_matrix[j, k] = P(transition to state k | currently in state j).
    """

    def __init__(self, outcome_states, hidden_states, emission_matrix, transition_matrix):
        self.outcome_states = outcome_states
        self.hidden_states = hidden_states
        self.emission_matrix = emission_matrix
        self.transition_matrix = transition_matrix
        self.log_emission_matrix = np.log(emission_matrix)
        self.log_transition_matrix = np.log(transition_matrix)

    # ------------------------------------------------------------------
    # Viterbi decoding — linear probability version
    # ------------------------------------------------------------------
    def viterbi(self, outcome):
        emission_matrix = self.emission_matrix
        transition_matrix = self.transition_matrix

        viterbi_scores = np.zeros((len(self.hidden_states), len(outcome)))
        backpointers = np.zeros((len(self.hidden_states), len(outcome)), dtype=int)
        init_prob = 1 / len(self.hidden_states)

        for k in range(len(self.hidden_states)):
            viterbi_scores[k, 0] = init_prob * emission_matrix[k, self.outcome_states.index(outcome[0])]

        for site in range(1, len(outcome)):
            prev_scores = viterbi_scores[:, site - 1].reshape(-1, 1)
            candidate_scores = prev_scores * transition_matrix
            viterbi_scores[:, site] = np.max(candidate_scores, axis=0) * emission_matrix[:, self.outcome_states.index(outcome[site])]
            backpointers[:, site] = np.argmax(candidate_scores, axis=0)

        hidden_path = []
        current_state = np.argmax(viterbi_scores[:, -1])
        hidden_path.append(self.hidden_states[current_state])
        for site in range(len(outcome) - 1, 0, -1):
            current_state = backpointers[current_state, site]
            hidden_path.append(self.hidden_states[current_state])
        return hidden_path[::-1]

    # ------------------------------------------------------------------
    # Viterbi decoding — log-space version (numerically stable; max/argmax
    # require no special log-sum handling)
    # ------------------------------------------------------------------
    def viterbi_log(self, outcome):
        log_emission = self.log_emission_matrix
        log_transition = self.log_transition_matrix

        viterbi_scores = np.zeros((len(self.hidden_states), len(outcome)))
        backpointers = np.zeros((len(self.hidden_states), len(outcome)), dtype=int)
        log_init_prob = np.log(1 / len(self.hidden_states))

        for k in range(len(self.hidden_states)):
            viterbi_scores[k, 0] = log_init_prob + log_emission[k, self.outcome_states.index(outcome[0])]

        for site in range(1, len(outcome)):
            prev_scores = viterbi_scores[:, site - 1].reshape(-1, 1)
            candidate_scores = prev_scores + log_transition
            viterbi_scores[:, site] = np.max(candidate_scores, axis=0) + log_emission[:, self.outcome_states.index(outcome[site])]
            backpointers[:, site] = np.argmax(candidate_scores, axis=0)

        hidden_path = []
        current_state = np.argmax(viterbi_scores[:, -1])
        hidden_path.append(self.hidden_states[current_state])
        for site in range(len(outcome) - 1, 0, -1):
            current_state = backpointers[current_state, site]
            hidden_path.append(self.hidden_states[current_state])
        return hidden_path[::-1]

    # ------------------------------------------------------------------
    # Forward algorithm
    # ------------------------------------------------------------------
    def forward(self, outcome):
        emission_matrix = self.emission_matrix
        transition_matrix = self.transition_matrix

        forward_scores = np.zeros((len(self.hidden_states), len(outcome)))
        initial_prob = 1 / len(self.hidden_states)

        for k in range(len(self.hidden_states)):
            forward_scores[k, 0] = initial_prob * emission_matrix[k, self.outcome_states.index(outcome[0])]

        for site in range(1, len(outcome)):
            prev_scores = forward_scores[:, site - 1].reshape(-1, 1)
            candidate_scores = prev_scores * transition_matrix
            forward_scores[:, site] = np.sum(candidate_scores, axis=0) * emission_matrix[:, self.outcome_states.index(outcome[site])]

        return np.sum(forward_scores[:, -1]), forward_scores

    def forward_log(self, outcome):
        log_emission = self.log_emission_matrix
        log_transition = self.log_transition_matrix

        forward_scores = np.zeros((len(self.hidden_states), len(outcome)))
        log_initial_prob = np.log(1 / len(self.hidden_states))

        for k in range(len(self.hidden_states)):
            forward_scores[k, 0] = log_initial_prob + log_emission[k, self.outcome_states.index(outcome[0])]

        for site in range(1, len(outcome)):
            prev_scores = forward_scores[:, site - 1].reshape(-1, 1)
            candidate_scores = prev_scores + log_transition
            forward_scores[:, site] = logsumexp(candidate_scores, axis=0) + log_emission[:, self.outcome_states.index(outcome[site])]

        return logsumexp(forward_scores[:, -1]), forward_scores

    # ------------------------------------------------------------------
    # Backward algorithm
    # ------------------------------------------------------------------
    def backward(self, outcome):
        emission_matrix = self.emission_matrix
        transition_matrix = self.transition_matrix

        backward_scores = np.zeros((len(self.hidden_states), len(outcome)))
        for k in range(len(self.hidden_states)):
            backward_scores[k, -1] = 1

        for site in range(len(outcome) - 2, -1, -1):
            next_scores = backward_scores[:, site + 1]
            candidate_scores = next_scores * transition_matrix * emission_matrix[:, self.outcome_states.index(outcome[site + 1])]
            backward_scores[:, site] = np.sum(candidate_scores, axis=1)
        return backward_scores

    def backward_log(self, outcome):
        log_emission = self.log_emission_matrix
        log_transition = self.log_transition_matrix

        backward_scores = np.zeros((len(self.hidden_states), len(outcome)))
        for k in range(len(self.hidden_states)):
            backward_scores[k, -1] = 0

        for site in range(len(outcome) - 2, -1, -1):
            next_scores = backward_scores[:, site + 1]
            candidate_scores = next_scores + log_transition + log_emission[:, self.outcome_states.index(outcome[site + 1])]
            backward_scores[:, site] = logsumexp(candidate_scores, axis=1)
        return backward_scores

    # ------------------------------------------------------------------
    # Soft (posterior) decoding
    # ------------------------------------------------------------------
    def soft_decoding(self, prob_of_sequence, forward_scores, backward_scores):
        return ((forward_scores * backward_scores) / prob_of_sequence)

    def soft_decoding_log(self, prob_of_sequence, forward_scores, backward_scores):
        return np.exp(((forward_scores + backward_scores) - prob_of_sequence))

    # ------------------------------------------------------------------
    # Baum-Welch parameter estimation — linear probability version
    # ------------------------------------------------------------------
    def baum_welch_learning(self, num_iterations, outcome):
        outcome_states = self.outcome_states
        hidden_states = self.hidden_states
        transition_matrix = self.transition_matrix
        emission_matrix = self.emission_matrix

        outcome_array = np.array(list(outcome))
        for iteration in range(num_iterations):
            hmm = HiddenMarkovModel(outcome_states, hidden_states, emission_matrix, transition_matrix)
            prob_of_sequence, forward_scores = hmm.forward(outcome)
            backward_scores = hmm.backward(outcome)
            posterior_probabilities = hmm.soft_decoding(prob_of_sequence, forward_scores, backward_scores)
            updated_emission_matrix = np.zeros((len(hidden_states), len(outcome_states)))
            for state_idx in range(len(hidden_states)):
                for symbol_idx in range(len(outcome_states)):
                    updated_emission_matrix[state_idx, symbol_idx] = posterior_probabilities[state_idx, :][outcome_array == outcome_states[symbol_idx]].sum()
                updated_emission_matrix[state_idx, :] = updated_emission_matrix[state_idx, :] / updated_emission_matrix[state_idx, :].sum()
            updated_transition_matrix = np.zeros((len(hidden_states), len(hidden_states)))
            for site in range(len(outcome) - 1):
                forward_current_state = forward_scores[:, site].reshape(-1, 1)
                backward_next_state = backward_scores[:, site + 1]
                emission_next_symbol = emission_matrix[:, outcome_states.index(outcome[site + 1])]
                updated_transition_matrix += (forward_current_state * backward_next_state * emission_next_symbol * transition_matrix) / prob_of_sequence
            row_totals = updated_transition_matrix.sum(axis=1).reshape(-1, 1)
            updated_transition_matrix = np.divide(updated_transition_matrix, row_totals,
                                                   out=np.full_like(updated_transition_matrix, 1 / len(hidden_states)),
                                                   where=row_totals != 0)
            transition_matrix = updated_transition_matrix
            emission_matrix = updated_emission_matrix
        return transition_matrix, emission_matrix

    # ------------------------------------------------------------------
    # Baum-Welch parameter estimation — log-space version (numerically
    # stable for longer sequences / more iterations)
    # ------------------------------------------------------------------
    def baum_welch_learning_log(self, num_iterations, outcome):
        outcome_states = self.outcome_states
        hidden_states = self.hidden_states
        transition_matrix = self.transition_matrix
        emission_matrix = self.emission_matrix

        outcome_array = np.array(list(outcome))
        for iteration in range(num_iterations):
            hmm = HiddenMarkovModel(outcome_states, hidden_states, emission_matrix, transition_matrix)
            log_prob_of_sequence, log_forward_scores = hmm.forward_log(outcome)
            log_backward_scores = hmm.backward_log(outcome)
            posterior_probabilities = hmm.soft_decoding_log(log_prob_of_sequence, log_forward_scores, log_backward_scores)
            updated_emission_matrix = np.zeros((len(hidden_states), len(outcome_states)))
            for state_idx in range(len(hidden_states)):
                for symbol_idx in range(len(outcome_states)):
                    updated_emission_matrix[state_idx, symbol_idx] = posterior_probabilities[state_idx, :][outcome_array == outcome_states[symbol_idx]].sum()
                updated_emission_matrix[state_idx, :] = updated_emission_matrix[state_idx, :] / updated_emission_matrix[state_idx, :].sum()
            log_transition_matrix = np.log(transition_matrix)
            log_emission_matrix = np.log(emission_matrix)
            log_T_grids = []
            for site in range(len(outcome) - 1):
                log_forward_current_state = log_forward_scores[:, site].reshape(-1, 1)
                log_backward_next_state = log_backward_scores[:, site + 1]
                log_emission_next_symbol = log_emission_matrix[:, outcome_states.index(outcome[site + 1])]
                log_T_lk_site = (log_forward_current_state + log_backward_next_state + log_emission_next_symbol + log_transition_matrix) - log_prob_of_sequence
                log_T_grids.append(log_T_lk_site)
            log_T_stack = np.stack(log_T_grids, axis=0)
            log_transition_totals = logsumexp(log_T_stack, axis=0)
            updated_transition_matrix = np.exp(log_transition_totals)
            row_totals = updated_transition_matrix.sum(axis=1).reshape(-1, 1)
            updated_transition_matrix = np.divide(
                updated_transition_matrix, row_totals,
                out=np.full_like(updated_transition_matrix, 1 / len(hidden_states)),
                where=row_totals != 0
            )
            transition_matrix = updated_transition_matrix
            emission_matrix = updated_emission_matrix
        return transition_matrix, emission_matrix
