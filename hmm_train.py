import json
import math
from pathlib import Path

import numpy as np

from main_fft import load_recordings, sample_folder


def log_transitions(transitions):
    transitions = np.asarray(transitions)
    result = np.full(transitions.shape, -np.inf)
    positive = transitions > 0
    result[positive] = np.log(transitions[positive])
    return result


def log_emission_probability(observation, mean, sd):
    mean, sd = np.asarray(mean), np.asarray(sd)
    difference = (np.asarray(observation) - mean) / sd
    return float(-np.log(sd * math.sqrt(2 * math.pi)).sum()
                 - 0.5 * np.sum(difference**2))


def emission_matrix(matrix, means, sds):
    means, sds = np.asarray(means), np.asarray(sds)
    if np.any(sds <= 0):
        raise ValueError("Standard deviations must be positive")
    difference = (matrix[:, None, :] - means[None, :, :]) / sds
    normalizers = np.log(sds * math.sqrt(2 * math.pi)).sum(axis=1)
    return -normalizers - 0.5 * np.sum(difference**2, axis=2)


def forward(emissions, transitions):
    num_frames, num_states = emissions.shape
    if num_frames < num_states:
        raise ValueError("Recording has fewer frames than states")
    logs = log_transitions(transitions)
    stay = np.diag(logs)
    advance = logs[np.arange(num_states - 1), np.arange(1, num_states)]
    alpha = np.full(emissions.shape, -np.inf)
    alpha[0, 0] = emissions[0, 0]
    for t in range(1, num_frames):
        same_state = alpha[t - 1] + stay
        previous_state = np.full(num_states, -np.inf)
        previous_state[1:] = alpha[t - 1, :-1] + advance
        alpha[t] = emissions[t] + np.logaddexp(same_state, previous_state)
    return alpha


def backwards(emissions, transitions):
    num_frames, num_states = emissions.shape
    logs = log_transitions(transitions)
    stay = np.diag(logs)
    advance = logs[np.arange(num_states - 1), np.arange(1, num_states)]
    beta = np.full(emissions.shape, -np.inf)
    beta[-1, -1] = logs[-1, -1]
    for t in range(num_frames - 2, -1, -1):
        following = emissions[t + 1] + beta[t + 1]
        same_state = stay + following
        next_state = np.full(num_states, -np.inf)
        next_state[:-1] = advance + following[1:]
        beta[t] = np.logaddexp(same_state, next_state)
    return beta


def sequence_score(emissions, transitions):
    alpha = forward(emissions, transitions)
    return float(alpha[-1, -1] + log_transitions(transitions)[-1, -1])


def update_parameters(matrices, emission_matrices, transitions):
    num_states = len(transitions)
    len_mfcc = matrices[0].shape[1]
    weights = np.zeros(num_states)
    sums = np.zeros((num_states, len_mfcc))
    squares = np.zeros_like(sums)
    transition_counts = np.zeros((num_states, num_states + 1))
    logs = log_transitions(transitions)
    states = np.arange(num_states)

    for matrix, emissions in zip(matrices, emission_matrices):
        alpha = forward(emissions, transitions)
        beta = backwards(emissions, transitions)
        score = alpha[-1, -1] + logs[-1, -1]
        if not np.isfinite(score):
            raise ValueError("Recording has no possible state path")

        gamma = np.exp(alpha + beta - score)
        weights += gamma.sum(axis=0)
        sums += gamma.T @ matrix
        squares += gamma.T @ (matrix**2)

        log_stay = alpha[:-1] + np.diag(logs) + emissions[1:] + beta[1:] - score
        transition_counts[states, states] += np.exp(log_stay).sum(axis=0)
        log_advance = (alpha[:-1, :-1] + logs[states[:-1], states[1:]]
                       + emissions[1:, 1:] + beta[1:, 1:] - score)
        transition_counts[states[:-1], states[1:]] += np.exp(log_advance).sum(axis=0)
        transition_counts[-1, -1] += 1

    if np.any(weights <= 0):
        raise ValueError("A state has no assigned frames")
    means = sums / weights[:, None]
    variance = squares / weights[:, None] - means**2
    sds = np.sqrt(np.maximum(variance, 1e-10))
    new_transitions = transition_counts / transition_counts.sum(axis=1)[:, None]
    return means, sds, new_transitions


def viterbi(emissions, transitions):
    num_frames, num_states = emissions.shape
    if num_frames < num_states:
        raise ValueError("Recording has fewer frames than states")
    logs = log_transitions(transitions)
    stay = np.diag(logs)
    advance = logs[np.arange(num_states - 1), np.arange(1, num_states)]
    scores = np.full(emissions.shape, -np.inf)
    moved = np.zeros(emissions.shape, dtype=bool)
    scores[0, 0] = emissions[0, 0]

    for t in range(1, num_frames):
        same_state = scores[t - 1] + stay
        previous_state = np.full(num_states, -np.inf)
        previous_state[1:] = scores[t - 1, :-1] + advance
        moved[t] = previous_state > same_state
        scores[t] = emissions[t] + np.maximum(same_state, previous_state)

    score = scores[-1, -1] + logs[-1, -1]
    if not np.isfinite(score):
        raise ValueError("Recording has no possible state path")
    state = num_states - 1
    path = [state]
    for t in range(num_frames - 1, 0, -1):
        if moved[t, state]:
            state -= 1
        path.append(state)
    return list(reversed(path)), float(score)


def train_model(matrices, parameters, num_iterations=10):
    if num_iterations < 1:
        raise ValueError("Need at least one training iteration")
    parameters = dict(parameters)
    means = np.array(parameters["means"])
    sds = np.array(parameters["sds"])
    transitions = np.array(parameters["transition_matrix"])
    emissions = [emission_matrix(matrix, means, sds) for matrix in matrices]
    likelihoods = [sum(sequence_score(matrix, transitions) for matrix in emissions)]

    for iteration in range(num_iterations):
        means, sds, transitions = update_parameters(matrices, emissions, transitions)
        emissions = [emission_matrix(matrix, means, sds) for matrix in matrices]
        likelihoods.append(sum(sequence_score(matrix, transitions) for matrix in emissions))

    parameters.update(means=means.tolist(), sds=sds.tolist(),
                      transition_matrix=transitions.tolist(),
                      num_iterations=num_iterations, training_likelihoods=likelihoods)
    return parameters


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    parameters = json.loads((folder / "hmm_parameters_init.json").read_text())
    files, matrices = load_recordings(
        sample_folder / parameters["word"], parameters["feature_settings"]
    )
    means = np.array(parameters["means"])
    sds = np.array(parameters["sds"])
    transitions = np.array(parameters["transition_matrix"])
    num_iterations = 10
    emissions = [emission_matrix(matrix, means, sds) for matrix in matrices]

    for iteration in range(num_iterations):
        means, sds, transitions = update_parameters(matrices, emissions, transitions)
        parameters.update(means=means.tolist(), sds=sds.tolist(),
                          transition_matrix=transitions.tolist())
        output = folder / f"hmm_parameters_iteration_{iteration + 1}.json"
        output.write_text(json.dumps(parameters, indent=4, allow_nan=False))

        emissions = [emission_matrix(matrix, means, sds) for matrix in matrices]
        likelihood = sum(sequence_score(matrix, transitions) for matrix in emissions)
        path, score = viterbi(emissions[0], transitions)
        print("Iteration", iteration + 1, "- total forward log likelihood:", likelihood)
        print("First recording:", files[0].name)
        print("Viterbi score:", score)
        print("Viterbi path:", [state + 1 for state in path])
