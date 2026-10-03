import json
from pathlib import Path

import numpy as np

from main_fft import load_recordings, sample_folder, settings


def initialize_parameters(matrices, num_states):
    if num_states < 1 or not matrices:
        raise ValueError("Need at least one state and one recording")

    sections = [[] for state in range(num_states)]

    for matrix in matrices:
        if len(matrix) < num_states:
            raise ValueError("A trimmed recording has fewer frames than states")
        for state in range(num_states):
            start = state * len(matrix) // num_states
            end = (state + 1) * len(matrix) // num_states
            sections[state].append(matrix[start:end])

    means, sds, transitions, counts = [], [], [], []
    for state in range(num_states):
        observations = np.concatenate(sections[state])
        means.append(observations.mean(axis=0).tolist())
        variance = observations.var(axis=0)
        sds.append(np.sqrt(np.maximum(variance, 1e-10)).tolist())
        counts.append(len(observations))

        row = [0] * (num_states + 1)
        advance = len(matrices) / counts[state]
        row[state] = 1 - advance
        row[state + 1] = advance
        transitions.append(row)

    # The final column is END, which does not emit an observation.
    return {
        "num_states": num_states,
        "means": means,
        "sds": sds,
        "transition_matrix": transitions,
        "state_samples": counts,
        "num_recordings": len(matrices),
    }


if __name__ == "__main__":
    word = "right"
    num_states = 5
    files, matrices = load_recordings(sample_folder / word, settings)
    parameters = initialize_parameters(matrices, num_states)
    parameters["word"] = word
    parameters["feature_settings"] = settings

    output = Path(__file__).resolve().parent / "hmm_parameters_init.json"
    output.write_text(json.dumps(parameters, indent=4, allow_nan=False))
    print("Initialized", num_states, "states from", len(files), "recordings")
    print("State frame counts:", parameters["state_samples"])
    print("Saved:", output)
