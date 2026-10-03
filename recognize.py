import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from main_fft import extract_mfcc
from hmm_train import emission_matrix, sequence_score
from train_dictionary import words, model_folder


def word_likelihoods(file):
    models = []
    for word in words:
        path = model_folder / f"{word}.json"
        if not path.is_file():
            raise ValueError("Train the dictionary first with train_dictionary.py")
        models.append(json.loads(path.read_text()))

    feature_settings = models[0]["feature_settings"]
    expected_rate = models[0]["sample_rate"]
    if any(model["feature_settings"] != feature_settings or
           model["sample_rate"] != expected_rate for model in models):
        raise ValueError("Dictionary models must use the same feature settings and sample rate")

    sample_rate, audio = wavfile.read(file)
    if sample_rate != expected_rate:
        raise ValueError(f"Expected a {expected_rate} Hz recording, got {sample_rate} Hz")
    if audio.dtype != np.dtype("int16"):
        raise ValueError("Expected 16-bit PCM WAV audio, matching the training recordings")
    matrix = extract_mfcc(audio, sample_rate, **feature_settings)
    likelihoods = {}
    for model in models:
        emissions = emission_matrix(matrix, model["means"], model["sds"])
        likelihoods[model["word"]] = sequence_score(emissions, model["transition_matrix"])
    return likelihoods


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the forward log likelihood for each word")
    parser.add_argument("audio", type=Path, help="Path to a mono 16-bit PCM WAV recording")
    args = parser.parse_args()
    try:
        likelihoods = word_likelihoods(args.audio)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
    for word, score in sorted(likelihoods.items(), key=lambda item: item[1], reverse=True):
        print(f"{word:<6} {score:.6f}")
