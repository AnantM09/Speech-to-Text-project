import argparse
import json
from pathlib import Path

from scipy.io import wavfile

from main_fft import load_recordings, sample_folder, settings
from right_init import initialize_parameters
from hmm_train import train_model


words = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]
model_folder = Path(__file__).resolve().parent / "models"


def train_dictionary(num_states=7, num_recordings=30, num_iterations=10):
    model_folder.mkdir(exist_ok=True)
    for word in words:
        files, matrices = load_recordings(sample_folder / word, settings, num_recordings)
        sample_rates = {wavfile.read(file)[0] for file in files}
        if len(sample_rates) != 1:
            raise ValueError(f"Recordings for {word} have different sample rates")

        parameters = initialize_parameters(matrices, num_states)
        parameters.update(word=word, feature_settings=settings,
                          sample_rate=sample_rates.pop(), training_files=[file.name for file in files])
        parameters = train_model(matrices, parameters, num_iterations)
        output = model_folder / f"{word}.json"
        output.write_text(json.dumps(parameters, indent=4, allow_nan=False))
        print(word, "- trained", num_states, "states on", len(files), "recordings", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train one HMM for each dictionary word")
    parser.add_argument("--states", type=int, default=7)
    parser.add_argument("--recordings", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    try:
        train_dictionary(args.states, args.recordings, args.iterations)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
