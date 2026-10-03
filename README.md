# Ten-word speech recognizer

A small isolated-word recognizer built using MFCC features and Gaussian hidden Markov models. The dictionary contains **down, go, left, no, off, on, right, stop, up, and yes**.

The included models each have seven emitting states and were trained on 30 recordings for ten Baum–Welch iterations. The CLI prints a forward log likelihood for every word, sorted from highest to lowest.

## Setup

Use Python 3.11 or newer. From this folder, install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

The dependency versions are pinned to the versions used to train and check the included models.

## Score a recording

From this folder:

```powershell
python recognize.py "path/to/recording.wav"
```

From the original project's parent folder:

```powershell
python readable_version/recognize.py "speech_samples/right/00b01445_nohash_0.wav"
```

Input must be a mono, 16-bit PCM WAV recording at 16,000 Hz for the included models. Empty, entirely silent, or too-short recordings are rejected. Use a recording containing one spoken word.

The highest log likelihood is the best match among the ten models. Scores are not confidence percentages, and absolute scores from recordings of different lengths are not directly comparable. The recognizer has no unknown-word rejection or continuous-speech segmentation.

Recognition needs the Python files and all ten JSON files in `models/`. It does not need the training dataset.

## How it works

1. Measure frame RMS on the original audio and trim the leading and trailing regions below 35 dB relative to the maximum frame RMS. Preserve internal quiet regions and keep a 40 ms boundary margin.
2. Split the retained audio into overlapping 350-sample frames with a 150-sample step. Zero-pad partial frames.
3. Compute batched FFTs, apply 28 triangular Mel filters, and calculate 13 MFCC coefficients using an unnormalized DCT-II.
4. Evaluate the same MFCC matrix with each word's HMM using the forward algorithm, including its END transition.

Initialization divides each recording into equal consecutive sections. Each state uses a diagonal Gaussian distribution. Training refines its means, standard deviations, and stay/advance probabilities with Baum–Welch. States can stay or advance one step; the final state advances to END. No stronger minimum-duration constraint is imposed. Delta features are not included.

## Retrain the dictionary

The training scripts expect `speech_samples` beside this folder:

```text
parent/
├── readable_version/
│   ├── train_dictionary.py
│   ├── recognize.py
│   └── models/
└── speech_samples/
    ├── down/
    ├── go/
    ├── left/
    ├── no/
    ├── off/
    ├── on/
    ├── right/
    ├── stop/
    ├── up/
    └── yes/
```

Each word folder must contain at least 30 suitable WAV recordings. Use the same mono, 16-bit PCM, 16,000 Hz format as the included models. Training selects the first 30 alphabetically sorted filenames per folder; it does not perform a speaker-based split. The `right` folder is used, not `right1`.

From this folder, run:

```powershell
python train_dictionary.py
```

Settings can be changed through CLI arguments:

```powershell
python train_dictionary.py --states 7 --recordings 30 --iterations 10
```

Retraining overwrites the ten files in `models/`. Each model records its feature settings, training filenames, and total training log likelihood after every iteration. Feature settings are defined in `main_fft.py`.

## Files

| File | Purpose |
|---|---|
| `main_fft.py` | Silence trimming and MFCC extraction |
| `right_init.py` | General state initialization; also runnable as a five-state “right” demonstration |
| `hmm_train.py` | Forward/backward, Baum–Welch, Viterbi, and a single-model training demonstration |
| `train_dictionary.py` | Train and save the ten word models |
| `recognize.py` | Score a WAV file against the dictionary |
| `models/` | Trained seven-state dictionary models |

The single-model demonstrations write `hmm_parameters_init.json` and `hmm_parameters_iteration_*.json` beside the scripts. Those files are separate from the dictionary models and are not required by the recognition CLI. `__pycache__/` is also unnecessary to include in the repository.

The included models and CLI have been checked for valid parameters and scoring behavior. Held-out recognition accuracy has not been established for this specific 30-recording, seven-state configuration.
