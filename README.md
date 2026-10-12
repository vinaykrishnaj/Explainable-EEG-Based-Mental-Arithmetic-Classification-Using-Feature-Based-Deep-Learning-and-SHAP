# Explainable EEG-Based Mental Arithmetic Classification Using Feature-Based Deep Learning and SHAP

An EEG-based machine-learning research project that distinguishes
**resting-state EEG** from **mental-arithmetic EEG** using
frequency-domain features, a lightweight Multi-Layer Perceptron (MLP),
probability-threshold calibration, and SHAP explanations.

> **Research focus:** combine useful classification performance with
> interpretable feature-level evidence, rather than treating the model
> as a black box.

## Overview

Mental arithmetic requires attention and working memory and can produce
changes in EEG frequency activity. This project processes EEG recordings
into fixed-length windows, extracts interpretable signal features, and
trains a feature-based neural network to predict whether a window
represents resting state or mental arithmetic.

The repository includes the experimental notebook and the accompanying
research paper:

-   `SHAP_EEG_Org.ipynb` --- EEG preprocessing, model experiments,
    feature extraction, evaluation, and SHAP analysis.
-   `CRC_380_ICSDAS'26_V1.docx` --- research-paper manuscript describing
    the proposed framework and its experimental findings.

## Key Features

-   **EEG preprocessing:** load EDF recordings, select eight channels,
    and apply a 0.5--40 Hz band-pass filter.
-   **Windowing:** split continuous recordings into non-overlapping
    5-second windows.
-   **Frequency-domain features:** calculate theta, alpha, and beta band
    power using Welch's power spectral density estimate, alongside
    signal energy.
-   **Feature-based deep learning:** train an MLP on 32 features (four
    features per channel across eight channels).
-   **Decision-threshold calibration:** use a 0.35 probability threshold
    for the mental-arithmetic class in the notebook's final evaluation
    cell.
-   **Explainable AI:** use SHAP `GradientExplainer` for global feature
    importance and local/sample-level explanations.
-   **Evaluation:** inspect accuracy, precision, recall, F1-score,
    balanced accuracy, ROC-AUC, PR-AUC, Matthews correlation coefficient
    (MCC), Cohen's kappa, and a confusion matrix.
-   **Model experiments:** the notebook also includes raw-window 1D-CNN
    and CNN+BiLSTM experiments for comparison.

## Workflow

``` text
EDF EEG recordings
        |
        v
Load recordings with MNE
        |
        v
Select 8 EEG channels
        |
        v
Band-pass filter (0.5–40 Hz)
        |
        v
Create non-overlapping 5-second windows
        |
        +-------------------------------+
        |                               |
        v                               v
Raw-window experiments            Feature extraction
1D-CNN / CNN+BiLSTM                Welch PSD band power
                                  Theta / Alpha / Beta
                                  + signal energy
                                          |
                                          v
                                  Standardize features
                                          |
                                          v
                                      MLP model
                                          |
                                          v
                              Probability threshold = 0.35
                                          |
                                          v
                            Resting / Mental Arithmetic
                                          |
                                          v
                                  SHAP explanations
```

## Dataset

The notebook uses the publicly available **EEG During Mental Arithmetic
Tasks** dataset in EDF format. Recordings are labeled using their
filename suffix:

-   Files ending in `_1.edf` → **Resting state** (`0`)
-   Files ending in `_2.edf` → **Mental arithmetic** (`1`)

The research paper describes recordings from 36 subjects, with
resting-state and mental-arithmetic recordings for each subject. The
notebook expects the dataset to be downloaded and extracted locally or
mounted in Google Drive.

**Dataset access:** [PhysioNet --- EEG During Mental Arithmetic
Tasks](https://physionet.org/content/eegmat/1.0.0/)

Please review the dataset's terms and citation requirements on the
source website before redistribution. The dataset itself is not included
in this repository.

## Methodology

### 1. Preprocessing and segmentation

The notebook reads EDF files with MNE, selects the first eight available
channels, filters the signals between 0.5 and 40 Hz, and segments each
recording into non-overlapping five-second windows.

### 2. Feature extraction

For every channel in each window, the notebook estimates power spectral
density with Welch's method and computes:

  -----------------------------------------------------------------------
  Feature                 Frequency range /       Intended signal
                          definition              information
  ----------------------- ----------------------- -----------------------
  Theta power             4--8 Hz                 Activity associated
                                                  with attention and
                                                  working memory

  Alpha power             8--13 Hz                Activity often
                                                  associated with
                                                  relaxed/resting states

  Beta power              13--30 Hz               Activity associated
                                                  with active cognitive
                                                  processing

  Signal energy           Sum of squared samples  Overall signal
                          in the window           magnitude/energy
  -----------------------------------------------------------------------

Eight channels × four features per channel produce **32 input features
per window**. The features are standardized using a `StandardScaler`
fitted on the training split.

### 3. MLP classifier

The feature-based model is a PyTorch MLP with fully connected layers:

``` text
32 input features
       |
   Linear (128)
       |
      ReLU
       |
    Linear (32)
       |
      ReLU
       |
    Linear (2)
       |
Rest / Mental Arithmetic
```

The notebook trains the MLP for 50 epochs using the Adam optimizer and
cross-entropy loss. It uses a stratified 80/20 random window-level
train/test split (`random_state=42`).

### 4. Threshold calibration

Instead of relying only on the default 0.5 decision threshold, the final
evaluation uses **0.35** for the mental-arithmetic probability. This is
intended to increase sensitivity to the mental-arithmetic class.
Threshold selection should be performed on validation data---not on the
final test set---in a rigorous evaluation.

### 5. SHAP explainability

SHAP is applied to the MLP's predicted probabilities using
`GradientExplainer`. The notebook includes:

-   Local feature-attribution bar plot for an individual window.
-   SHAP summary plot for a selected subject's windows.
-   SHAP heatmap for subject-level windows.
-   Mean absolute SHAP feature importance across a sample of test
    windows.

Feature names follow the `Ch<channel-index>_<feature>` pattern, for
example `Ch0_Theta` and `Ch4_Alpha`. Channel indices refer to the
selected channel order in the notebook.

## Results

The following values are taken from the **executed notebook outputs**.
Results depend on the selected evaluation procedure and threshold.

### Feature-based MLP --- final notebook evaluation

  Metric                                       Value
  ------------------------------------- ------------
  Test accuracy                           **91.20%**
  Balanced accuracy                           88.34%
  Precision (mental-arithmetic class)         82.56%
  Recall (mental-arithmetic class)            82.56%
  F1-score (mental-arithmetic class)          82.56%
  ROC-AUC                                     0.9301
  PR-AUC / Average precision                  0.8491
  Matthews correlation coefficient            0.7668
  Cohen's kappa                               0.7668

Confusion matrix at threshold 0.35, with rows representing true labels
and columns predicted labels:

                                 Predicted Rest   Predicted Mental Arithmetic
  ---------------------------- ---------------- -----------------------------
  **True Rest**                             240                            15
  **True Mental Arithmetic**                 15                            71

### Additional results reported in the paper

The accompanying manuscript reports **98.09% training accuracy**,
**90.32% test accuracy**, and **0.927 ROC-AUC** for the proposed
feature-based framework. The notebook's final threshold-calibrated
evaluation reports 91.20% accuracy and ROC-AUC 0.9301. These are
different reported evaluation outputs and should not be presented as if
they came from one identical run.

The notebook's raw-window 1D-CNN and CNN+BiLSTM experiments each
produced approximately 74.78% accuracy in the recorded run, but
predicted only the resting-state class. Their results indicate that
those particular runs did not successfully learn the minority
mental-arithmetic class; they should not be interpreted as strong
baselines.

### SHAP interpretation

The manuscript's discussion describes theta and beta activity as
contributing positively to mental-arithmetic predictions, while alpha
activity is more associated with resting-state predictions. The notebook
also computes channel-specific SHAP importance; its top reported
features include `Ch4_Alpha`, `Ch6_Energy`, and `Ch0_Theta`. Feature
contributions are model- and sample-dependent, and SHAP attribution
indicates contribution to the model output---not causality or clinical
proof.

## Tech Stack

-   **Python**
-   **PyTorch** --- MLP and raw-window neural-network experiments
-   **MNE-Python** --- EDF loading and EEG filtering
-   **NumPy / SciPy** --- numerical processing and Welch PSD
-   **scikit-learn** --- data splitting, scaling, and evaluation metrics
-   **SHAP** --- model explainability
-   **Matplotlib** --- plots and evaluation visualizations
-   **Google Colab** --- notebook execution environment used in the
    project

## Getting Started

### 1. Clone the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd EEG-SHAP
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with the URL of your GitHub
repository.

### 2. Install dependencies

The notebook installs some packages directly. To set up a local
environment, use:

``` bash
python -m venv .venv
```

Activate it:

**Windows PowerShell**

``` powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

``` bash
source .venv/bin/activate
```

Install the core libraries:

``` bash
pip install mne torch numpy scipy scikit-learn shap matplotlib
```

A compatible PyTorch build depends on your operating system and whether
you want CPU or CUDA execution. Follow the [official PyTorch
installation instructions](https://pytorch.org/get-started/locally/) if
needed.

### 3. Download and configure the dataset

1.  Download the dataset from
    [PhysioNet](https://physionet.org/content/eegmat/1.0.0/).
2.  Extract the EDF files.
3.  Update `BASE_PATH` in the notebook to point to the extracted dataset
    directory.

The original notebook uses a Google Drive path:

``` python
BASE_PATH = "/content/drive/MyDrive/eeg-during-mental-arithmetic-tasks-1.0.0"
```

For local execution, replace it with your own dataset path. If running
in Google Colab, mount Google Drive and keep the path consistent with
your folder structure.

### 4. Run the notebook

Open `SHAP_EEG_Org.ipynb` in Google Colab or Jupyter Notebook and run
the cells in order. The notebook performs preprocessing, feature
extraction, training, evaluation, and SHAP analysis. Some cells are
interactive and prompt for a test sample index.

## Repository Structure

``` text
EEG-SHAP/
├── SHAP_EEG_Org.ipynb
├── index.html                  # if included: exported notebook view
├── README.md
└── CRC_380_ICSDAS'26_V1.docx  # if you choose to publish the manuscript
```

Only list files that you actually commit to the repository. If you
publish the paper, check the conference/publisher's rules and ensure
that sharing the manuscript is permitted.

## Limitations and Future Work

-   **Subject-wise generalization:** the current notebook uses a random
    window-level split. Windows from the same subject/recording may
    appear in both training and test sets, so the reported scores do not
    establish subject-independent generalization.
-   **Class imbalance:** the extracted dataset contains more resting
    windows than mental-arithmetic windows (1,273 vs. 432 in the
    recorded run).
-   **Threshold validation:** select the decision threshold on a
    separate validation set and evaluate once on an untouched test set.
-   **Reproducibility:** set and record random seeds, package versions,
    data ordering, and model checkpoints to improve reproducibility.
-   **Stronger baselines:** tune the CNN/CNN+BiLSTM models and compare
    with classical models such as logistic regression, SVM, and Random
    Forest using the same split protocol.
-   **Generalization:** evaluate with subject-wise splits and, where
    possible, independent EEG datasets.
-   **Interpretability:** assess SHAP stability across subjects and
    repeated runs; feature attribution should not be interpreted as
    causal evidence.
-   **Practical deployment:** investigate channel selection, real-time
    inference, and validation on more diverse recordings before
    considering real-world use.

## Research Paper

**Title:** *Explainable EEG-Based Mental Arithmetic Classification Using
Feature-Based Deep Learning and SHAP*

The manuscript describes a feature-based MLP approach, threshold
calibration, and SHAP-based local/global explanations for distinguishing
resting-state and mental-arithmetic EEG. Refer to the paper included in
the repository, if publication permissions allow.

## Citation

If you use or build on this work, please cite the paper using the final
bibliographic information once it is available. A placeholder citation
is provided below; replace it with the official conference proceedings
details after publication.

``` text
Jagarlamudi, V. K., Basha, S. S., Prathap, B. R., and B., P.
"Explainable EEG-Based Mental Arithmetic Classification Using Feature-Based
Deep Learning and SHAP." ICSDAS 2026 manuscript.
```

## Disclaimer

This repository is for research and educational purposes. It is not a
medical device and must not be used for clinical diagnosis or
high-stakes decisions.
