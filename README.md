# BIG-MRI

Official repository for **BIG-MRI: Breast Imaging Group MRI Challenge**.

BIG-MRI focuses on **breast cancer risk prediction from breast MRI**. The goal is to estimate, for each woman, the absolute risk of incident breast cancer within **2 years** and **5 years**. The challenge benchmarks MRI-only and multimodal AI methods in screening/high-risk MRI cohorts under blinded evaluation.

This repository is the official challenge codebase. It will be updated with baseline methods, preprocessing utilities, Grand Challenge submission templates, submission format specifications, and evaluation/ranking scripts as they become available.

## Overview

Breast MRI is increasingly used in women at elevated risk of breast cancer, including high-risk screening populations. However, validated MRI-based tools for individualized risk stratification remain limited.

BIG-MRI asks participants to develop models that predict future incident breast cancer risk from dynamic contrast-enhanced breast MRI, with or without clinical covariates depending on the selected track. The challenge is designed to support clinically meaningful benchmarking for personalized screening and risk-adapted follow-up.

## Challenge Tracks

| Track | Name | Input | Output |
| --- | --- | --- | --- |
| Track 1 | Image-based / MRI-only risk prediction | DCE breast MRI sequences only | Absolute 2-year and 5-year incident breast cancer risk |
| Track 2 | Multimodal risk prediction | DCE breast MRI sequences and provided clinical covariates | Absolute 2-year and 5-year incident breast cancer risk |

DCE-MRI data are planned to include pre-contrast and post-contrast sequences. Clinical covariates will be provided only for the multimodal track.

Each track will have a separate leaderboard.

## Repository Contents / Roadmap

This repository is under active preparation. Planned contents include:

| Component | Status | Description |
| --- | --- | --- |
| `README.md` | Available | Official repository overview |
| Baseline training pipeline | Coming soon | Reference model training code for participants |
| Preprocessing utilities | Coming soon | Utilities for preparing DCE breast MRI data |
| Grand Challenge submission template | Coming soon | Docker template and local testing instructions |
| Submission format specification | Coming soon | Required model input/output contract and prediction format |
| Evaluation and ranking scripts/details | Coming soon | Metrics, aggregation, and leaderboard computation details |

The repository structure may change as challenge materials are finalized.

## Data

Training data will be released through Hugging Face. The planned release will include de-identified breast MRI data from a screening/high-risk MRI cohort.

Planned data format and contents:

- DCE breast MRI in NIfTI format
- Pre-contrast and post-contrast MRI sequences
- De-identified clinical covariate table for Track 2
- Dataset documentation and data dictionary

Data access instructions, usage terms, and the Hugging Face link are **Coming soon**.

The held-out test set will be used only for blinded evaluation through Grand Challenge. Test labels will not be released to participants.

## Submission

Participants will train their models locally. Final algorithms will be submitted to Grand Challenge for blinded evaluation.

Important submission principles:

- Model training is performed locally by participants.
- Grand Challenge is used for final algorithm submission and evaluation.
- No model training will take place on the Grand Challenge platform.
- The exact Docker I/O contract is Coming soon.
- The exact prediction file format and validation rules are Coming soon.
- Local testing templates will be provided in this repository.

## Evaluation

Submitted algorithms will predict two absolute risk estimates per case:

- 2-year incident breast cancer risk
- 5-year incident breast cancer risk

Planned evaluation metrics include:

- Time-dependent AUC at the 2-year horizon
- Time-dependent AUC at the 5-year horizon
- Harrell’s C-index over full follow-up

Calibration metrics may also be reported to support clinical interpretation, but they are not necessarily ranking metrics. The final ranking procedure, statistical reporting details, and evaluation scripts will be released with the official evaluation specification.

## Important Links

- Challenge website: https://big-radboudumc.github.io/
- GitHub repository: https://github.com/BIG-Radboudumc/BIG-MRI
- Training data on Hugging Face: Coming soon
- Grand Challenge submission portal: Coming soon
- Submission instructions: Coming soon

## Contact

For challenge-level questions, please use the contact information provided on the official challenge website.

For repository-specific questions, issues, or code-related requests, please open an issue in this GitHub repository.

## Citation

Citation information will be provided once the official challenge overview paper or citation record is available.

## Acknowledgements

BIG-MRI is organized by the Breast Imaging Group and collaborators from Radboudumc and the Netherlands Cancer Institute.
