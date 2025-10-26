# README

This directory contains the code for training and evaluating the RQ1 locator model.

## 📂 Content

* `bleu.py`: Script to compute BLEU scores.
* `code_window.py`: Script to create code windows for the locator model.
* `convert_label.py`: Script to convert 6-label format to 3-label format.
* `locator.py`: Implementation of the locator model.
* `run.sh`: Script to train the locator model.
* `eval.sh`: Script to evaluate the locator model.
* `run.py`: Main script for training the locator model.
* `utils.py`: Utility functions for data processing and model evaluation.
* `model_6`: Directory containing the 6-label locator model files, should at least contain:
    * `model_6/all/checkpoint-last/pytorch_model.bin`: Pre-trained weights for the 6-label locator model.

## 🚀 Getting Started

* Train the locator model by running:

    ```bash
    bash run.sh
    ```

* Evaluate the locator model by running:

    ```bash
    bash eval.sh
    ```