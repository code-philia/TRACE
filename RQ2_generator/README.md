# README

This directory contains the code for training and evaluating the RQ2 generator model.

## 📂 Content

* `bleu.py`: Script for calculating BLEU scores.
* `code_window.py`: Script to create edit hunks for the generator model.
* `eval.sh`: Script to evaluate the generator model.
* `generator_metics.py`: Script for calculating generator-specific metrics.
* `generator.py`: Script for training and evaluating the generator model.
* `run.py`: Main script for training and evaluating the generator model.
* `run.sh`: Script to train the generator model.
* `utils.py`: Utility functions for the generator model.
* `extract_tool_feedback/`: Directory containing scripts to extract potential edit composition types from edit hunk data samples.
    * `extract_tool_feedback/arg_val.py`: Script to identify the number of arguments and values in the edit hunk.
    * `extract_tool_feedback/clone_detect.py`: Script to identify clones between edit hunks.
    * `extract_tool_feedback/pre_process.py`: Script to use AST parsing to extract potential edit composition types.
* `model_6/`: Directory containing the 6-label generator model files, should at least contain:
    * `model_6/all/checkpoint-last/pytorch_model.bin`: Pre-trained weights for the 6-label generator model.

## 🚀 Getting Started

> [!Note]
>
> To skip the training process, you may execute command `bash download_model.sh` under working directory `RQ2_generator/`.

* To extract the potential edit composition type of each edit hunk data sample, run the following command under working directory `RQ2_generator/extract_tool_feedback/`:

    ```bash
    python pre_process.py
    ```

* Train the locator model by running the following command under working directory `RQ2_generator/`:

    ```bash
    bash run.sh
    ```

* Evaluate the locator model by running the following command under working directory `RQ2_generator/`:

    ```bash
    bash eval.sh
    ```