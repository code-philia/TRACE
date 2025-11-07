# README

This directory contains the code for creating dataset for Edit-composition Invoker, training and evaluating the RQ4 invoker model.

## 📂 Content

* `ask_lsp.py`: Script to query LSP to verify whether the edit belongs to an edit composition;
* `construct_sample.py`: Script to construct dataset for invoker model;
* `invoker_utils.py`: Utility functions for invoker model;
* `make_raw_dataset.py`: Script to curate raw dataset that contains meta information for constructing invoker dataset;
* `run.py`: Python script to train/eval the invoker model;
* `run.sh`: Script to train the invoker model;
* `eval.sh`: Script to evaluate the invoker model;

## 🚀 Getting Started

> [!Note]
>
> To skip the dataset curation process and training process for dataset and trained model, you may execute command `bash download_model.sh` under working directory `RQ4_invoker/`.

* To curate the dataset for invoker model, run the following command under working directory `RQ4_invoker/`:

    ```bash
    python make_raw_dataset.py
    python construct_sample.py
    ```

* To train the invoker model, run the following command under working directory `RQ4_invoker/`:

    ```bash
    bash run.sh
    ```

* To evaluate the invoker model, run the following command under working directory `RQ4_invoker/`:

    ```bash
    bash eval.sh
    ```
