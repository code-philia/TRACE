# README

This directory contains the code for creating dataset for Edit-composition Invoker, training and evaluating the RQ4 invoker model.

## 📂 Content

TBD 

## 🚀 Getting Started

> [!Note]
>
> To skip the dataset curation process and training process, you may execute command `bash download_model.sh` under working directory `RQ4_invoker/`.

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
