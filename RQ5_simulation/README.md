# README

This directory contains the code for RQ5 real-world simulation. 

* If you aim to fully reproduce the experiment results, please follow the instructions in this README.

* If you intend to reuse the simulation framework and use TRACE as a baseline, please refer to the [Simulation framework repository](We are working on it!).


## 🚀 Getting Started

> [!Note]
>
> To skip the training process, you may execute command `bash download_models.sh` under working directory `RQ5_simulation/`.

* Install LSPs:

    Python LSP (pyright) has been included in `requirements.txt`, Java LSP (jdtls) has been included in `RQ5_simulation/LSPs/jdt-language-server`.
    ```bash
    go install golang.org/x/tools/gopls@latest
    npm install -g typescript-language-server typescript
    ```
    
* To evaluate indivisual benchmark, run the following command under working directory `RQ5_simulation/`:

    ```bash
    bash 0_TRACE.sh
    bash 1_woinvoker.sh
    bash 2_enriched.sh
    bash 3_plain.sh
    bash 4_coedpilot.sh
    bash 5_ccd.sh
    ```

* Or if you want to run all benchmarks sequentially, run the following command under working directory `RQ5_simulation/`:

    ```bash
    bash all.sh
    ```