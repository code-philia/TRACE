# README

This directory crawls top-starred repositories' commit from GitHub. You may use this code to create your own dataset.

To acquire the dataset used in our experiments, please refer to the instruction in the main [README](../README.md) file, directly download from HuggingFace.


## 📂 Content

* `0_all_in_one.py`: This script performs all data processing in one go and directly produces `train.json`, `dev.json`, and `test.json` files under the path `ROOT_PATH/dataset_fine_grain/all`;

* `1_crawl.py`: This script crawl the information of top-starred repositories;

* `2_clean_commit_info.py`: This script filters out commit with undesired commit messages;

* `3_clean_edit_info.py`: this script filters out commit with undesired edits;

* `4_make_dataset.py`: This script transforms commit into data samples;

* `5_filter_by_length.py`: This script filters out commit that contains data sample of exceeding token length

* `6_count.py`: This script counts the number of data samples in each dataset split;

* `7_combine.py`: This script combines dataset from different languages into one.

## 🚀 Getting Started

* Run the following command under working directory `dataset_collection/`:

```bash
python 0_all_in_one.py
```