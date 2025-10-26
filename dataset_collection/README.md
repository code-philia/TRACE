# README

This directory crawls top-starred repositories' commit from GitHub


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

* When running scripts, your current working directory should be `dataset_collection`;

* Please fill hyper-parameters in `.env`;

    > Acquire GitHub token please refer to [Creating a personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic)

* 