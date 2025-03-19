from importlib import import_module

if __name__ == '__main__':
    lang = 'java' 
    num_of_repo = 100 

    # Step 1: get repos, commits and clone to local
    step_1 = import_module("1_crawl")
    step_1.crawl(lang, num_of_repo)
    
    # Step 2: filter commit based on commit information
    step_2 = import_module("2_clean_commit_info")
    step_2.clean_commit(lang)
    
    # Step 3: filter commit based on edit information
    step_3 = import_module("3_clean_edit_info")
    step_3.clean_edit(lang)
    
    # Step 4: make a dataset
    step_4 = import_module("4_make_dataset")
    step_4.make_dataset(lang, "dataset_fine_grain")
    
    # Step 5: filter a commit if contains any code window / sliding window that is longer than input length
    step_5 = import_module("5_filter_by_length")
    step_5.filter_by_length(lang, "dataset_fine_grain", tokenizer_name="salesforce/codet5-base")
    
    # Step 6: count dataset
    step_6 = import_module("6_count")
    step_6.count(lang, "dataset_fine_grain")