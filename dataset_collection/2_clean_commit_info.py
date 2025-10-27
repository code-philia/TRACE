# This script is used to filter the commit based on the commit information, check function commit_filter
# The url of commits that pass the cleaning are stored in {ROOT_PATH}/commit_info/{lang}_filtered_commit_urls.json
import re
import os
import json

from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv("../.env")
ROOT_PATH = os.getenv("ROOT_PATH")
    
def remove_pull_id(commit_message):
    pull_id_pattern = re.compile(r'\(#(\d+)\)')

    updated_message = re.sub(pull_id_pattern, '', commit_message).strip()

    return updated_message

def commit_filter(commit_dict):
    def count_file_names(input_string):
        file_name_pattern = r'([^/.]+\.\w+)'
        matches = re.findall(file_name_pattern, input_string)
        return len(matches)
    
    def count_external_references(input_string):
        reference_pattern = r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"
        matches = re.findall(reference_pattern, input_string)
        return len(matches)
    
    def detect_multiple_edit_intent(msg):
        # should at least have imperative mood
        imperative_regex = re.compile(r'\b(?:fix|add|change|update|remove|refactor|improve|make|start|stop|debug|test|ensure|delete|merge|move|rename|clean|correct|allow|avoid|implement|complete|revert|set|increase|decrease|optimize|docs)\b', re.IGNORECASE)
        matches = imperative_regex.findall(msg)
        if len(matches) == 0:
            return False
        title = msg.split("\n")[0]
        body = "\n".join(msg.split("\n")[1:])

        # when body is empty
        if body.strip() == "":
            target = title
        else:
            target = body

        matches = imperative_regex.findall(msg)
        if len(set(matches)) == 1:
            return True
        else:
            return False
    
    commit_msg = commit_dict['commit']['message']  
    # 1. return False if commit message contain multiple edit intents
    single_intent = detect_multiple_edit_intent(commit_msg)
    if not single_intent:
        raise ValueError('1 Commit msg contain > 1 edit intention')
    
    # 2. return False if commit message is not in English
    if not commit_msg.isascii():
        raise ValueError('2 Commit msg contain non-ascii char')
    
    # 3. return False if commit message contain < 8 or > 128 words
    num_words = len(commit_msg.split(" "))
    if num_words < 8 or num_words > 128:
        raise ValueError('3 Commit msg contain < 8 words or > 128 words')
    # commit_msg_lines = commit_msg.split("\n")
    # if len(commit_msg_lines) < 3 or len(commit_msg_lines) > 5:
    #     return False
    
    # 4. return False if author or committer is not a User
    try:
        author_type = commit_dict['author']['type']
        committer_type = commit_dict['committer']['type']
        assert author_type == 'User' and committer_type == 'User'
    except:
        raise ValueError('4 Commit author / committer not real user')
    
    # 5. return False if commit message contain file names
    num_file_names = count_file_names(commit_msg)
    if num_file_names > 0:
        raise ValueError('5 Commit msg contain file name')
    
    # 6. return False if commit message contain external references
    num_external_references = count_external_references(commit_msg)
    if num_external_references > 0:
        raise ValueError('6 Commit msg contain external reference')
    
    # 7. return False if commit merge pull request, merge branch
    if "Merge pull request" in commit_msg or "Merge branch" in commit_msg:
        raise ValueError('7 Merge pull request / branch commit')
    
    return True

def clean_commit(lang):
    global ROOT_PATH
    with open(os.path.join(ROOT_PATH,f"commit_info/{lang}_commit_info.jsonl"), "r") as f:
        commit_set = [json.loads(line) for line in f.readlines()]
    error_cnt = {}
    filtered_commit_urls = []
    for commit in tqdm(commit_set):
        try:
            commit_filter(commit)
            filtered_commit_urls.append(commit["html_url"])
        except Exception as e:
            label = str(e).split(' ')[0]
            if label not in ['1', '2', '3', '4', '5', '6', '7']:
                print('Unexpected Error:', e)
                print('Commit url:', commit['html_url'])
                break
            else:
                if label not in error_cnt:
                    error_cnt[label] = 1
                else:
                    error_cnt[label] += 1
                    
    print(f'{lang} have {len(filtered_commit_urls)} left, survive rate: {len(filtered_commit_urls)/len(commit_set)*100:.2f}%')
    print('Commit filtered out because:')
    error_dict = {
        "1": "Commit msg contain > 1 edit intention",
        "2": "Commit msg contain non-ascii char",
        "3": "Commit msg contain < 8 words or > 128 words",
        "4": "Commit author / committer not real user",
        "5": "Commit msg contain file name",
        "6": "Commit msg contain external reference",
        "7": "Merge pull request / branch commit"
    }
    for error_idx, error_num in error_cnt.items():
        print(f'Rule {error_idx} {error_dict[error_idx]}: {error_num}')

    with open(os.path.join(ROOT_PATH, f'commit_info/{lang}_filtered_commit_urls.json'), 'w') as f:
        json.dump(filtered_commit_urls, f, indent=4)
    
if __name__ == '__main__':
    lang = 'python'
    clean_commit(lang)  