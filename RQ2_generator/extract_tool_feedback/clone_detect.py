import os
import subprocess

from utils import *
from rapidfuzz import fuzz

def find_line_numbers(start_char_pos, end_char_pos, document_in_lines):
    line_idx = [] 
    current_char_count = 0
    
    for index, line in enumerate(document_in_lines):
        line_length = len(line)
        next_char_count = current_char_count + line_length
        
        if start_char_pos < next_char_count and end_char_pos > current_char_count:
            start = max(start_char_pos, current_char_count)
            end = min(next_char_count, next_char_count)

            intersection_length = max(0, end - start + 1)
            if intersection_length / len(line) > 0.75:
                line_idx.append(index)
        
        current_char_count = next_char_count
    
    return line_idx#[1:-1]

def partial_scs(query, document, threshold, left, right):
    result = fuzz.partial_ratio_alignment(query, document, score_cutoff=threshold)
    if result is None or (result.src_end - result.src_start) / len(query) < 0.75:
        return []
    start_char = left + result.dest_start
    end_char = left + result.dest_end
    segments = [{
        'score': result.score,
        'start_char': start_char,
        'end_char': end_char
    }]
    left_segments = partial_scs(query, document[left : start_char], threshold, left=left, right=start_char)
    right_segments = partial_scs(query, document[end_char : right], threshold, left=end_char, right=right)
    return left_segments + segments + right_segments
    
def find_similar_code_segment(query, original_document_lines, threshold=80):
    """
    Func:
        Find all similar code segments in the document
    Args:
        query: str, the code segment to search
        document: str, the document to search in
        threshold: int, the similarity threshold
    Returns:
        found_segments: list, a list of found segments
                        {
                            "score": int, the similarity score,
                            "matched_lines": list, a list of line numbers where the code is found, indexed from 0
                        }
    """
    if len(query.strip()) < 15:
        return []
    
    found_segments = []
    document = "".join(original_document_lines)

    char_segments = partial_scs(query, document, threshold, left=0, right=len(document))
    for segment in char_segments:
        found_line_range = find_line_numbers(segment['start_char'], segment['end_char'], original_document_lines)
        if found_line_range == []:
            continue
        found_segments.append({
            "score": segment['score'],
            "matched_lines": found_line_range
        })
    return found_segments
  
def find_clone_in_project(latest_version, query, threshold=80):
    """
    Func:
        Find all similar code segments in the project
    Args:
        project_path: str, the path of the project to search in
        query: str, the code segment to search
        threshold: int, the similarity threshold
    Returns:
        found_clones: list, a list of found segments
                        {
                            "file_path": str, the file path where the code is found,
                            "score": int, the similarity score,
                            "matched_lines": list, a list of line numbers where the code is found, indexed from 0
                        }
    """
    found_clones = []
    
    for file_path, original_document_lines in latest_version.items():
        found_segments = find_similar_code_segment(query, original_document_lines, threshold)
        if found_segments != []:
            for segment in found_segments:
                assert segment["matched_lines"] != []
                found_clones.append({
                    "file_path": file_path,
                    "score": segment["score"],
                    "matched_lines": segment["matched_lines"]
                })

    return found_clones

if __name__ == "__main__":
    project_dir = '/media/user/repos/stable-diffusion-webui'
    sha = '0646040667b59526ac8346d53efd14dc0e75b01e'
    command = f'git -C {project_dir} checkout {sha}^'
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    code_to_search = """self.url_txt2img = "http://localhost:7860/sdapi/v1/txt2img"""
    results = find_clone_in_project(project_dir, code_to_search, 80)
    for result in results:
        print(f"Found similar code in {result['file']} with similarity {result['score']} at line {result['matched_lines']}%")