import re
import time
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore")

def load_llama3():
    access_token = ''
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained("/media/user/llama3", token=access_token)
    model = AutoModelForCausalLM.from_pretrained(
        "/media/user/llama3",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=access_token
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer 

def ask_llama3(question: str, model=None, tokenizer=None, answer_length=256):
    if model is None or tokenizer is None:
        access_token = ''
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained("/media/user/llama3", token=access_token)
        model = AutoModelForCausalLM.from_pretrained(
            "/media/user/llama3",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=access_token
        )
        
    messages = [
        {"role": "system", "content": "You are a commit review expert with extensive experience in software development and code quality assurance. With a deep understanding of programming languages and software engineering principles, you excel at identifying potential issues in code commits. Your meticulous attention to detail ensures that every line of code is scrutinized for efficiency, security, and compliance with best practices. Your feedback is constructive and aimed at enhancing code readability and maintainability. You are adept at using version control systems and code review tools to facilitate effective communication and collaboration among development teams. Your expertise also includes mentoring junior developers and promoting a culture of continuous learning and improvement within your team."},
        {"role": "user", "content": question},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.1,
        top_p=0.1,
    )
    response = outputs[0][input_ids.shape[-1]:]
    
    return tokenizer.decode(response, skip_special_tokens=True)

def parse_answer(answer):
    # Define regex patterns
    change_description_pattern = r"specific change description: (Yes|No)"
    single_modification_pattern = r"single modification task: (Yes|No)"
    simplified_message_pattern = r"Simplified commit message: (.+)"

    # Extract information using regex
    change_description = re.search(change_description_pattern, answer)
    single_modification = re.search(single_modification_pattern, answer)
    simplified_message = re.search(simplified_message_pattern, answer)

    # Convert extracted information
    change_description = change_description.group(1) == 'Yes' if change_description else False
    single_modification = single_modification.group(1) == 'Yes' if single_modification else False
    simplified_message = simplified_message.group(1) if simplified_message else ""

    return change_description, single_modification, simplified_message

if __name__ == "__main__":
    model, tokenizer = load_llama3()
    with open("prompt.txt", "r") as f:
        prompt = f.read()

    commit_msg = """Adding OpenLigaDB (#2746)

* Adding OpenLigaDB

* Rename YES -> Yes

Ref #2746

Co-authored-by: Matheus Felipe <50463866+matheusfelipeog@users.noreply.github.com>

Co-authored-by: Pawel Borkar <36134699+pawelborkar@users.noreply.github.com>
Co-authored-by: Matheus Felipe <50463866+matheusfelipeog@users.noreply.github.com>"""

    prompt = prompt.replace("<commit_message>", commit_msg)
    times = []
    for i in range(1):
        start = time.time() 
        answer = ask_llama3(prompt, model, tokenizer)
        print(answer)
        # print(parse_answer(answer))
        end = time.time()
        times.append(end - start)
    print(f"Average time cost: {sum(times) / len(times):.6f}s")
    