from config import MAX_TOKENS, TEMPERATURE, TOP_P
from utils import extract_and_format_response
import torch

system_prompt_plaintiff = (
    "You are reading a transcript from a courtroom conversation.\n\n"
    "Step 1: Carefully read the dialogue.\n"
    "Step 2: Think step-by-step about what the Plaintiff's statements suggest.\n"
    "Step 3: Reason about the Plaintiff's goal or motive behind their words.\n"
    "Step 4: Summarize the Plaintiff's intent in three complete sentences maximum. "
    "Ensure your response is complete and properly punctuated.\n\n"
    "Now analyze the given dialogue.\n"
)


system_prompt_defendant = (
    "You are reading a transcript from a courtroom conversation.\n\n"
    "Step 1: Carefully read the dialogue.\n"
    "Step 2: Think step-by-step about what the Defendant's statements suggest.\n"
    "Step 3: Reason about the Defendant's goal or motive behind their words.\n"
    "Step 4: Summarize the Defendant's intent in three complete sentences maximum. "
    "Ensure your response is complete and properly punctuated.\n\n"
    "Now analyze the given dialogue.\n"
)

def generate_text(prompt, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = output[0, input_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def analyze(dialogue, role, model, tokenizer, device):
    prompt = system_prompt_plaintiff if role == "plaintiff" else system_prompt_defendant

    messages = [
        {"role": "user", "content": f"{prompt}\nDialogue:\n{dialogue}\n\nProvide your analysis:"}
    ]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    generated_text = generate_text(input_text, model, tokenizer, device)
    return extract_and_format_response(generated_text)