import re

def extract_and_format_response(text):
    if "Step 4:" in text:
        response_text = text.split("Step 4:")[-1].strip()
    else:
        response_text = text.strip()

    sentences = re.split(r'(?<=[.!?]) +', response_text)
    complete_sentences = [s for s in sentences if s and s[-1] in ['.', '!', '?']]
    limited_sentences = complete_sentences[:3]

    formatted_response = ' '.join(limited_sentences)

    if not formatted_response:
        if response_text and response_text[-1] not in ['.', '!', '?']:
            response_text += '.'
        return response_text[:500]

    return formatted_response