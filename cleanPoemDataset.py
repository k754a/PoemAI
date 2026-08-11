#THIS WAS NOT CODED BY ME, BUT WAS FOUND ONLINE. I AM NOT THE AUTHOR OF THIS CODE. I AM SIMPLY USING IT TO CLEAN MY POEM DATASET.

import re

def clean_text_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # Keep only letters (a-z, A-Z) and whitespace
    cleaned = re.sub(r'[^a-zA-Z\s]', '', text)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned)

clean_text_file('poems.txt', 'output.txt')
