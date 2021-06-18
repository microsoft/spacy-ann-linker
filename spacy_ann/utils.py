import string

def normalize_text(text):
    # remove special characters
    PUNCTABLE = str.maketrans("", "", string.punctuation)
    ascii_punc_chars = dict(
        [it for it in PUNCTABLE.items() if chr(it[0])])
    text = text.translate(ascii_punc_chars)
    # remove space if not english
    if not text.isascii():
        text = text.replace(' ', '')
    return text
