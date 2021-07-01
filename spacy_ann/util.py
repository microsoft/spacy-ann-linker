import re
from typing import Callable, List, Tuple
from spacy.tokens import Doc, Span
from spacy_ann.consts import stopwords
import string

def normalize_text(text):
    # remove special characters
    PUNCTABLE = str.maketrans("", "", string.punctuation)
    ascii_punc_chars = dict(
        [it for it in PUNCTABLE.items() if chr(it[0])])
    text = text.translate(ascii_punc_chars)
    # remove space if not english
    if not all([c for c in text if ord(c)>127]):
        text = text.replace(' ', '')
    return text

def get_spans(doc: Doc)-> List[Span]:
    """get all spans from doc"""
    link_spans =doc.spans.get('annlink',[])
    return list(doc.ents) + link_spans


def get_span_text(nlp, span):
    """ transform span text by delete redundency words

    Args:
        span ([type]): entity

    Returns:
        span text
    """
    text = span.text

    if len(text) > 3:
        if span.label_ == 'ingredient':
            exc_words = stopwords
            text = re.sub('|'.join(exc_words), '', text)
        elif span.label_ == 'brand' and '/' in text:
            text = text.split('/')[0]
        # replace GPE
        doc = nlp.get_pipe('ner')(nlp.make_doc(span.text))
        loc_ents = [ent for ent in doc.ents if ent.label_ == 'GPE']
        for ent in loc_ents:
            text = text.replace(ent.text, '')
        text = normalize_text(text)
    return text.strip() or span.text