import re
from typing import Callable, List, Tuple
from spacy.tokens import Doc, Span
from spacy_ann.consts import stopwords

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
    return text.strip() or span.text