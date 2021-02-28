"""Functions for spaCy language processing pipeline and computing
cosine similarity on spacy `Doc`s' word embeddings.
"""

import numpy as np

from tqdm import tqdm
from typing import List
from typing import Mapping
from typing import Union
from .utils import cosine_similarity
from .utils import normalise_vector

from spacy.language import Language
from spacy.tokens import Doc


# Selected POS tags (https://universaldependencies.org/u/pos/)
PARTS_OF_SPEECH = []

# Spacy pipe parameters
PIPE_PARAMS = {'n_process': 1,
               'batch_size': 50}


def spacy_preprocessor(
    texts: List[str],
    nlp: Language,
    disable: List[str] = ['ner', 'parser', 'textcat'],
    make_lemma: bool = True,
    make_lowercase: bool = True,
    keep_alpha: bool = True,
    keep_pos: List[str] = PARTS_OF_SPEECH,
    remove_punctuation: bool = True,
    remove_stopwords: bool = True,
    pipe_params: Mapping[str, object] = PIPE_PARAMS
) -> List[Doc]:
    """Given a list of text strings, returns a list of processed spaCy `Doc`
    objects (one for each text). A `Doc` object is a sequence of Token objects
    with methods to access Tokens' information (both original and processed)
    and to perform NLP tasks (e.g. words vector and similarity).

    An custom pipeline component is added to register a custom extention
    on the Doc. This custom extention (`Doc._.selected_tokens`) contains tokens
    that satisfy the `keep_words, `keep_pos`, `remove_punctuation`, and
    `remove_stopwords` conditions. Note that these conditions do not produce
    disjoint sets of tokens (e.g. only keeping NOUNS with `pos` logically
    excludes all punctuation).

    Note: if the conditions above are all False or None, then
    `Doc._.selected_tokens` contains all tokens that matches with the
    regular expression (\\w+)

    Args:
        texts (list of str): 
            List of text strings to process.

        nlp (spaCy Language object): 
            Pre-trained spaCy model (https://spacy.io/usage/models).

        disable (list of str): 
            Names of spaCy pipeline components to disable.

        make_lemmas (bool): 
            If True, adds tokens' lemma into the
            custom extention `Doc._.selected_tokens`;
            defaults to True. If False, disables spaCy's lemmatizer pipeline
            component (even if it not included in `disable`) and adds
            tokens' original text into the custom extention
            Doc._.selected_tokens.

        make_lowercase (bool): 
            If True, converts all tokens in custom extension
            Doc._selected_tokens into lowercase.

        keep_alpha (bool): 
            If True, only tokens with alphabetic characters are
            kept in the custom extention `Doc._.selected_tokens`

        keep_pos (list of str): 
            Only tokens with parts-of-speech tags in `keep_pos`
            are kept in the custom extention `Doc._.selected_tokens`.
            If `keep_pos` is empty, adds all tokens regardless of its
            part-of-speech tag. Defaults to `PARTS_OF_SPEECH = []`.

        remove_punctuation (bool): 
            If True, removes all punctuation in the custom extention
            Doc._.selected_tokens; defaults to True.

        remove_stopwords (bool): 
            If True, removes stopword tokens in the custom extention
            `Doc._.selected_tokens`; defaults to False. Stopword removal
            seems to be computationally expensive. Avoid using this
            unless necessary.

        pipe_params (Kwargs in `spacy.language.Language.pipe`): 
            See https://spacy.io/api/language#pipe

    Returns:
        A list of processed spaCy Doc objects.

    References: https://spacy.io/usage/processing-pipelines
    """
    component_config = {'filters': {
        'make_lemma': make_lemma,
        'make_lowercase': make_lowercase,
        'keep_alpha': keep_alpha,
        'keep_pos': keep_pos,
        'remove_punctuation': remove_punctuation,
        'remove_stopwords': remove_stopwords}}
    component_name = 'filter'
    if not(component_name in nlp.pipe_names):
        # Add custom component
        nlp.add_pipe('filter', config=component_config, last=True)
    docs = []
    for doc in tqdm(nlp.pipe(texts,
                             disable=disable,
                             **PIPE_PARAMS),
                    total=len(texts)):
        docs.append(doc)
    return docs


def spacy_similarity(docs: List[Doc],
                     text: str,
                     nlp: Language,
                     norm: Union[None, str] = "l2") -> List[float]:
    """Given a list of text in `docs` and `text` string,
    returns list of cosine similarity scores to `text`. Order of elements
    are left unchanged (i.e. identical to `docs`).

    Note: `norm` can be set to "l1", "l2", or "max" to scale vectors using
    the l1, l2, and max norm. Defaults to "l2".
    If None, vectors are not normalised.
    """

    def normalise(norm):
        def _normalise(v):
            # Numba linear algebra operations are only supported on
            # contiguous arrays
            v = np.ascontiguousarray(v)
            return normalise_vector(v, order)
        if norm == 'l1':
            order = 1
        if norm == 'l2':
            order = 2
        return _normalise

    query_doc = nlp(text)  # Convert text into a spaCy Doc object
    u = query_doc.vector
    vectors = list(map(
        lambda doc: np.average([np.array(token.vector_) for token
                                in doc._.filtered_matches], axis=0), docs))
    if norm:
        func = normalise(norm)
        u = func(u)
        vectors = list(map(lambda v: func(v), vectors))
    scores = list(map(lambda v: cosine_similarity(u, v)
                      if not(np.isnan(v).any()) else 0, vectors))
    return scores


if __name__ == "__main__":
    pass
