"""spaCy custom components and custom extensions

Custom Pipeline Components:
- filter -- filter tokens by alphabetical characters, parts-of-speech tags,
punctuation, and stopwords.

Custom Extensions on Doc:
- FilteredTokensComponent -- a list of filtered token matches.
Each match is a namedtuple called Match with three fields
(text, token_, vector_).
"""


from collections import OrderedDict
from collections import namedtuple
from typing import List
from typing import Mapping
from typing import Union

from .utils import intersect_dicts

from spacy.language import Language
from spacy.tokens import Doc
from spacy.matcher import Matcher


@Language.factory('filter')
def create_filter_component(
        nlp: Language,
        name: str,
        filters: Mapping[str, Union[bool, List[str]]]):
    return FilteredTokensComponent(nlp, filters)


# Custom extention on Doc
class FilteredTokensComponent:

    def __init__(self,
                 nlp: Language,
                 filters: Mapping[str, Union[bool, List[str]]]):
        # Create a list of rules (match_id, patterns) to match with
        # spacy.matcher.Matcher (https://spacy.io/api/matcher)
        self.matchers = []
        rules = []
        if filters.get('keep_alpha'):
            rules.append(('ALPHABETICAL',
                         [[{'IS_ALPHA': True}]]))
        if filters.get('keep_pos'):
            rules.append(('SELECTED_POS',
                         [[{'POS': pos}] for pos in filters.get('keep_pos')]))
        if filters.get('remove_punctuation'):
            rules.append(('NON_PUNC',
                         [[{'IS_PUNCT': False}]]))
        if filters.get('remove_stopwords'):
            rules.append(('NON_STOPWORDS',
                         [[{'IS_STOP': False}]]))
        # If no rules specified
        if not(rules):
            # Match on any whole word using regex
            matcher = Matcher(nlp.vocab)
            pattern = [{'TEXT': {'REGEX': r'(\w+)'}}]
            self.matchers.append(('ANY_WORD', matcher))
        else:
            # Create a list of Matcher objects
            for match_id, pattern in rules:
                matcher = Matcher(nlp.vocab)
                matcher.add(match_id, pattern)
                self.matchers.append((match_id, matcher))
        # Attributes
        self.make_lemma = filters.get('make_lemma')
        self.make_lowercase = filters.get('make_lowercase')
        # Register custom extention on the Doc
        if not Doc.has_extension('filtered_matches'):
            Doc.set_extension('filtered_matches', default=[])

    def __call__(self, doc: Doc) -> Doc:
        """Sets the intersection of matches from the list of
        Matchers in self.matches to `doc._.filtered_matches`.
        Returns `doc` with the custom extension.

        Note: a match is a namedtuple called Match with three fields
        (text, token_, vector_).
        """
        Match = namedtuple('Match', 'text token_ vector_')
        nested_filtered_matches = []
        # For each matcher
        for match_id, matcher in self.matchers:
            matches = OrderedDict()
            # Get all matched spans
            for match_id, start, end in matcher(doc):
                span = doc[start:end]
                if self.make_lemma and self.make_lowercase:
                    token = span.lemma_.lower()
                elif self.make_lemma:
                    token = span.lemma_
                elif self.make_lowercase:
                    token = span.text.lower()
                else:
                    token = span.text
                # Append match to matches set
                matches[span] = Match(span, token, span.vector.tolist())
            nested_filtered_matches.append(matches)
        # Get intersection of matches
        for i in range(len(nested_filtered_matches)):
            matches = nested_filtered_matches[i]
            if i == 0:
                filtered_matches = matches
            else:
                filtered_matches = intersect_dicts(filtered_matches, matches)
        # Set custom extension
        doc._.filtered_matches = list(filtered_matches.values())
        return doc


if __name__ == "__main__":
    pass
