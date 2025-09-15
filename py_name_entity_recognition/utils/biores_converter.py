from typing import Optional

import spacy
from spacy.language import Language

from py_name_entity_recognition.observability.logging import logger


class BIOSESConverter:
    """
    Converts structured entity predictions into BIOSES-tagged tokens.

    This class uses a spaCy model to tokenize the source text and then maps the
    character-level spans of extracted entities to the token level, assigning
    the appropriate BIOSES (Beginning, Inside, Outside, Single, End) tag.
    """

    def __init__(self, nlp: Optional[Language] = None):
        """
        Initializes the converter with a spaCy Language object.

        Args:
            nlp: A loaded spaCy Language object. If None, it will attempt to load
                 'en_core_web_sm' by default.

        Raises:
            IOError: If the default spaCy model cannot be loaded.
        """
        if nlp:
            self.nlp = nlp
        else:
            logger.info("No spaCy model provided. Loading 'en_core_web_sm' by default.")
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError as e:
                logger.error(
                    "Could not load 'en_core_web_sm'. Please run the following command to install it:\n"
                    "python -m spacy download en_core_web_sm"
                )
                raise OSError("Default spaCy model not found.") from e

    def convert(
        self, text: str, entities_with_spans: list[tuple[int, int, str]]
    ) -> list[tuple[str, str]]:
        """
        Converts a list of entities with character spans into BIOSES-tagged tokens.

        Args:
            text: The original source text.
            entities_with_spans: A list of tuples, where each tuple contains
                                 (start_char, end_char, entity_type).

        Returns:
            A list of (token, tag) tuples.
        """
        doc = self.nlp(text)
        tags = ["O"] * len(doc)

        if not entities_with_spans:
            return list(zip([token.text for token in doc], tags))

        # Sort by span length (descending) to handle nested entities correctly.
        sorted_entities = sorted(
            entities_with_spans, key=lambda e: e[1] - e[0], reverse=True
        )

        for start_char, end_char, entity_type in sorted_entities:
            span = doc.char_span(start_char, end_char, label=entity_type)

            if span is None:
                entity_text = text[start_char:end_char]
                logger.warning(
                    f"Entity span '{entity_text}' at chars ({start_char}-{end_char}) "
                    "does not align with token boundaries. Skipping this occurrence."
                )
                continue

            if any(tags[i] != "O" for i in range(span.start, span.end)):
                logger.debug(
                    f"Skipping entity span at ({start_char}-{end_char}) due to token overlap."
                )
                continue

            if len(span) == 1:
                tags[span.start] = f"S-{entity_type}"
            else:
                tags[span.start] = f"B-{entity_type}"
                for i in range(span.start + 1, span.end - 1):
                    tags[i] = f"I-{entity_type}"
                tags[span.end - 1] = f"E-{entity_type}"

        return list(zip([token.text for token in doc], tags))
