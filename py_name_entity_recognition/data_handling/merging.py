import re

from py_name_entity_recognition.schemas.core_schemas import BaseEntity
from py_name_entity_recognition.utils.biores_converter import BIOSESConverter


class ChunkMerger:
    """
    A class to merge entity extractions from overlapping text chunks.

    This merger resolves conflicts and duplicates by prioritizing entities
    extracted from the center of a chunk, where LLMs are typically more accurate.
    """

    def __init__(self):
        """Initializes the ChunkMerger."""
        self.biores_converter = BIOSESConverter()

    def _calculate_confidence(
        self, entity_start_in_doc: int, chunk_start: int, chunk_end: int
    ) -> float:
        """
        Calculates a confidence score based on the entity's position within its chunk.

        The score is highest (1.0) at the chunk's center and decays linearly to 0.0
        at the edges, reflecting the tendency of LLMs to perform better in the middle
        of their context window.

        Returns:
            A confidence score between 0.0 and 1.0.
        """
        chunk_len = chunk_end - chunk_start
        if chunk_len == 0:
            return 0.0

        entity_pos_in_chunk = entity_start_in_doc - chunk_start
        distance_from_center = abs(entity_pos_in_chunk - (chunk_len / 2))

        # Normalize the distance to a score from 0.0 to 1.0
        score = 1.0 - (distance_from_center / (chunk_len / 2))
        return max(0.0, score)

    def merge(
        self,
        full_text: str,
        chunk_results: list[tuple[list[BaseEntity], int, int]],
    ) -> list[tuple[str, str]]:
        """
        Merges entity extractions from overlapping chunks into a single result.

        Args:
            full_text: The original, complete text document.
            chunk_results: A list of tuples, where each contains:
                           - A list of BaseEntity objects for a chunk.
                           - The chunk's start character offset in the full text.
                           - The chunk's end character offset in the full text.

        Returns:
            A final, unified list of BIOSES-tagged tokens for the full text.
        """
        scored_entities = []
        for entities, chunk_start, chunk_end in chunk_results:
            chunk_text = full_text[chunk_start:chunk_end]
            for entity in entities:
                try:
                    pattern = re.escape(entity.text)
                    for match in re.finditer(pattern, chunk_text):
                        entity_start_in_chunk = match.start()
                        entity_global_start = chunk_start + entity_start_in_chunk

                        confidence = self._calculate_confidence(
                            entity_global_start, chunk_start, chunk_end
                        )

                        # Store each potential entity with its global position and confidence
                        scored_entities.append(
                            (
                                entity_global_start,
                                entity_global_start + len(entity.text),
                                entity.type,
                                confidence,
                            )
                        )
                except re.error:
                    continue

        # Resolve overlaps by prioritizing entities with higher confidence scores.
        # We sort descending by confidence, so we process the best candidates first.
        scored_entities.sort(key=lambda x: x[3], reverse=True)

        final_spans: list[tuple[int, int, str]] = []
        claimed_chars: set[int] = set()

        for start, end, entity_type, _ in scored_entities:
            # If the character span of this entity overlaps with one we've already
            # accepted, we discard it.
            if any(c in claimed_chars for c in range(start, end)):
                continue

            # This entity is a high-confidence, non-overlapping candidate. Accept it.
            final_spans.append((start, end, entity_type))
            claimed_chars.update(range(start, end))

        # Perform a final BIOSES conversion on the clean, merged list of entities.
        return self.biores_converter.convert(full_text, final_spans)
