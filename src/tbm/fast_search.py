"""
Fast Template Search Using K-mer Similarity

Uses k-mer (substring) matching for rapid template search instead of
full pairwise alignment. This is orders of magnitude faster.
"""

import numpy as np
from collections import defaultdict, Counter


def get_kmers(sequence, k=6):
    """
    Extract all k-mers from a sequence.

    Args:
        sequence (str): RNA sequence
        k (int): K-mer length

    Returns:
        Counter: K-mer counts
    """
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmers.append(sequence[i:i+k])
    return Counter(kmers)


def calculate_kmer_similarity(query_kmers, template_kmers):
    """
    Calculate Jaccard similarity between k-mer sets.

    Args:
        query_kmers (Counter): Query k-mers
        template_kmers (Counter): Template k-mers

    Returns:
        float: Jaccard similarity (0-1)
    """
    # Get all k-mers
    all_kmers = set(query_kmers.keys()) | set(template_kmers.keys())

    if len(all_kmers) == 0:
        return 0.0

    # Calculate intersection and union
    intersection = 0
    union = 0

    for kmer in all_kmers:
        q_count = query_kmers.get(kmer, 0)
        t_count = template_kmers.get(kmer, 0)
        intersection += min(q_count, t_count)
        union += max(q_count, t_count)

    if union == 0:
        return 0.0

    return intersection / union


class FastTemplateSearch:
    """
    I built this class to speed up template search using k-mer similarity.
    Full pairwise alignment is too slow for thousands of templates.
    """
    """
    Fast template search using k-mer similarity.

    Much faster than full pairwise alignment for initial screening.
    """

    def __init__(self, train_sequences_dict, k=6):
        """
        Initialize fast search index.

        Args:
            train_sequences_dict (dict): Maps template IDs to sequences
            k (int): K-mer length
        """
        self.k = k
        self.train_sequences = train_sequences_dict

        print(f"Building k-mer index (k={k})...")
        self.template_kmers = {}

        for template_id, sequence in train_sequences_dict.items():
            self.template_kmers[template_id] = get_kmers(sequence, k)

        print(f"  Indexed {len(self.template_kmers)} templates")

    def find_similar_templates(self, query_seq, top_n=10, min_similarity=0.2):
        """
        Find similar templates using k-mer similarity.

        Args:
            query_seq (str): Query sequence
            top_n (int): Number of templates to return
            min_similarity (float): Minimum k-mer similarity threshold

        Returns:
            list: List of (template_id, similarity) tuples
        """
        query_kmers = get_kmers(query_seq, self.k)

        similarities = []

        for template_id, template_kmers in self.template_kmers.items():
            sim = calculate_kmer_similarity(query_kmers, template_kmers)

            if sim >= min_similarity:
                similarities.append((template_id, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_n]


if __name__ == "__main__":
    print("Fast K-mer Based Template Search")
    print("\nExample:")

    # Example sequences
    train_seqs = {
        'template1': 'GGGGAAAAAUCCCCUUUUUUGGGGAAAA',
        'template2': 'AAAAUUUUGGGGCCCCUUUUAAAAGGG',
        'template3': 'CCCCGGGGAAAAUUUUCCCCGGGGAAA',
    }

    query = 'GGGGAAAAAUCCCCUUUU'

    # Build search index
    searcher = FastTemplateSearch(train_seqs, k=4)

    # Find templates
    matches = searcher.find_similar_templates(query, top_n=3)

    print("\nQuery:", query)
    print("\nTop matches:")
    for template_id, similarity in matches:
        print(f"  {template_id}: {similarity:.2%}")
