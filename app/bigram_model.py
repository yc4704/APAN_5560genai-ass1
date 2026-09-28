import random
import re
from collections import Counter, defaultdict


class BigramModel:
    def __init__(self, corpus: list[str]):
        """
        Create a bigram language model from a list of training sentences.
        """
        text = " ".join(corpus)
        words = self._tokenize(text)

        bigrams = list(zip(words[:-1], words[1:]))

        bigram_counts = Counter(bigrams)
        outgoing_counts = Counter(word1 for word1, _ in bigrams)

        self.bigram_probs: dict[str, dict[str, float]] = defaultdict(dict)

        for (word1, word2), count in bigram_counts.items():
            self.bigram_probs[word1][word2] = (
                count / outgoing_counts[word1]
            )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Convert text to lowercase and extract word tokens.
        """
        return re.findall(r"\b\w+\b", text.lower())

    def generate_text(self, start_word: str, length: int = 20) -> str:
        """
        Generate text by repeatedly sampling from bigram probabilities.
        """
        current_word = start_word.lower().strip()
        generated_words = [current_word]

        for _ in range(length - 1):
            next_words = self.bigram_probs.get(current_word)

            if not next_words:
                break

            current_word = random.choices(
                population=list(next_words.keys()),
                weights=list(next_words.values()),
                k=1,
            )[0]

            generated_words.append(current_word)

        return " ".join(generated_words)