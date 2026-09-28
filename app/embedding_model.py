import spacy


nlp = spacy.load("en_core_web_lg")


def calculate_embedding(input_word: str) -> list[float]:
    """
    Calculate the 300-dimensional spaCy embedding for an input word.
    """
    cleaned_word = input_word.strip()

    if not cleaned_word:
        raise ValueError("The input word cannot be empty.")

    word = nlp(cleaned_word)
    return word.vector.tolist()