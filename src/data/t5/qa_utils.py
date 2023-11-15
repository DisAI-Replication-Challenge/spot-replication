import re
import string


def _normalize_answer(text, punc_chars, punc_repl):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(s):
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def replace_punctuation(s):
        to_replace = set(punc_chars)
        return "".join(punc_repl if ch in to_replace else ch for ch in s)

    def white_space_fix(s):
        return " ".join(s.split())

    text = text.lower()
    text = replace_punctuation(text)
    text = remove_articles(text)
    text = white_space_fix(text)

    return text


def normalize_trivia_qa(answer):
    """Normalization used in official TriviaQA evaluation script."""
    return _normalize_answer(
        answer, punc_chars=string.punctuation + "‘’´`_", punc_repl=" ").strip()


def normalize_squad(answer):
    """Normalization used in official SQuAD evaluation script."""
    return _normalize_answer(answer, punc_chars=string.punctuation, punc_repl="")
