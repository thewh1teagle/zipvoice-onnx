import logging
from typing import List

# Punctuation set for chunking
_punctuation = {";", ":", ",", ".", "!", "?", "；", "：", "，", "。", "！", "？"}


class Tokenizer:
    """Simple tokenizer that reads token mappings from a file and converts text to tokens."""

    def __init__(self, token_file: str):
        """
        Args:
            token_file: Path to token file with format '{token}\t{token_id}' per line.
        """
        self.token2id: dict[str, int] = {}
        with open(token_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                info = line.rstrip().split("\t")
                if len(info) == 2:
                    token, token_id = info[0], int(info[1])
                    self.token2id[token] = token_id
        self.pad_id = self.token2id.get("_", 0)
        self.vocab_size = len(self.token2id)

    def texts_to_tokens(self, texts: List[str]) -> List[List[str]]:
        return [list(text.strip()) for text in texts]

    def tokens_to_token_ids(self, tokens_list: List[List[str]]) -> List[List[int]]:
        token_ids_list = []
        for tokens in tokens_list:
            token_ids = []
            for token in tokens:
                if token in self.token2id:
                    token_ids.append(self.token2id[token])
                else:
                    logging.debug(f"Skip OOV token: {token}")
            token_ids_list.append(token_ids)
        return token_ids_list


def chunk_tokens_punctuation(tokens_list: List[str], max_tokens: int = 100) -> List[List[str]]:
    """Split tokens into chunks at punctuation boundaries, each at most max_tokens long."""
    # Split into sentences at punctuation; trailing punctuation/spaces attach to previous sentence
    sentences: List[List[str]] = []
    current: List[str] = []
    for token in tokens_list:
        is_trailing = token in _punctuation or token == " "
        if not current and sentences and is_trailing:
            sentences[-1].append(token)
        else:
            current.append(token)
            if token in _punctuation:
                sentences.append(current)
                current = []
    if current:
        sentences.append(current)

    # Merge sentences into chunks up to max_tokens
    chunks: List[List[str]] = []
    current = []
    for sentence in sentences:
        if len(current) + len(sentence) <= max_tokens:
            current.extend(sentence)
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)

    return chunks
