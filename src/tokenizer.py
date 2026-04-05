import torch
from collections import Counter

class LatexTokenizer:
    def __init__(self, min_freq=3):
        """
        min_freq: Tokens appearing fewer times than this will be treated as <UNK>.
        """
        self.PAD_IDX = 0
        self.SOS_IDX = 1
        self.EOS_IDX = 2
        self.UNK_IDX = 3
        self.min_freq = min_freq
        
        self.word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.vocab_size = 4

    def fit_on_texts(self, list_of_formulas):
        print("Counting token frequencies...")
        token_counts = Counter()
        
        for formula in list_of_formulas:
            tokens = formula.split()
            token_counts.update(tokens)
            
        print(f"Found {len(token_counts)} unique raw tokens.")
        
        for token, count in token_counts.items():
            if count >= self.min_freq:
                self.word2idx[token] = self.vocab_size
                self.idx2word[self.vocab_size] = token
                self.vocab_size += 1

        print(f"Final Vocabulary built! Total unique tokens: {self.vocab_size}")

    def encode(self, formula_string):
        tokens = formula_string.split()
        encoded = [self.SOS_IDX]
        for token in tokens:
            encoded.append(self.word2idx.get(token, self.UNK_IDX))
        encoded.append(self.EOS_IDX)
        return torch.tensor(encoded, dtype=torch.long)

    def decode(self, token_ids):
        words = []
        for idx in token_ids:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            if idx == self.EOS_IDX:
                break
            if idx not in [self.PAD_IDX, self.SOS_IDX]:
                words.append(self.idx2word.get(idx, '<UNK>'))
        return " ".join(words)