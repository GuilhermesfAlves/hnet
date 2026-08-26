import numpy as np


class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256
        self.bos_idx = 254
        self.eos_idx = 255
        self.dtype = np.uint8
        self.vocab = {
            i: bytes([i])
            for i in range(self.vocab_size)
        }

        self.id_to_token = self.vocab

    def encode(self, seqs: list[str], add_bos: bool = False, add_eos: bool = False, **kwargs) -> list[dict[str, np.ndarray]]:
        total_outputs = []
        for text in seqs:
            text_byte = text.encode("utf-8")

            if add_bos:
                text_byte = bytes([self.bos_idx]) + text_byte
            if add_eos:
                text_byte = text_byte + bytes([self.eos_idx])
            text_byte = bytearray(text_byte)
            text_byte_ids = np.array(text_byte, dtype=self.dtype)

            total_outputs.append({"input_ids": text_byte_ids})

        return total_outputs

    def decode(self, tokens: np.ndarray | list[int], skip_special_tokens=False, **kwargs) -> str:
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        return bytearray(tokens).decode("utf-8", **kwargs)
    
    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self.id_to_token[ids]

        return [
            self.id_to_token[token_id]
            for token_id in ids
        
        ]
