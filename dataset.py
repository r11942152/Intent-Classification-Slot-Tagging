from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab
import torch

class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        #raise NotImplementedError
        text_sequence = [sample['text'].split() for sample in samples]
        encoded_text_sequence = self.vocab.encode_batch(text_sequence)
        encoded_tensor_sequence = torch.LongTensor(encoded_text_sequence)
        id = [sample['id'] for sample in samples]

        if 'intent' in samples[0].keys():
            intent_id = [self.label2idx(sample['intent']) for sample in samples]
            label = torch.LongTensor(intent_id)
            return encoded_tensor_sequence, label
        else:
            return encoded_tensor_sequence, id

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        tag_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.tag_mapping = tag_mapping
        self._idx2tag = {idx: intent for intent, idx in self.tag_mapping.items()}
        self.max_len = max_len
        self.ignore_idx = -100

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    
    @property
    def num_classes(self) -> int:
        return len(self.tag_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        text_sequence = [sample['tokens'] for sample in samples]
        batch_len = [len(seq) for seq in text_sequence]
        encoded_text_sequence = self.vocab.encode_batch(text_sequence, self.max_len)
        data = torch.LongTensor(encoded_text_sequence)
        id = [sample['id'] for sample in samples]
        
        if 'tags' in samples[0].keys():
            batch_tags = [[self.tag2idx(tag) for tag in sample['tags']] + [self.ignore_idx]*(self.max_len-len(sample['tags'])) for sample in samples]
            label = torch.LongTensor(batch_tags)
            return data, batch_len, label
        else:
            return data, batch_len, id
        

    def tag2idx(self, tag: str):
        return self.tag_mapping[tag]

    def idx2tag(self, idx: int):
        return self._idx2tag[idx]

class SeqTaggingClsDataset2(SeqClsDataset):
    ignore_idx = -100

    def collate_fn(self, samples):
        # TODO: implement collate_fn
        raise NotImplementedError
