from typing import Dict

import torch
from torch.nn import Embedding
from torch import nn

class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.gru = nn.GRU(
            input_size=embeddings.shape[1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        D = 2 if bidirectional else 1
        self.output_size = hidden_size * D
        self.dp = nn.Dropout(p = dropout)
        self.fc = nn.Linear(self.output_size, num_class)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        #raise NotImplementedError
        return self.output_size

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        #raise NotImplementedError
        embedings = self.embed(batch)
        raw_output, _ = self.gru(embedings)
        output = raw_output[:, -1, :]
        output = self.dp(output)
        output = self.fc(output)

        return output
    
# slot tagging model
class SeqTagger(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        padding_idx: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqTagger, self).__init__()
        self.embed = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx = padding_idx)
        # TODO: model architecture
        self.gru = nn.GRU(
            input_size=embeddings.shape[1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        D = 2 if bidirectional else 1
        self.output_size = hidden_size * D
        self.dp = nn.Dropout(p = dropout)
        self.fc = nn.Linear(self.output_size, num_class)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        return self.output_size

    def forward(self, batch) -> torch.Tensor:
        # TODO: implement model forward
        embedings = self.embed(batch)
        output, _ = self.gru(embedings)
        output = self.dp(output)
        output = output.reshape(-1, self.output_size)
        output = self.fc(output)
        
        return output

class SeqTagger2(SeqClassifier):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        raise NotImplementedError
