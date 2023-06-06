import torch
from typing import Optional, Tuple
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput

@dataclass
class SequenceAnalysisOutput(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    # delta_logits: torch.FloatTensor = None
    recon_logits: torch.FloatTensor = None

    influntial_ids: Optional[torch.FloatTensor] = None
    influntial_idx_coss: Optional[torch.FloatTensor] = None
    part_input_idss: Optional[torch.FloatTensor] = None
    word_cosss: Optional[torch.FloatTensor] = None

@dataclass
class WordLists:
    def __init__(self, words_1, words_2, words_3):
        self.words_1 = words_1
        self.words_2 = words_2
        self.words_3 = words_3
    
    def get(self):
        return self.words_1 , self.words_2, self.words_3

@dataclass
class SVDSequenceAnalysisOutput(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    # delta_logits: torch.FloatTensor = None
    recon_logits: torch.FloatTensor = None

    word_coss: Optional[WordLists] = None