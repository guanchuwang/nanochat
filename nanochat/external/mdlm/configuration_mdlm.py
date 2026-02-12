"""MDLM config for Hugging Face.

"""

import transformers


class MDLMConfig(transformers.PretrainedConfig):
  """Hugging Face configuration class for MDLM."""
  model_type = "mdlm"

  def __init__(
    self,
    vocab_size: int = 50258,
    model_length: int = 1024,
    hidden_dim: int = 768,
    cond_dim: int = 129,
    n_blocks: int = 12,
    n_heads: int = 12,
    dropout: float = 0.1,
    time_conditioning: bool = False,
    ** kwargs):
    super().__init__(**kwargs)
    self.vocab_size = vocab_size
    self.model_length = model_length
    self.hidden_dim = hidden_dim
    self.cond_dim = cond_dim
    self.n_blocks = n_blocks
    self.n_heads = n_heads
    self.dropout = dropout
    self.time_conditioning = time_conditioning
