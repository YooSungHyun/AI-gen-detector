from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(default=None)
    max_length: int = field(default=512)

    # It just for DPO
    ref_model_name_or_path: Optional[str] = field(
        default="", metadata={"help": "DPO ref model path if None or '', same to training model"}
    )
