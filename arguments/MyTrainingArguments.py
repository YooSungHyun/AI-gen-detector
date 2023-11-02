from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class MyTrainingArguments(TrainingArguments):
    task: Optional[str] = field(default="training", metadata={"help": "what is your process name?"})
    dropout_rate: Optional[float] = field(default=0.0, metadata={"help": "dropout rate"})

    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})

    # used for DPOTrainer
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    max_prompt_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum length of the prompt. This argument is required if you want to use the default data collator."
        },
    )
    label_pad_token_id: Optional[int] = field(
        default=-100,
        metadata={
            "help": "The label pad token id. This argument is required if you want to use the default data collator."
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
