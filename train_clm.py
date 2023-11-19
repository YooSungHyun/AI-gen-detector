import copy
import json
import logging
import os

import datasets
import torch
import torch._dynamo
from arguments import ModelArguments, MyTrainingArguments
from datasets import concatenate_datasets, load_from_disk
from setproctitle import setproctitle
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, Trainer, set_seed
from transformers.trainer_utils import is_main_process
from utils import DataCollatorForSupervisedDataset

os.environ["TORCHDYNAMO_DISABLE"] = "1"
torch._dynamo.config.verbose = True
USER = "### Question:\n"
SYSTEM = "\n\n### Answer:\n"
IGNORE_INDEX = -100


def main(model_args: ModelArguments, training_args: MyTrainingArguments):
    setproctitle(training_args.task)
    set_seed(training_args.seed)
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, model_max_length=model_args.max_length)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        use_cache=False,
        low_cpu_mem_usage=True,
        device_map=device_map,
        # torch_dtype=torch.float16,
    )

    dataset = load_from_disk("")

    def preprocess(raw):
        input_text = USER + raw["table"] + "\n" + raw["question"] + SYSTEM
        label_text = raw["answer"] + tokenizer.eos_token
        total_text = input_text + label_text
        input_seq_token_len = len(tokenizer(input_text)["input_ids"])
        tokenized_text = tokenizer(total_text, return_token_type_ids=False, return_tensors="pt")
        raw["input_ids"] = tokenized_text["input_ids"][0]
        raw["attention_mask"] = tokenized_text["attention_mask"][0]

        labels_ids = copy.deepcopy(raw["input_ids"])
        labels_ids[:input_seq_token_len] = IGNORE_INDEX

        raw["labels"] = labels_ids
        return raw

    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda x: len(x["input_ids"]) < 2048, batched=False)
    dataset.set_format("torch")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        train_dataset=dataset,
        args=training_args,
    )
    trainer.train()
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, MyTrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )
    main(model_args=model_args, training_args=training_args)
