# for type annotation
import logging
import os
from argparse import Namespace
from typing import Any, Dict

import numpy as np
import torch._dynamo
from arguments import ModelArguments, MyTrainingArguments, DatasetsArguments
from datasets import Dataset, concatenate_datasets, load_from_disk
from evaluate import load
from setproctitle import setproctitle
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import EvalPrediction, is_main_process

os.environ["TORCHDYNAMO_DISABLE"] = "1"
torch._dynamo.config.verbose = True
SEP_TOKEN = "[SEP]"


# 학습 시작 전에, 꼭 data_preprocess.ipynb 확인 후 진행할 것
def main(model_args: ModelArguments, datasets_args: DatasetsArguments, training_args: MyTrainingArguments):
    setproctitle(training_args.task)
    set_seed(training_args.seed)

    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
    print(tokenizer.model_max_length)
    model.config.num_labels = 2

    train_datasets = load_from_disk(datasets_args.train_datasets_path)
    eval_datasets = load_from_disk(datasets_args.eval_datasets_path)

    # SEP Token 합쳐서 학습해볼것(concat)
    def preprocess(raw):
        input_text = raw["text"]
        tokenized_text = tokenizer(input_text, return_token_type_ids=False, return_tensors="pt")
        # tokenized_text = tokenizer(input_text, return_tensors="pt")
        raw["input_ids"] = tokenized_text["input_ids"][0]
        raw["attention_mask"] = tokenized_text["attention_mask"][0]
        raw["labels"] = int(raw["generated"])

        return raw

    def filter_and_min_sample(datasets: Dataset, max_length: int = 512, min_sample_count: int = 0):
        datasets = datasets.filter(lambda x: len(x["input_ids"]) <= max_length)
        true_datasets = datasets.filter(lambda x: x["labels"] == 1)
        false_datasets = datasets.filter(lambda x: x["labels"] == 0)

        if min_sample_count:
            sampling_count = min(len(true_datasets), len(false_datasets), min_sample_count)
        else:
            sampling_count = min(len(true_datasets), len(false_datasets))

        sampling_true = Dataset.from_dict(true_datasets.shuffle()[:sampling_count])
        sampling_false = Dataset.from_dict(false_datasets.shuffle()[:sampling_count])
        filter_sampled_dataset = concatenate_datasets([sampling_true, sampling_false])
        assert len(filter_sampled_dataset) % 2 == 0, "`split=all` sampling error check plz"
        return filter_sampled_dataset

    train_datasets = train_datasets.map(preprocess, remove_columns=train_datasets.column_names)
    train_datasets = filter_and_min_sample(train_datasets, tokenizer.model_max_length)
    eval_datasets = eval_datasets.map(preprocess, remove_columns=eval_datasets.column_names)
    eval_datasets = filter_and_min_sample(eval_datasets, tokenizer.model_max_length)

    print("@@@@@ Train Datasets:", len(train_datasets), "\t", "@@@@@ Eval Datasets:", len(eval_datasets))

    # [NOTE]: load metrics & set Trainer arguments
    roc_auc = load("evaluate-metric/roc_auc")

    def metrics(evaluation_result: EvalPrediction) -> Dict[str, float]:
        """_metrics_
            evaluation과정에서 모델의 성능을 측정하기 위한 metric을 수행하는 함수 입니다.
            이 함수는 Trainer에 의해 실행되며 Huggingface의 Evaluate 페키로 부터
            각종 metric을 전달받아 계산한 뒤 결과를 반환합니다.

        Args:
            evaluation_result (EvalPrediction): Trainer.evaluation_loop에서 model을 통해 계산된
            logits과 label을 전달받습니다.

        Returns:
            Dict[str, float]: metrics 계산결과를 dict로 반환합니다.
        """

        metrics_result = dict()

        predictions = evaluation_result.predictions
        references = evaluation_result.label_ids

        predictions = np.argmax(predictions, axis=-1)

        accuracy_result = roc_auc.compute(prediction_scores=predictions, references=references)

        metrics_result["roc_auc"] = accuracy_result["roc_auc"]
        return metrics_result

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_datasets,
        eval_dataset=eval_datasets,
        args=training_args,
        compute_metrics=metrics,
        data_collator=collator,
    )

    # [NOTE]: run train, eval, predict
    if training_args.do_train:
        train(trainer, training_args)
    if training_args.do_eval:
        eval(trainer, eval_datasets)


def train(trainer: Trainer, args: Namespace) -> None:
    """_train_
        Trainer를 전달받아 Trainer.train을 실행시키는 함수입니다.
        학습이 끝난 이후 학습 결과 그리고 최종 모델을 저장하는 기능도 합니다.

        만약 학습을 특정 시점에 재시작 하고 싶다면 Seq2SeqTrainingArgument의
        resume_from_checkpoint을 True혹은 PathLike한 값을 넣어주세요.

        - huggingface.trainer.checkpoint
        https://huggingface.co/docs/transformers/main_classes/trainer#checkpoints

    Args:
        trainer (Trainer): Huggingface의 torch Trainer를 전달받습니다.
        args (Namespace): Seq2SeqTrainingArgument를 전달받습니다.
    """
    outputs = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    metrics = outputs.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_model(args.output_dir)


def eval(trainer: Trainer, eval_data: Dataset) -> None:
    """_eval_
        Trainer를 전달받아 Trainer.eval을 실행시키는 함수입니다.
    Args:
        trainer (Trainer): Huggingface의 torch Trainer를 전달받습니다.
        eval_data (Dataset): 검증을 하기 위한 Data를 전달받습니다.
    """
    trainer.evaluate(eval_data)


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DatasetsArguments, MyTrainingArguments))
    model_args, datasets_args, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )
    main(model_args=model_args, datasets_args=datasets_args, training_args=training_args)
