"""
Usage:

$ python -m psymatrix.finetune.progressive -e "acl25" \
    -d "contemmcm/cls_amazonreviews2013_ReviewsummaryReviewtextVsReviewscore__ArtsFull"

$ python -m psymatrix.finetune.progressive -m "meta-llama/Llama-3.2-1B" \
  -d "contemmcm/cls_amazonreviews2013_ReviewsummaryReviewtextVsReviewscore__ArtsFull"
"""

import argparse
import json
import time
import os

from functools import partial

from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    AutoTokenizer,
)

from datasets import load_dataset

from psymatrix.finetune.utils import get_num_labels, tokenize_function
from psymatrix.experiments import load_models

parser = argparse.ArgumentParser(
    description="Finetune a pretrained model on a specific task-dataset."
)

parser.add_argument(
    "-m",
    "--model",
    dest="model_id",
    type=str,
    help="The pre-trained model ID. Overwrites model if used in conjunction with --experiment. E.g., 'google/bert-base-cased'.",
    required=False,
)

parser.add_argument(
    "-d",
    "--dataset",
    dest="dataset_id",
    type=str,
    help="The name or path of the dataset. Overwrites datasets if used in conjunction with --experiment. E.g., 'SetFit/20_newsgroups'.",
    required=False,
)

parser.add_argument(
    "-e",
    "--experiment",
    dest="experiment_id",
    type=str,
    help="The name of the experiment. E.g., 'finetune'.",
    required=False,
)


class SaveMetricsCallback(TrainerCallback):
    """
    Callback to save the metrics to
    """

    def __init__(
        self,
        output_file="metrics.json",
    ):
        self.metrics = []
        self.output_file = os.path.join("results", "progressive", output_file)
        super().__init__()

    def is_output_file_present(self):
        return os.path.exists(self.output_file)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Save the metrics to a file.
        """
        # Adding timestamp to metrics
        metrics["timestamp"] = time.time()

        self.metrics.append(metrics)

        fname = self.output_file

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(fname), exist_ok=True)

        with open(fname, "w", encoding="utf8") as f:
            json.dump(self.metrics, f, indent=2)


class ProgressiveFineTuning:

    def __init__(
        self,
        model_id,
        dataset_name_or_path,
        train_split="train",
        test_split="test",
        hyperparameters=None,
        **kwargs,
    ):
        self.model_id = model_id
        self.train_split = train_split
        self.test_split = test_split

        self.dataset = load_dataset(dataset_name_or_path).shuffle(seed=42)
        self.num_labels = get_num_labels(self.dataset)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=self.num_labels,
        )

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = tokenizer.pad_token_id

        self.tokenize = partial(
            tokenize_function,
            tokenizer,
            self.model_id,
            hyperparameters,
        )

        self.train_dataset = self.dataset[train_split].map(self.tokenize, batched=True)
        self.test_dataset = self.dataset[test_split].map(self.tokenize, batched=True)
        self.output_file = f"{dataset_name_or_path}/{model_id}/metrics.json"

        self.save_callback = SaveMetricsCallback(
            output_file=self.output_file,
        )

    def is_output_file_present(self):
        return self.save_callback.is_output_file_present()

    def finetune(self, dataset_size: float = 1.0, **kwargs):

        train_size = int(len(self.dataset[self.train_split]) * dataset_size)
        test_size = int(len(self.dataset[self.test_split]) * dataset_size)

        if 0 < dataset_size < 1:
            train_subset = self.train_dataset.select(range(train_size))
            test_subset = self.test_dataset.select(range(test_size))
        else:
            train_subset = self.train_dataset
            test_subset = self.test_dataset

        # Create a Trainer instance
        trainer = Trainer(
            model=self.model,
            args=self.get_training_args(**kwargs),
            train_dataset=train_subset,
            eval_dataset=test_subset,
            callbacks=[self.save_callback],
        )

        if dataset_size == 0:
            trainer.evaluate()  # Save the metrics before any training
        else:
            trainer.train()

    def get_training_args(self, **kwargs):
        default_args = {
            "evaluation_strategy": "epoch",
            "save_strategy": "no",
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "num_train_epochs": 3,
            "seed": 42,
            "load_best_model_at_end": False,
            "output_dir": "results",
        }

        default_args.update(kwargs)

        return TrainingArguments(**default_args)


def run():
    """
    Run the progressive fine-tuning experiment.
    """
    args = parser.parse_args()

    learning_rate = 1e-5
    per_device_train_batch_size = 8
    per_device_eval_batch_size = 8
    max_seq_length = 128
    num_epochs_per_size = 1

    hyperparameters = {
        "max_tokens": max_seq_length,
    }

    if args.experiment_id:
        models_ids = load_models(args.experiment_id)
    elif args.model_id:
        models_ids = [args.model_id]
    else:
        raise ValueError("Please provide an experiment ID or model ID.")

    for model_id in models_ids:
        print(f"Running {model_id}...")
        try:
            ftuner = ProgressiveFineTuning(
                model_id=model_id,
                dataset_name_or_path=args.dataset_id,
                hyperparameters=hyperparameters,
            )
        except Exception as e:
            print(f"Error: {e}")
            continue

        if ftuner.is_output_file_present():
            print(f"Skipping {model_id}...")
            continue

        for dataset_size in (
            0,
            1 / 128,
            1 / 64,
            1 / 32,
            1 / 16,
            1 / 8,
            1 / 4,
            1 / 2,
            1,
        ):

            try:
                ftuner.finetune(
                    dataset_size=dataset_size,
                    num_train_epochs=num_epochs_per_size,
                    learning_rate=learning_rate,
                    per_device_train_batch_size=per_device_train_batch_size,
                    per_device_eval_batch_size=per_device_eval_batch_size,
                )
            except Exception as e:
                print(f"Error: {e}")
                # continue


if __name__ == "__main__":
    run()
