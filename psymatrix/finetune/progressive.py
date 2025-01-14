import json
import time

from functools import partial

import numpy as np

from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    AutoTokenizer,
)

from datasets import load_dataset

from psymatrix.finetune.utils import get_num_labels, tokenize_function


class SaveMetricsCallback(TrainerCallback):
    """
    Callback to save the metrics to
    """

    def __init__(
        self,
    ):
        self.metrics = []
        super().__init__()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Save the metrics to a file.
        """
        # Adding timestamp to metrics
        metrics["timestamp"] = time.time()

        self.metrics.append(metrics)

        fname = "metrics.json"

        # Ensure the output directory exists

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
            self.model.config.pad_token_id = tokenizer.pad_token_id

        self.tokenize = partial(
            tokenize_function,
            tokenizer,
            self.model_id,
            hyperparameters,
        )

        self.train_dataset = self.dataset[train_split].map(self.tokenize, batched=True)
        self.test_dataset = self.dataset[test_split].map(self.tokenize, batched=True)

        self.save_callback = SaveMetricsCallback()

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

        # Train the model
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
    learning_rate = 1e-5
    per_device_train_batch_size = 8
    per_device_eval_batch_size = 8
    max_seq_length = 128

    hyperparameters = {
        "max_tokens": max_seq_length,
    }

    ftuner = ProgressiveFineTuning(
        model_id="distilbert/distilbert-base-uncased",
        dataset_name_or_path="contemmcm/cls_amazonreviews2013_ReviewsummaryReviewtextVsReviewscore__ArtsFull",
        hyperparameters=hyperparameters,
    )

    for dataset_size in np.linspace(0.01, 1.0, 50):
        ftuner.finetune(
            dataset_size=dataset_size,
            num_train_epochs=1,
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
        )

    # Go for more epochs with the full dataset
    ftuner.finetune(
        dataset_size=1.0,
        num_train_epochs=3,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
    )


if __name__ == "__main__":
    run()
