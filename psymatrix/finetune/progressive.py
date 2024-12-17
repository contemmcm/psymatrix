import json
import time

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
            None,  # Hyperparameters
        )

        self.train_dataset = self.dataset[train_split].map(self.tokenize, batched=True)
        self.test_dataset = self.dataset[test_split].map(self.tokenize, batched=True)

        self.save_callback = SaveMetricsCallback()

    def finetune(self, dataset_size: float = 1.0, num_train_epochs: int = 3):
        train_size = int(len(self.dataset[self.train_split]) * dataset_size)
        test_size = int(len(self.dataset[self.test_split]) * dataset_size)

        train_subset = self.train_dataset.select(range(train_size))
        test_subset = self.test_dataset.select(range(test_size))

        # Create a Trainer instance
        trainer = Trainer(
            model=self.model,
            args=self.get_training_args(num_train_epochs=num_train_epochs),
            train_dataset=train_subset,
            eval_dataset=test_subset,
            callbacks=[self.save_callback],
        )

        # Train the model
        trainer.train()

        # Evaluate the model
        results = trainer.evaluate()
        print(f"Results at increment {dataset_size*100}%: {results}")

    def get_training_args(self, **kwargs):
        default_args = {
            "output_dir": "./results",  # Output directory
            "evaluation_strategy": "epoch",  # Evaluation during training
            "save_strategy": "epoch",  # Save the model at the end of each epoch
            "per_device_train_batch_size": 8,  # Training batch size
            "per_device_eval_batch_size": 8,  # Evaluation batch size
            "num_train_epochs": 3,  # Number of training epochs
            "seed": 42,  # Seed for reproducibility
            "load_best_model_at_end": True,  # Load the best model at the end of training
        }

        default_args.update(kwargs)

        return TrainingArguments(**default_args)


if __name__ == "__main__":
    ftuner = ProgressiveFineTuning(
        model_id="distilbert/distilbert-base-uncased",
        dataset_name_or_path="PsyMatrix/cls_20newsgroups_SubjectTextVsLabel__BaseDefault",
    )

    ftuner.finetune(dataset_size=0.01, num_train_epochs=3)
    ftuner.finetune(dataset_size=0.02, num_train_epochs=3)
    ftuner.finetune(dataset_size=0.04, num_train_epochs=3)
    ftuner.finetune(dataset_size=0.08, num_train_epochs=3)
    ftuner.finetune(dataset_size=0.16, num_train_epochs=3)
    ftuner.finetune(dataset_size=0.32, num_train_epochs=3)
    ftuner.finetune(dataset_size=0.64, num_train_epochs=3)
    ftuner.finetune(dataset_size=1.00, num_train_epochs=3)
