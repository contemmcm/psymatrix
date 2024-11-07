import os

from django.db import models

from model_utils.managers import InheritanceManager


# Create your models here.
class TextDataset(models.Model):
    """
    A generic text dataset
    """

    name_or_path = models.CharField(max_length=256, db_index=True)

    train_size = models.FloatField(blank=True, null=False)
    test_size = models.FloatField(blank=True, null=False)

    objects = InheritanceManager()

    _dataset = None

    def __str__(self):
        return self.name_or_path


class ClassificationTextDataset(TextDataset):
    """
    A dataset for text classification
    """

    LABEL_COLUMN = "label"

    num_labels = models.IntegerField(blank=True, null=False)

    def save(self, *args, **kwargs):

        dataset = self.load()

        # Counting the number of labels
        labels = set(dataset["train"][self.LABEL_COLUMN])
        labels.update(set(dataset["test"][self.LABEL_COLUMN]))

        self.num_labels = len(labels)

        super().save(*args, **kwargs)

    class Meta:
        verbose_name = "Text Classification Dataset"
        verbose_name_plural = "Text Classification Datasets"
