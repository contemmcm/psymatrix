# PsyMatrix


PsyMatrix helps to determine the most suitable pre-trained model for fine-tuning on a given text dataset. This guide walks you through a case study where we identify the best pre-trained model for the dataset [cardiffnlp/tweet_topic_single](https://huggingface.co/datasets/cardiffnlp/tweet_topic_single).

## Prerequisites
- Python 3.9 or superior
- `wget`
- `gdown`
- `make`
- CohMetrix (requires a license)

## Case Study

In this case study, we'll use PsyMatrix to determine the best pre-trained model for the [cardiffnlp/tweet_topic_single dataset](https://huggingface.co/datasets/cardiffnlp/tweet_topic_single).

### Step 1: Set Environment Variables

First, set the environment variables to specify the dataset and splits (train and test):

```bash
export DATASET="cardiffnlp/tweet_topic_single"
export TRAIN_SPLIT="train_random"
export TEST_SPLIT="validation_random"
```

### Step 2: Download PsyMatrix Models and Tools

To get started, download PsyMatrix's pre-trained models and companion software. Simply run:

```bash
make download
```

This will fetch all necessary resources. Ensure that `wget` and `gdown` are installed.

PsyMatrix uses CohMetrix for dataset characterization. The recommended way to install CohMetrix is via Docker. Run the following command to build the Docker image (only tested on Linux):

```bash
COHMETRIX_LICENSE=xxxx make cohmetrix
```

Replace `xxxx` with your CohMetrix license key. Please refer to http://cohmetrix.memphis.edu/home and https://github.com/memphis-iis/cohmetrix-issues for more information


### Step 3: Characterize the Dataset

Next, characterize the dataset to extract key features. This step may take some time depending on your hardware:


```bash
./characterize.sh $DATASET $TRAIN_SPLIT $TEST_SPLIT
```

### Step 4: Predict Model Ranking

After characterizing the dataset, you can predict the which of the various pre-trained models is more suited to be fine-tuned on the selected dataset. Run the following command to use the pre-trained network to rank the models:

```bash
./predict_rank.sh emnlp24 $DATASET $TRAIN_SPLIT
```

The output will be in the following format:

|Rank|Score|Model|
|----|-----|-----|
|1|0.2368|microsoft/mpnet-base|
|2|0.1944|openai-community/gpt2|
|3|0.1578|facebook/opt-1.3b|
|4|0.0797|xlnet/xlnet-base-cased|
|5|0.0561|studio-ousia/luke-base|
|...|...|...|


### Step 5: Fine-Tune the Top Models

Once you've identified the top models, you can fine-tune them for your specific dataset. Keep in mind that fine-tuning is a resource-intensive process, so it's important to consider your available computational resources and budget. While we provide a simple `finetune.sh` script for this, you may need to customize it further depending on your specific task or additional requirements.


For example, to fine-tune the `microsoft/mpnet-base` model, run the following command:


```bash
./finetune.sh -d "$DATASET" -m "microsoft/mpnet-base" \
    --train-split="$TRAIN_SPLIT" \
    --test-split="$TEST_SPLIT"
```

To improve your chances of finding the optimal model, we recommend fine-tuning the top five models based on PsyMatrix’s predictions. This approach typically provides a high probability (around 80%) of identifying the best model, or at least one that performs within 93% of the optimal model's performance.

Here are the commands to fine-tune the next few recommended models for the selected dataset:


```bash
./finetune.sh -d "$DATASET" -m "openai-community/gpt2" \
    --train-split="$TRAIN_SPLIT" \
    --test-split="$TEST_SPLIT"

./finetune.sh -d "$DATASET" -m "facebook/opt-1.3b" \
    --train-split="$TRAIN_SPLIT" \
    --test-split="$TEST_SPLIT"

./finetune.sh -d "$DATASET" -m "xlnet/xlnet-base-cased" \
    --train-split="$TRAIN_SPLIT" \
    --test-split="$TEST_SPLIT"

./finetune.sh -d "$DATASET" -m "studio-ousia/luke-base" \
    --train-split="$TRAIN_SPLIT" \
    --test-split="$TEST_SPLIT"
```