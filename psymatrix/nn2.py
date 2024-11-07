import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold

BASE_PATH = os.path.join("data", "neural_net")


# Define the neural network model
class RankingModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes, dropout_prob=0.5):
        super().__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Dropout layer after the first hidden layer
        self.dropout1 = nn.Dropout(dropout_prob)
        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Dropout layer after the second hidden layer
        self.dropout2 = nn.Dropout(dropout_prob)
        # Output layer
        self.fc3 = nn.Linear(hidden_size, num_classes)
        # Activation function for hidden layers
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        # out = self.dropout1(out)
        out = self.relu(self.fc2(out))
        # out = self.dropout2(out)
        out = self.fc3(out)
        return torch.softmax(out, dim=1)  # Apply softmax to the output layer

    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))


def one_hot_encode(indexes, num_classes):
    one_hot = np.zeros((len(indexes), num_classes))
    one_hot[np.arange(len(indexes)), indexes] = 1
    return torch.tensor(one_hot, dtype=torch.float32)


def eval_topk_accuracy(y_pred_rank, y_real_rank, topk=3):

    accuracy = np.zeros(y_pred_rank.shape[0])

    for i in range(y_pred_rank.shape[0]):
        if int(y_real_rank[i, 0]) in list(y_pred_rank[i, :topk]):
            accuracy[i] = 1

    return accuracy.mean()


def eval_random_accuracy(y_real_rank, topk=3):

    n_repeats = 1000
    accuracy = np.zeros(n_repeats)

    for repeat in range(n_repeats):
        y_pred_rank = np.zeros(y_real_rank.shape, np.int64)
        for i in range(y_real_rank.shape[0]):
            y_pred_rank[i] = np.random.permutation(y_real_rank[i])
        topk_accuracy = eval_topk_accuracy(y_pred_rank, y_real_rank, topk=topk)
        accuracy[repeat] = topk_accuracy

    return accuracy.mean()


def eval_accuracy_error(y_pred, y_real, topk=1):

    min_errors = np.zeros(y_pred.shape[0])

    for i in range(y_pred.shape[0]):
        errors = np.zeros(topk)

        correct = y_real[i].max()
        y_pred_rank = y_pred[i].argsort(descending=True)

        for j in range(0, topk):
            predicted = y_real[i][y_pred_rank[j]]
            errors[j] = (correct - predicted) ** 2

        # Select the minimum error
        min_errors[i] = errors.min()

    # topk mean square error
    return min_errors.mean()


def train(
    X_tensor,
    y_tensor,
    k_folds,
    hidden_size=256,
    num_epochs=2500,
    save_model_fname=None,
    early_stopping_patience=30,
):

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    topk_accuracies = np.zeros((k_folds, y_tensor.shape[1]))
    topk_mse = np.zeros((k_folds, y_tensor.shape[1]))
    y_pred_rank_all = []
    y_real_rank_all = []

    results = {}

    for fold, (train_ids, test_ids) in enumerate(kfold.split(X_tensor, y_tensor)):
        fold_key = f"fold_{fold+1}"

        results[fold_key] = {
            "train_ids": train_ids.tolist(),
            "test_ids": test_ids.tolist(),
            "prob_optimal_topk": [],
        }

        X_train_tensor = X_tensor[train_ids]
        X_test_tensor = X_tensor[test_ids]

        y_train_tensor = y_tensor[train_ids]
        y_test_tensor = y_tensor[test_ids]

        # Convert to one-hot encoding
        y_train_tensor = one_hot_encode(
            y_train_tensor.argmax(1), num_classes=y_tensor.shape[1]
        )
        y_test_tensor = one_hot_encode(
            y_test_tensor.argmax(1), num_classes=y_tensor.shape[1]
        )

        # Reset the model
        model = RankingModel(
            input_size=X_tensor.shape[1],
            hidden_size=hidden_size,
            num_classes=y_tensor.shape[1],
        )

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        best_test_loss = np.inf
        best_model_state_dict = model.state_dict().copy()
        n_epochs_no_improve = 0

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()

            # Forward pass
            outputs = model(X_train_tensor)
            train_loss = criterion(outputs, y_train_tensor)

            # Backward pass and optimization
            train_loss.backward()
            optimizer.step()

            # Evaluate the model
            model.eval()

            with torch.no_grad():
                y_pred = model(X_test_tensor)
                test_loss = criterion(y_pred, y_test_tensor)

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                n_epochs_no_improve = 0
                best_model_state_dict = model.state_dict().copy()
            else:
                n_epochs_no_improve += 1

            # if epoch % 10 == 0:
            #     print(epoch, train_loss.item(), test_loss.item())

            if n_epochs_no_improve >= early_stopping_patience:
                break

        # Check top-k accuracy
        model.load_state_dict(best_model_state_dict)
        model.eval()

        with torch.no_grad():
            y_pred = model(X_test_tensor)

        y_pred_rank = y_pred.argsort(dim=1, descending=True)
        y_real_rank = y_tensor[test_ids].argsort(dim=1, descending=True)

        results[fold_key]["y_pred_rank"] = y_pred_rank.numpy().tolist()
        results[fold_key]["y_real_rank"] = y_real_rank.numpy().tolist()

        y_pred_rank_all.append(y_pred_rank)
        y_real_rank_all.append(y_real_rank)

        # Top-k accuracy
        for i in range(topk_accuracies.shape[1]):
            topk_accuracies[fold][i] = eval_topk_accuracy(
                y_pred_rank, y_real_rank, topk=i + 1
            )
            topk_mse[fold][i] = eval_accuracy_error(y_pred, y_test_tensor, topk=i + 1)

            results[fold_key][f"prob_optimal_topk"].append(topk_accuracies[fold][i])

        if save_model_fname:
            # Ensure the directory exists
            if not os.path.exists(BASE_PATH):
                os.makedirs(BASE_PATH)

            torch.save(
                model.state_dict(),
                os.path.join(BASE_PATH, f"fold_{fold}_{save_model_fname}"),
            )

    return (
        y_pred_rank_all,
        y_real_rank_all,
        topk_accuracies.mean(0),
        topk_accuracies.std(0),
        topk_mse.mean(0),
        topk_mse.std(0),
        results,
    )


def train_nnet(X_tensor, y_tensor, num_repeats, k_folds=10, hidden_size=32):

    accuracy_mean_vec = []
    accuracy_std_vec = []
    mse_mean_vec = []
    mse_std_vec = []
    y_pred_all_vec = []
    y_real_all_vec = []

    for r in range(num_repeats):
        fout = f"repeat_{r}_nnet.pth"
        (
            y_real_rank_all,
            y_pred_rank_all,
            accuracy_mean,
            accuracy_std,
            mse_mean,
            mse_std,
            results,
        ) = train(
            X_tensor,
            y_tensor,
            k_folds=k_folds,
            hidden_size=hidden_size,
            save_model_fname=fout,
        )

        accuracy_mean_vec.append(accuracy_mean)
        accuracy_std_vec.append(accuracy_std)
        mse_mean_vec.append(mse_mean)
        mse_std_vec.append(mse_std)
        y_pred_all_vec.append(y_pred_rank_all)
        y_real_all_vec.append(y_real_rank_all)

        with open(f"data/neural_net/repeat_{r}_nnet.json", "w") as f:
            f.write(json.dumps(results, indent=2))

    # Calculate the area under the curve
    accuracy_mean_vec = np.array(accuracy_mean_vec).mean(axis=0)
    accuracy_std_vec = np.array(accuracy_std_vec).mean(axis=0)

    max_auc = y_tensor.shape[1]

    norm_auc = np.trapz(accuracy_mean_vec, axis=0) / max_auc

    return y_pred_all_vec, y_real_all_vec, norm_auc, accuracy_mean_vec, accuracy_std_vec
