import os

import mlflow
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms


DEVICE = torch.device("cpu")
BATCHSIZE = 128
CLASSES = 10
DIR = os.getcwd()
EPOCHS = 10
LOG_INTERVAL = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10

class Net(nn.Module):
  def __init__(self, trial):
    super().__init__()
    n_channels = trial.suggest_int('n_channels', 6, 12)
    n_neurons = trial.suggest_int('n_neurons', 100, 500)
    self.act_func = trial.suggest_categorical('act_func', ["ReLU", "ELU"])

    self.conv1 = nn.Conv2d(1, n_channels, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(n_channels, 16, 5)
    self.fc1 = nn.Linear(16 * 4 * 4, n_neurons)
    self.fc2 = nn.Linear(n_neurons, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    if self.act_func == "ReLU":
      func = F.relu
    else:
      func = F.elu

    x = self.pool(func(self.conv1(x)))
    x = self.pool(func(self.conv2(x)))
    x = x.view(-1, 16 * 4 * 4)
    x = func(self.fc1(x))
    x = func(self.fc2(x))
    x = self.fc3(x)
    return F.log_softmax(x, dim=1)


def get_mnist():
    # Load MNIST dataset.
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DIR, train=True, download=True, transform=transforms.ToTensor()),
        batch_size=BATCHSIZE,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DIR, train=False, transform=transforms.ToTensor()),
        batch_size=BATCHSIZE,
        shuffle=True,
    )

    return train_loader, valid_loader


def mlflow_callback(study, trial):
    trial_value = trial.value if trial.value is not None else float("nan")
    with mlflow.start_run(run_name=study.study_name):
        mlflow.log_params(trial.params)
        mlflow.log_metrics({"accuracy": trial_value})


def objective(trial):

    # Generate the model.
    model = Net(trial).to(DEVICE)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])

    if optimizer_name == "Adam":
      lr = trial.suggest_float("adam_lr", 1e-10, 1e-3, log=True)
    elif optimizer_name == "SGD":
      lr = trial.suggest_float("sgd_lr", 1e-5, 1e-1)
    else:
      lr = trial.suggest_float("rmsprop_lr", 1e-4, 1e-1)

    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the MNIST dataset.
    train_loader, valid_loader = get_mnist()

    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            #data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                    break
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600, callbacks=[mlflow_callback])

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))