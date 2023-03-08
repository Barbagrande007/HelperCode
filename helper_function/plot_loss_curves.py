# plot_loss_curves.py
import matplotlib.pyplot as plt


# Plot the validation and training curves separately
def plot_loss_curves(history, train_metric, validation_metric):
    """
    Plot function to be used with tensorflow.
    It plots the loss and metrics in 2 separate plots
    :param history: variable
        Where history is stored
    :param train_metric: parameter
        Selected metric that has been used for training
    :param validation_metric: parameter
        Selected metric used for validation
    :return:
        Left plot: Loss curves
        Right plot: Training and validation curves
    """

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    train_met = history.history[train_metric]
    val_met = history.history[validation_metric]

    epochs = range(1, (len(history.history["loss"]) + 1), 1)

    plt.subplots(1, 2, figsize=(10, 4))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_met, label=train_metric)
    plt.plot(epochs, val_met, label=validation_metric)
    plt.title(train_metric)
    plt.xlabel("Epoch")
    plt.legend()

    plt.show()
