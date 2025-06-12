import matplotlib.pyplot as plt

def PlotResultCurves(result):
    """
    result: Dict containing:
        train loss: 
        train acc:
        train miou:
        train dice:
        val loss: 
        val acc:
        val miou:
        val dice:
        """
    epochs = list(range(len(result["train loss"])))
    fig, ax = plt.figure(1, 4, figsize=(24, 6))

    ax[0].plot(epochs, result["train loss"], label="Train")
    ax[0].plot(epochs, result["val loss"], label="Validation")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epochs")
    ax[0].legend()

    ax[1].plot(epochs, result["train acc"], label="Train")
    ax[1].plot(epochs, result["val acc"], label="Validation")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Epochs")
    ax[1].legend()

    ax[2].plot(epochs, result["train miou"], label="Train")
    ax[2].plot(epochs, result["val miou"], label="Validation")
    ax[2].set_title("MeanIoU")
    ax[2].set_xlabel("Epochs")
    ax[2].legend()

    ax[3].plot(epochs, result["train dice"], label="Train")
    ax[3].plot(epochs, result["val dice"], label="Validation")
    ax[3].set_title("MeanDICE")
    ax[3].set_xlabel("Epochs")
    ax[3].legend()

    plt.tight_layout()
    plt.show()

def PlotInference():
    pass