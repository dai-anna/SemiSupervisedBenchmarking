import torch


def set_device():
    if torch.cuda.is_available():  # check if NVIDIA GPU is available
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # check if Apple's Metal is available
        device = torch.device("mps")
        print(
            f"Checking pytorch is built with mps activated: {torch.backends.mps.is_built()}"
        )
    else:
        device = torch.device("cpu")

    if device == torch.device("cuda") or device == torch.device("mps"):
        print(f"Running on {device} GPU...")
    else:
        print("Running on CPU...")

    return device
