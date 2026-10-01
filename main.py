from minitorch import (
    autodiff_demo,
    train_xor,
    train_mnist,
    compare_with_pytorch
)
if __name__ == "__main__":
    autodiff_demo()
    train_xor()
    choice = input("\nTrain on MNIST?: ")
    if choice.lower() == "y":
        train_mnist()

    choice = input("\nCompare with PyTorch?: ")
    if choice.lower() == "y":
        compare_with_pytorch()