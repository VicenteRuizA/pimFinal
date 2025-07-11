import argparse
from train import train_model
from test import test_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], required=True, help="train or test the model")
    args = parser.parse_args()

    if args.mode == "train":
        train_model()
    elif args.mode == "test":
        test_model()
