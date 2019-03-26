# Set our hyper-parameters using command line

import argparse
# ----------------------------------------
# Global variables within this script
arg_lists = []
parser = argparse.ArgumentParser()


# ----------------------------------------
# Some nice macros to be used for arparse
def str2bool(v):
    return v.lower() in ("true", "1")


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# ----------------------------------------
# Arguments for the main program
main_arg = add_argument_group("Main")


main_arg.add_argument("--mode", type=str,
                      default="train",
                      choices=["train", "test"],
                      help="Run mode")

# ----------------------------------------
# Arguments for training
train_arg = add_argument_group("Training")


train_arg.add_argument("--data_dir", type=str,
                       default="/data",
                       help="Directory with CIFAR10 data")

train_arg.add_argument("--learning_rate", type=float,
                       default=1e-4,
                       help="Learning rate (gradient step size)")

train_arg.add_argument("--batch_size", type=int,
                       default=100,
                       help="Size of each training batch")

train_arg.add_argument("--num_epoch", type=int,
                       default=100,
                       help="Number of epochs to train")

train_arg.add_argument("--val_intv", type=int,
                       default=1000,
                       help="Validation interval")

train_arg.add_argument("--rep_intv", type=int,
                       default=1000,
                       help="Report interval")

train_arg.add_argument("--log_dir", type=str,
                       default="./logs",
                       help="Directory to save logs and current model")

train_arg.add_argument("--save_dir", type=str,
                       default="./save",
                       help="Directory to save the best model")

train_arg.add_argument("--resume", type=str2bool,
                       default=True,
                       help="Whether to resume training from existing checkpoint")
# ----------------------------------------
# Arguments for model
model_arg = add_argument_group("Model")

model_arg.add_argument("--model", type=str,
                       default="Lasso",
                       choices=["LogisticRegression", "SGDClassifier", "Lasso"],
                       help="Type of linear model to be used")

model_arg.add_argument("--normalize", type=str2bool,
                       default=True,
                       help="Whether to normalize with mean/std or not")

model_arg.add_argument("--l2_reg", type=float,
                       default=1e-4,
                       help="L2 Regularization strength")

model_arg.add_argument("--num_class", type=int,
                       default=3,
                       help="Number of classes in the dataset")

model_arg.add_argument("--activ_type", type=str,
                       default="relu",
                       choices=["relu", "tanh"],
                       help="Activation type")


def get_config():
    config, unparsed = parser.parse_known_args()

    return config, unparsed


def print_usage():
    parser.print_usage()

#
# config.py ends here
