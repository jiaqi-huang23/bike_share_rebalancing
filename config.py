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
main_arg.add_argument("--frac", type=float,
                    default=1, help="Fraction of axis data to use"
                    )
main_arg.add_argument("--test_size", type=float,
                    default=0.1, help="The proportion of test set size in the whole data set."
                    )
main_arg.add_argument("--data_path", type=str,
                    default='./data/', help="The path to the dataset."
                    )
def get_config():
    config, unparsed = parser.parse_known_args()

    return config, unparsed


def print_usage():
    parser.print_usage()

#
# config.py ends here
