import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

from maddpg_o.experiments.train_helper.train_helpers_eval import parse_args, train, eval

if __name__ == "__main__":
    arglist = parse_args()
    eval(arglist)