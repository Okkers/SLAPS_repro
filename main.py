import argparse 
import os 
import torch 
from exp.exp_main import Exp_Main
import random 
import numpy as np 

def main():
    fix_seed = 42
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description="Reproduction of SLAPS")

    parser.add_argument('--is_training', type = int, required = True, default = 1, help = "status")
    
    
    
    parser.add_argument('--use_gpu', type = bool, default = True, default = 1, help = "use gpu")
    # ...
    # ...
    # ...

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    print("Args:")
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            setting = '{}'.format(args.use_gpu)

        exp = Exp(args) 
        print("TRAINING START: {}".format(setting))
        args.train(setting)

        print("TESTING START: {}".format(setting))
        exp.test(setting)

        torch.cuda.empty_cache()

    else:
        # TESTING
        pass

if __name__ == "__main__":
    main()