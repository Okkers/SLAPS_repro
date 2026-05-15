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
    parser.add_argument('--model_id', type = str, required = True, default = 'test', help = 'model id')
    parser.add_argument('--model', type = str, required = True, default = "SLAPS", help = "model name")



    ### MODEL ARGS
    ### MAIN MODEL
    parser.add_argument('--input_dim', type = int, default = 5, help = "input dim")
    parser.add_argument('--hidden_dim', type = int, default = 25, help = "'hidden dim")
    parser.add_argument('--output_dim', type = int, default = 5, help = "out dim")
    parser.add_argument('--is_discrete', type = bool, default = True, help = 'data type (int, continuous)')
    parser.add_argument('--r', type = float, default = 0.25, help = 'hyperparam r')
    parser.add_argument('--eta', type = float, default = 0.5, help = "hyperparam eta")
    parser.add_argument('--noise_type', type = str, default = "zero", help = "noise type for continuous data")
    parser.add_argument('--generator', type = str, default = "MLP", help = "Type of Generator used")
    

    ### GENERATOR ARGS
    parser.add_argument('--gen_input_dim', type = int, default = 10, help = "Generator input dimension")
    parser.add_argument('--gen_layers_size', type = int, default = 10, help = "No. of layers in Generator")
    parser.add_argument('--gen_k', type = int, default = 10, help = "k param for Generator")


    parser.add_argument('--use_gpu', type = bool, default = True, help = "use gpu")
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
        exp.train(setting)

        print("TESTING START: {}".format(setting))
        exp.test(setting)

        torch.cuda.empty_cache()

    else:
        exp = Exp(args)
        # TESTING
        # ------
        # PLACHOLDER BUGFIXING FOR NOW, TO BE USED FOR TESTING + EVALUATION
        # -----
        exp.bugfix()
        

if __name__ == "__main__":
    main()