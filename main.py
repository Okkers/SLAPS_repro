import argparse 
import os 
import torch 
from exp.exp_main import Exp_Main
import random 
import numpy as np 
from utils.metrics import *

def main():
    fix_seed = 42
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description="Reproduction of SLAPS")

    parser.add_argument('--is_training', type = int, required = True, default = 1, help = "status")
    parser.add_argument('--model_id', type = str, required = True, default = 'test', help = 'model id')
    parser.add_argument('--model', type = str, required = True, default = "SLAPS", help = "model name")
    parser.add_argument('--train_epochs', type = int, required = True, default = 2500, help = "no. of epochs to train for")
    parser.add_argument('--dataset', type = str, required = True, default = "wine", help = "name of dataset to use")   
    parser.add_argument('--mnist_training_size', type = int, default=1000, help='Size of training size for MNIST data') 
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--itr', type = int, default = 1, help = "no. of experiments to run")
    parser.add_argument('--patience', type = int, default = 10, help = "Early Stopping patience param")

    ### MODEL ARGS
    ### MAIN MODEL
    parser.add_argument('--input_dim', type = int, default = 5, help = "input dim")
    parser.add_argument('--hidden_dim', type = int, default = 25, help = "'hidden dim")
    parser.add_argument('--output_dim', type = int, default = 5, help = "out dim")
    parser.add_argument('--is_discrete', action = 'store_true', help = 'data type (int, continuous)')
    parser.add_argument('--r', type = float, default = 0.25, help = 'hyperparam r')
    parser.add_argument('--eta', type = float, default = 0.5, help = "hyperparam eta")
    parser.add_argument('--noise_type', type = str, default = "zero", help = "noise type for continuous data")
    parser.add_argument('--generator', type = str, default = "MLP", help = "Type of Generator used")

    ### GENERATOR ARGS
    parser.add_argument('--gen_input_dim', type = int, default = 10, help = "Generator input dimension")
    parser.add_argument('--gen_layers_size', type = int, default = 10, help = "No. of layers in Generator")
    parser.add_argument('--gen_k', type = int, default = 10, help = "k param for Generator")


    parser.add_argument('--use_gpu', type = bool, default = True, help = "use gpu")
    parser.add_argument('--gpu', type = int, default = 0, help = "gpu")
    parser.add_argument('--use_multi_gpu', action = 'store_true', help = "use multiple gpus")
    parser.add_argument('--devices', type = str, default = "0", help = "device ids for multi gpu")

    ### CLASSIFIER
    parser.add_argument('--lr_c', type=float, default=0.01, help="Learning rate for the classifier GNN_C")
    parser.add_argument('--dropout_c', type=float, default=0.5, help="Dropout probability for the classifier")

    ### DENOISING AUTOENCODER
    parser.add_argument('--dropout_DAE', type=float, default=0.5, help="Dropout probability for the DAE")
    parser.add_argument('--lr_DAE', type=float, default=0.001, help="Learning rate for the denoising autoencoder GNN_DAE")

    ### LAMBDA FOR JOINT LOSS
    parser.add_argument('--lambda_val', dest='lambda_val', type=float, default=0.1,
                        help="Controls the relative importance of the two losses")
    
    ### SELF-TRAINING
    parser.add_argument('--self_training', action='store_true', help="Include self-training step or not.")
    parser.add_argument('--self_training_zeta', type=int, default=50)

    ### ADA EDGE
    parser.add_argument('--ada_edge', action='store_true', help="Run ada edge.")
    parser.add_argument('--conf_threshold', type=float, default=0.9, help="Threshold for probability per class.")
    parser.add_argument('--max_add', type=int, default=10, help="Max edges added.")
    parser.add_argument('--max_remove', type=int, default=10, help="Max edges removed.")
    parser.add_argument('--max_iter', type=int, default=10, help="Max iterations.")
    parser.add_argument('--order', type=str, default="add", help="Start with add or remove edges.")

    parser.add_argument('--weight_decay_c', type=float, default=0, help="Weight decay for classifier opt")
    parser.add_argument('--weight_decay_dae', type=float, default=0, help="Weight decay for dae")

    parser.add_argument('--hidden_dim_dae', type=int, default=32, help="hidden dimension for dae")

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    print("Args:")
    print(args)

    Exp = Exp_Main

    if args.is_training:
        all_accuracies = []
        for ii in range(args.itr):
            # current_seed = fix_seed + ii
            # random.seed(current_seed)
            # torch.manual_seed(current_seed)
            # np.random.seed(current_seed)

            setting =  '{}_{}_{}_{}'.format(args.model_id, args.model, args.dataset, ii)

            exp = Exp(args) 
            print("TRAINING START: {}".format(setting))
            exp.train(setting)

            print("TESTING START: {}".format(setting))
            if args.ada_edge:
                acc = exp.test_ada_edge(setting)
            else:
                acc = exp.test(setting)
            all_accuracies.append(acc)

            torch.cuda.empty_cache()

        # calculate and save the std and mean for all 10 runs
        aggregate_metrics(all_accuracies, args)

    else:
        exp = Exp(args)
        # TESTING
        # ------
        # PLACEHOLDER BUGFIXING FOR NOW, TO BE USED FOR TESTING + EVALUATION
        # -----
        exp.bugfix()
        

if __name__ == "__main__":
    main()