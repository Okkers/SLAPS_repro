### THIS IS BOILER PLATE FOR DATA LOADING AND CREATION
from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_A

data_dict = {

}

def data_provider(args, flag):
    Data = data_dict[args.data]
    
    if flag == "test":
        ### FLAGS FOR TESTING

        ### e.g. batch_size = args.batch_size
        ### e.g. shuffle_data = False

        pass
        
    elif flag == "pred":
        ### FLAGS FOR PREDICTING

        ### e.g. batch_size = 1
        ### e.g. shuffle_falg = True
        pass

    else:
        ### FLAGS FOR TRAINING
        pass


    # data_set = Data(
    #  root_path = args.root_path,
    #  size = [args.seq_len]
    #)
    #

    # data_loader = DataLoader(
    # data_set,   
    # batch_size = batch_size   
    #    )
    return # data_set, data_loader
