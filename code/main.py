# Main Running File
import torch as th
import argparse
import sys
from sktime.datasets import load_gunpoint
from dataset import toarray, MyDataset
from Module import FCN 
from trainer import train_mixup_model_epoch
from plt import plot_results
from utils import set_global_seed
from dataset_load import load_ucr_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasource', type=str, default='ucr', required=True, help='ucr or uea datasets')
    parser.add_argument('--dataset', type=str, default='gunpoint', required=True, help='The dataset name')
    parser.add_argument('--epochs', type=int, default=10, help='Epoch number')
    parser.add_argument('--alpha', type=float, default=1.0, help='The alpha')
    parser.add_argument('--device', type=str, default='cpu' ,help='cpu for CPU and cuda for NVIDIA GPU')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    args = parser.parse_args()
    print("Data Source:", args.datasource.upper())
    print("Dataset:", args.dataset)
    print("Arguments:", str(args))

    data_source = args.datasource
    device = args.device
    epochs = args.epochs
    alpha = args.alpha
    filename = args.dataset
    random_seed = args.seed

    set_global_seed(random_seed)
    if data_source == 'ucr':
        x_tr, y_tr = load_ucr_data(filename=filename, typename='train')
        x_te, y_te = load_ucr_data(filename=filename, typename='test')

        training_set = MyDataset(x_tr, y_tr)
        test_set = MyDataset(x_te, y_te)
    elif data_source == 'uea':
        print('Not supported yet')
        sys.exit()
    else:
        print('Please enter our supported data source')
        sys.exit()
   
    model = FCN(training_set.x.shape[1]).to(device)
    optimizer = th.optim.Adam(model.parameters())
    LossListM, AccListM = train_mixup_model_epoch(model, training_set, test_set,
                                              optimizer, alpha, epochs)

    #print(LossListM)

    print(f"Score for alpha = {alpha}: {AccListM[-1]}")
    plot_results(LossListM, AccListM)

else:
    print('Someting Wrong')










