import torch as th
import argparse
from sktime.datasets import load_gunpoint
from dataset import toarray, MyDataset
from Module import FCN 
from trainer import train_mixup_model_epoch
from plt import plot_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gunpoint', help='The dataset name')
    parser.add_argument('--epochs', type=int, default=100, help='Epoch number')
    parser.add_argument('--alpha', type=float, default=1.0, help='The alpha')
    parser.add_argument('--device', type=str, default='cpu' ,help='cpu for CPU and cuda for NVIDIA GPU')
    
    args = parser.parse_args()

    print("Dataset:", args.dataset)
    print("Arguments:", str(args))

    
    device = args.device
    epochs = args.epochs
    alpha = args.alpha

    from scipy.io.arff import loadarff
    train_data = loadarff(f'D:\新建文件夹 (2)\课题组\复刻\Time Series Review Task\weihao\PRL_22\code\Mixup\data\Multivariate_arff\BasicMotions\BasicMotions_TRAIN.arff')[0]
    test_data = loadarff(f'D:\新建文件夹 (2)\课题组\复刻\Time Series Review Task\weihao\PRL_22\code\Mixup\data\Multivariate_arff\BasicMotions\BasicMotions_TEST.arff')[0]

    x_tr = [list(i[0]) for i in train_data]
    x_tr = [[list(y) for y in x] for x in x_tr]
    y_tr = [i[1] for i in train_data]

    new = []
    for i in y_tr:
        n = 0
        for j in list(set(y_tr)):
            if i == j:
                new.append(n)
            else:
                n += 1
    y_tr = new.copy()

    x_te = [list(i[0]) for i in test_data]
    x_te = [[list(y) for y in x] for x in x_te]
    y_te = [i[1] for i in test_data]

    new = []
    for i in y_te:
        n = 0
        for j in list(set(y_te)):
            if i == j:
                new.append(n)
            else:
                n += 1
    y_te = new.copy()

    training_set = MyDataset(x_tr, y_tr)
    test_set = MyDataset(x_te, y_te)

    '''
    device = 'cpu'
    epochs = 5
    alpha = 1.0
    x_tr, y_tr = load_gunpoint(split='train', return_X_y=True)
    x_te, y_te = load_gunpoint(split='test', return_X_y=True)
    print(type(y_te))
    training_set = MyDataset(x_tr, y_tr)
    test_set = MyDataset(x_te, y_te)
    '''


    model = FCN(training_set.x.shape[1]).to(device)
    optimizer = th.optim.Adam(model.parameters())
    LossListM, AccListM = train_mixup_model_epoch(model, training_set, test_set,
                                              optimizer, alpha, epochs)

    #print(LossListM)

    print(f"Score for alpha = {alpha}: {AccListM[-1]}")
    plot_results(LossListM, AccListM)