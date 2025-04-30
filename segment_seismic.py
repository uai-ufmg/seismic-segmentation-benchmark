import torch
from argparse import ArgumentParser
from datetime import datetime
from tqdm import tqdm
from math import ceil
from sklearn.model_selection import KFold

import core
from core.loader.data_loader import *
from core.models import load_empty_model
from core.metrics import RunningMetrics
from core.utils import store_results


def train_test_split(args, dataset):
    kf = KFold(n_splits=5, shuffle=False)

    if args.swap_train_test:
        # 20% train / 80% test
        splits = [
            (test_idx.tolist(), train_idx.tolist()) for train_idx, test_idx in kf.split(dataset)
        ]
    else:
        # 80% train / 20% test
        splits = [
            (train_idx.tolist(), test_idx.tolist()) for train_idx, test_idx in kf.split(dataset)
        ]
    
    if args.cross_validation:
        return splits
    else:
        return [splits[args.test_fold - 1]]


def train(args, dataset, device, criterion, n_classes, indices, model):
    if args.model_path is None:
        print('\nCreating model...\n')
    else:
        print(f'\nResuming training from stored model: {args.model_path}\n')

    print('Architecture:   ', args.architecture.upper())
    print('Optimizer:      ', args.optimizer)
    print('Device:         ', device)
    print('Loss function:  ', args.loss_function)
    print('Learning rate:  ', args.learning_rate)
    print('Batch size:     ', args.batch_size)
    print('N. of epochs:   ', args.n_epochs)

    print('\nNumber of train examples:', len(indices))

    print('\nWeighted loss ENABLED' if args.weighted_loss else 'Weighted loss DISABLED')
    print(f'Training with {"INLINES" if args.orientation == "in" else "CROSSLINES"}')

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    # Defining the optimizer
    optimizer_map = {
        'adam': torch.optim.Adam,
        'sgd' : torch.optim.SGD
    }

    OptimizerClass = optimizer_map[args.optimizer]
    optimizer = OptimizerClass(
        params=model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Initializing metrics
    train_metrics = RunningMetrics(n_classes=n_classes, bf1_threshold=2)

    # Storing the loss for each epoch
    train_loss_list = []

    for epoch in range(args.n_epochs):
        # Training phase
        model.train()
        train_loss = 0

        print(datetime.now().strftime('\n%Y/%m/%d %H:%M:%S'))
        print(f'Training on epoch {epoch + 1}/{args.n_epochs}\n')

        for images, labels in tqdm(train_loader, ascii=' >='):
            optimizer.zero_grad()

            images = images.type(torch.FloatTensor).to(device)
            labels = labels.type(torch.FloatTensor).to(device)

            outputs = model(images)

            # Updating the running metrics
            train_metrics.update(images=outputs, targets=labels)

            # Computing the loss and updating weights
            loss = criterion(images=outputs, targets=labels.long())
            train_loss += loss.item()
            loss.backward()

            optimizer.step()
        
        train_loss = train_loss / ceil((len(train_loader) / args.batch_size))
        train_scores = train_metrics.get_scores()

        print(f'Train loss: {train_loss}')
        print(f'Train mIoU: {train_scores["mean_iou"]}')

        train_loss_list.append(train_loss)
        train_metrics.reset()

    results = {
        'train_scores' : train_scores,
        'train_losses' : train_loss_list,
    }
    
    return model, results


def test(args, dataset, device, criterion, n_classes, indices, model):
    print('\nTesting the model...\n')

    print('Architecture:   ', args.architecture.upper())
    print('Device:         ', device)
    print('Loss function:  ', args.loss_function)
    print('Batch size:     ', args.batch_size)

    print(f'\nNumber of test examples: {len(indices)}')

    print('\nWeighted loss ENABLED' if args.weighted_loss else 'Weighted loss DISABLED')
    print(f'Testing with {"INLINES" if args.orientation == "in" else "CROSSLINES"}')
    
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # Initializing metrics
    test_metrics  = RunningMetrics(n_classes=n_classes, bf1_threshold=2)

    # Storing the loss for each epoch
    test_loss_list  = []
    
    preds = {}
    slice_idx = indices[0]

    # Testing phase
    with torch.no_grad():
        model.eval()
        test_loss = 0

        print(datetime.now().strftime('\n%Y/%m/%d %H:%M:%S'))

        for images, labels in tqdm(test_loader, ascii=' >='):
            images = images.type(torch.FloatTensor).to(device)
            labels = labels.type(torch.FloatTensor).to(device)

            outputs = model(images)

            # Iterating over the batch
            for pred in outputs:
                preds[slice_idx] = pred
                slice_idx += 1

            # Updating the running metrics
            test_metrics.update(images=outputs, targets=labels)

            # Computing the loss
            loss = criterion(images=outputs, targets=labels.long())
            test_loss += loss.item()
        
        test_loss = test_loss / ceil((len(test_loader) / args.batch_size))
        test_scores = test_metrics.get_scores()

        print(f'Test loss: {test_loss}')
        print(f'Test mIoU: {test_scores["mean_iou"]}')

        test_loss_list.append(test_loss)
        test_metrics.reset()
    
    results = {
        'test_scores'  : test_scores,
        'test_losses'  : test_loss_list,
    }

    return preds, results


def run(args):
    print('')
    print('Data path:    ', args.data_path)
    print('Labels path:  ', args.labels_path)
    print('Results path: ', args.results_path)
    print('')

    print('Loading dataset...')

    dataset = SeismicDataset(
        data_path=args.data_path,
        labels_path=args.labels_path,
        orientation=args.orientation,
        compute_weights=args.weighted_loss,
        faulty_slices_list=args.faulty_slices_list
    )

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.weighted_loss:
        class_weights = torch.tensor(dataset.get_class_weights(), device=device, requires_grad=False)
        class_weights = class_weights.float()
    else:
        class_weights = None

    loss_map = {
        'cel': ('CrossEntropyLoss', {'reduction': 'sum', 'weight': class_weights})
    }

    # Defining the loss function
    loss_name, loss_args = loss_map[args.loss_function]
    criterion = getattr(core.loss, loss_name)(**loss_args)

    # Splitting the data into train and test
    splits = train_test_split(args, dataset)

    results = []

    for fold_number, (train_indices, test_indices) in enumerate(splits):
        if args.cross_validation:
            print(f'\n================= FOLD {fold_number + 1}/5 =================')
        else:
            print(f'\n================= FOLD {args.test_fold} =================')
        
        train_indices, test_indices = splits[fold_number]
        
        train_set = torch.utils.data.Subset(dataset, train_indices)
        test_set = torch.utils.data.Subset(dataset, test_indices)

        model = load_empty_model(args.architecture, dataset.get_n_classes())
        model = model.to(device)

        # Loading previously stored weights if provided
        if args.model_path is not None:
            if not os.path.isfile(args.model_path):
                raise FileNotFoundError(f'No such file or directory for stored model: {args.model_path}')
        
            model.load_state_dict(torch.load(args.model_path))

        if args.train:
            # Training the model
            model, train_results = train(
                args,
                dataset=train_set,
                device=device,
                criterion=criterion,
                n_classes=dataset.get_n_classes(),
                indices=train_indices,
                model=model
            )
        else:
            print('\nTraining is OFF.')

            if args.model_path is None:
                raise FileNotFoundError('No directory for stored model was provided.')

            train_results = {}

        # Testing the model
        preds, test_results = test(
            args,
            dataset=test_set,
            device=device,
            criterion=criterion,
            n_classes=dataset.get_n_classes(),
            indices=test_indices,
            model=model
        )

        results.append({
            'model': model,
            'preds': preds.copy(),
            'train_indices': train_indices,
            'test_indices': test_indices,
            **train_results,
            **test_results
        })

    store_results(args, results)


if __name__ == '__main__':
    parser = ArgumentParser(description='Hyperparameters')

    parser.add_argument('-a', '--architecture',
        dest='architecture',
        type=str,
        help='Architecture to use [segnet, unet, deconvnet]',
        choices=['segnet', 'unet', 'deconvnet']
    )
    parser.add_argument('-d', '--data-path',
        dest='data_path',
        type=str,
        help='Path to the data file in numpy or segy format'
    )
    parser.add_argument('-l', '--labels-path',
        dest='labels_path',
        type=str,
        help='Path to the labels file in numpy format'
    )
    parser.add_argument('-t', '--train',
        dest='train',
        action='store_true',
        default=False,
        help='Whether to train a model or to simply test from a stored model'
    )
    parser.add_argument('-b', '--batch-size',
        dest='batch_size',
        type=int,
        default=16,
        help='Batch Size'
    )
    parser.add_argument('-D', '--device',
        dest='device',
        type=str,
        default='cuda:0',
        help='Device to train on [cuda:n]'
    )
    parser.add_argument('-v', '--cross-validation',
        dest='cross_validation',
        action='store_true',
        default=False,
        help='Whether to use 5-fold cross validation'
    )
    parser.add_argument('-f', '--test-fold',
        dest='test_fold',
        type=int,
        default=1,
        help='Which fold to use for testing',
        choices=[i for i in range(1, 6)]
    )
    parser.add_argument('-L', '--loss-function',
        dest='loss_function',
        type=str,
        default='cel',
        help='Loss function to use [cel (Cross_Entropy Loss)]',
        choices=['cel']
    )
    parser.add_argument('-o', '--optimizer',
        dest='optimizer',
        type=str,
        default='adam',
        help='Optimizer to use [adam, sgd (Stochastic Gradient Descent)]',
        choices=['adam', 'sgd']
    )
    parser.add_argument('-r', '--learning-rate',
        dest='learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument('-w', '--weight-decay',
        dest='weight_decay',
        type=float,
        default=1e-5,
        help='L2 regularization. Value 0 indicates no weight decay'
    )
    parser.add_argument('-W', '--weighted-loss',
        dest='weighted_loss',
        action='store_true',
        default=False,
        help='Whether to use class weights in the loss function'
    )
    parser.add_argument('-e', '--n-epochs',
        dest='n_epochs',
        type=int,
        default=30,
        help='Number of epochs'
    )
    parser.add_argument('-O', '--orientation',
        dest='orientation',
        type=str,
        default='in',
        help='Whether the model should be trained using inlines or crosslines',
        choices=['in', 'cross']
    )
    parser.add_argument('-F', '--faulty-slices-list',
        dest='faulty_slices_list',
        type=str,
        default=None,
        help='Path to a json file containing a list of faulty slices to remove'
    )
    parser.add_argument('-s', '--swap-train-test',
        dest='swap_train_test',
        action='store_true',
        default=False,
        help='Whether to swap the train and test sets to train on less data'
    )
    parser.add_argument('-m', '--model-path',
        dest='model_path',
        type=str,
        default=None,
        help='Directory for loading saved model'
    )
    parser.add_argument('-p', '--results-path',
        dest='results_path',
        type=str,
        default=os.path.join(os.getcwd(), 'results'),
        help='Directory for storing execution results'
    )

    args = parser.parse_args(args=None)
    run(args)
