import torch
import math
import os
import time
import json
import logging
import numpy as np
import random
from torch.utils.data import DataLoader
from torchmeta.utils.data import BatchMetaDataLoader

from maml.datasets import get_benchmark_by_name
from maml.metalearners import ModelAgnosticMetaLearning

from torchsummary import summary

print("!!!!!!!!!!!!!!!!GPU!!!!!!!!!!!!!!!", torch.cuda.is_available())
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    device = torch.device('cuda' if args.use_cuda
                          and torch.cuda.is_available() else 'cpu')

    #(args.unseen)


    if (args.output_folder is not None):
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
            logging.debug('Creating folder `{0}`'.format(args.output_folder))

        folder = os.path.join(args.output_folder,
                              time.strftime('%Y-%m-%d_%H%M%S'))
        os.makedirs(folder)
        logging.debug('Creating folder `{0}`'.format(folder))

        # print('folder:', folder)

        args.folder = os.path.abspath(args.folder)
        args.model_path = os.path.abspath(os.path.join(folder, 'model.th'))
        # Save the configuration in a config.json file
        with open(os.path.join(folder, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
        logging.info('Saving configuration file in `{0}`'.format(
                     os.path.abspath(os.path.join(folder, 'config.json'))))

    meta_train_dataset, unseen_dataset, model, loss_function, seen_dataset_dict = get_benchmark_by_name(args.unseen,
                                                                      args.meta_train_bs,
                                                                      args.meta_test_bs,
                                                                      num_iterations=args.num_batches * args.batch_size)

    print('meta_train_dataset', meta_train_dataset)
    print('unseen_dataset', unseen_dataset)
    print('model', model) 
    print('loss_function', loss_function)
    print('----------------------------------------------')
    #print('benchmark', benchmark.meta_train_dataset, benchmark.meta_test_dataset)
    # meta_train_dataloader = BatchMetaDataLoader(benchmark.meta_train_dataset,
    #                                             batch_size=args.batch_size,
    #                                             shuffle=True,
    #                                             num_workers=args.num_workers,
    #                                             pin_memory=True)
    meta_train_dataloader = DataLoader(meta_train_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       drop_last=True)
    seen_dataloader_dict = {dataset_name: DataLoader(seen_dataset_dict[dataset_name], batch_size=1, shuffle=False) for dataset_name in seen_dataset_dict}

    unseen_dataloader = DataLoader(unseen_dataset,
                                     batch_size=1,
                                     shuffle=True,
                                     drop_last=True)


    meta_optimizer = torch.optim.Adam(list(model[0].parameters()) + list(model[1].parameters()), lr=args.meta_lr)

    metalearner = ModelAgnosticMetaLearning(model,
                                            meta_optimizer,
                                            first_order=args.first_order,
                                            num_adaptation_steps=args.num_steps,
                                            step_size=args.step_size,
                                            loss_function=loss_function,
                                            device=device,
                                            unseen_name=args.unseen,
                                            )

    best_value = None

    # Training loop
    epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(args.num_epochs)))

    f = open(folder + '/result.txt', 'a')
    f_t = open(folder + '/train_results.txt', 'a')

    for epoch in range(args.num_epochs):
        # print(args.num_epochs, '----', args.num_batches, '----', args.batch_size)
        # print()
        print('Epoch:', epoch)
        metalearner.curriculum_weight(metalearner.unseen_name, metalearner.max_mIoU)

        metalearner.train(meta_train_dataloader,
                          max_batches=args.num_batches,
                          verbose=args.verbose,
                          save_pth=folder,
                          desc='Training',
                          leave=False)

        metalearner.cal_max_mIoU(metalearner.unseen_name, trainingdataloader_dict=seen_dataloader_dict)
        
        with open(folder+'/result.txt', 'a') as f:
            f.write('Epoch: {}'.format(epoch))
            f.write('\n')
            
        with open(folder+'/train_results.txt', 'a') as f_t:
            f_t.write('Epoch: {}'.format(epoch))
            f_t.write('\n')
        print('open file txt:')
#         print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
#         print('len of unseen:', len(unseen_dataset))

        results = metalearner.evaluate(unseen_dataloader,
                                       max_batches=len(unseen_dataset),#args.num_batches,
                                       verbose=args.verbose,
                                       epoch = epoch,
                                       # desc=epoch_desc.format(epoch + 1)
                                       desc='test',
                                       save_pth=folder
                                       )

        # Save best model
        if 'accuracies_after' in results:
            if (best_value is None) or (best_value < results['accuracies_after']):
                best_value = results['accuracies_after']
                save_model = True
        elif (best_value is None) or (best_value > results['mean_outer_loss']):
            best_value = results['mean_outer_loss']
            save_model = True
        else:
            save_model = False

        #save_model=True
        if save_model and (args.output_folder is not None):
            with open(args.model_path, 'wb') as f:
                torch.save({'UNet': model[0].state_dict(),
                            'embed': model[1].state_dict()}, f)
                print('Saved model..............')

    if hasattr(meta_train_dataset, 'close'):
        meta_train_dataset.close()
        unseen_dataset.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('MAML')

    # General
    parser.add_argument('folder', type=str,
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--unseen', type=str,
                        default='CPM',
                        help='Name of unseen dataset')
    parser.add_argument('--output-folder', type=str, default=None,
        help='Path to the output folder to save the model.')
    parser.add_argument('--meta_train_bs', type=int, default=2,
        help='meta train batch size')
    parser.add_argument('--meta_test_bs', type=int, default=2,
        help='meta test batch size')

    # Model
    parser.add_argument('--hidden-size', type=int, default=64,
        help='Number of channels in each convolution layer of the VGG network '
        '(default: 64).')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=8,
        help='Number of tasks in a batch of tasks (default: 8).')
    parser.add_argument('--num-steps', type=int, default=1,
        help='Number of fast adaptation steps, ie. gradient descent '
        'updates (default: 1).')
    parser.add_argument('--num-epochs', type=int, default=50,
        help='Number of epochs of meta-training (default: 50).')
    parser.add_argument('--num-batches', type=int, default=256,
        help='Number of batch of tasks per epoch (default: 256 for CPM).')
    parser.add_argument('--cl_w', type=float, default=1.0,
        help='Number of contrastive leanring loss weight (default: 1.0).')
    parser.add_argument('--cl_lamda', type=float, default=0.3,
        help='Value of contrastive leanring lamda parameter (default: 0.3).')
    parser.add_argument('--step-size', type=float, default=0.1,
        help='Size of the fast adaptation step, ie. learning rate in the '
        'gradient descent update (default: 0.1).')
    parser.add_argument('--first-order', action='store_true',
        help='Use the first order approximation, do not use higher-order '
        'derivatives during meta-optimization.')
    parser.add_argument('--meta-lr', type=float, default=0.001,
        help='Learning rate for the meta-optimizer (optimization of the outer '
        'loss). The default optimizer is Adam (default: 1e-3).')
    parser.add_argument('--seed', type=int, default=42)

    # Misc
    parser.add_argument('--num-workers', type=int, default=1,
        help='Number of workers to use for data-loading (default: 1).')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--use-cuda', action='store_true') 

    args = parser.parse_args()


    main(args)


