from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import time
import numpy as np
import random
import os
from model import Model
from dataset import Dataset
from train import train
from test import test
import option


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    # setup_seed(2333)
    args = option.parser.parse_args()
    device = torch.device("cuda")
    train_loader = DataLoader(Dataset(args, test_mode=False),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=5, shuffle=False,
                              num_workers=args.workers, pin_memory=True)


    device = torch.device('cuda:{}'.format(args.gpus) if args.gpus != '-1' else 'cpu')
    model = Model(args).to(device)

    for name, value in model.named_parameters():
        print(name)
    approximator_param = list(map(id, model.approximator.parameters()))
    approximator_param += list(map(id, model.conv1d_approximator.parameters()))
    base_param = filter(lambda p: id(p) not in approximator_param, model.parameters())

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')
    optimizer = optim.Adam([{'params': base_param},
                            {'params': model.approximator.parameters(), 'lr': args.lr / 2},
                            {'params': model.conv1d_approximator.parameters(), 'lr': args.lr / 2},
                            ],
                            lr=args.lr, weight_decay=0.000)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)
    criterion = torch.nn.BCELoss()

    is_topk = True
    gt = np.load(args.gt)
    pr_auc, pr_auc_online = test(test_loader, model, device, gt)
    print('Random initalization: offline pr_auc:{0:.4}; online pr_auc:{1:.4}\n'.format(pr_auc, pr_auc_online))
    for epoch in range(args.max_epoch):
        scheduler.step()
        st = time.time()
        train(train_loader, model, optimizer, criterion, device, is_topk)
        if epoch % 2 == 0 and not epoch == 0:
            torch.save(model.state_dict(), './ckpt/'+args.model_name+'{}.pkl'.format(epoch))

        pr_auc, pr_auc_online = test(test_loader, model, device, gt)
        print('Epoch {0}/{1}: offline pr_auc:{2:.4}; online pr_auc:{3:.4}\n'.format(epoch, args.max_epoch, pr_auc, pr_auc_online))
    torch.save(model.state_dict(), './ckpt/' + args.model_name + '.pkl')
