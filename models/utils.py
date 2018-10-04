import json
from datetime import datetime
from pathlib import Path

import random
import numpy as np

import shutil
import torch
import tqdm
from torch import optim
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, StepLR, ReduceLROnPlateau, LambdaLR, CosineAnnealingLR


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def write_event(log, step: int, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]
    return lr[0]


def init_optimizer(config, warmup, model):
    optimizer_parameters = {
        'adam': {},
        'sgd': {'momentum': 0.9, 'weight_decay': 0.0001},
    }
    optimizers = {
        'adam': optim.Adam,
        'sgd': optim.SGD
    }

    lr = config.initial_lr if warmup != 1 else config.warmap_lr

    # we can config optimizer to have different lr for encoder and decoder
    # lr_enc = lr
    # base_params = list(map(id, model.encoder.parameters()))
    # logits_params = filter(lambda p: id(p) not in base_params, model.parameters())
    # params = [{"params": logits_params, "lr": lr},
    #          {"params": model.encoder.parameters(), "lr": lr_enc}]
    # init_optimizer = lambda params: optimizers[config.optimizer](params, **optimizer_parameters[config['optimizer']])
    # optimizer = init_optimizer(params)

    optimizer = optimizers[config.optimizer]
    return optimizer(filter(lambda p: p.requires_grad, model.parameters()),
                                                             lr=lr,
                                                             **optimizer_parameters[config.optimizer])

def init_scheduler(config, optimizer):
    if config.scheduler  == 'StepLR':
        lr_scheduler = StepLR(optimizer, config.lr_steps[0], config.lr_gamma)
    elif config.scheduler == 'MultiStepLR':
        lr_scheduler = MultiStepLR(optimizer, milestones=config.lr_steps, gamma=config.lr_gamma)
    elif config.scheduler == 'CyclicScheduler':
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = config.initial_lr
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=config.lr_steps[0], eta_min=config.min_lr, last_epoch=0)
    else:
        print("LR Scheduler type is not defined in config file")
        raise NotImplementedError

    return lr_scheduler



def train(args, model, config, criterion, train_loader, valid_loader, validation, n_epochs=None, fold=None, num_classes=None):
    SEED = 47
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    np.random.seed(SEED)
    random.seed(SEED)

    torch.backends.cudnn.deterministic = True

    n_epochs = n_epochs or args.n_epochs

    optimizer = init_optimizer(config,args.warmup, model)
    lr_scheduler = init_scheduler(config, optimizer)

    if torch.cuda.is_available():
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
        #model = model.to(device)


    root = Path(args.root)
    model_path = root / 'model_{fold}.pt'.format(fold=fold)

    if args.config.endswith('finetune.json') and args.resume == 0:
        state = torch.load(
            'data/models/{model_type}/pretrained/model_{fold}.pt'.format(model_type=config.model, fold=fold))
        model.load_state_dict(state['model'])
        epoch = 1
        step = 0
        print("Using pretrained model")
    elif args.config.endswith('finetune2.json') and args.resume == 0:
        state = torch.load(
            'data/models/{model_type}/pretrained2/model_{fold}.pt'.format(model_type=config.model, fold=fold))
        model.load_state_dict(state['model'])
        epoch = 1
        step = 0
        print("Using pretrained model2")
    elif args.resume == 1 and model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        #'lr_scheduler': lr_scheduler.state_dict(),
    }, str(model_path))

    def save_best_model(model, ep, step,  miou):
        """
        each 10 epoch we will save best model
        """

        file_name = 'model_best_{fold}_{epoch}_{miou:.0f}.pt'.format(epoch=ep, fold=fold, miou=miou*10000)
        torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        }, str(root / file_name))
        best_models[file_name] = miou

    def save_best_model_for_future_use():
        """
        after training ends we will copy best models in 'pretrained' folder
        for use in the next stage of training
        """
        if len(best_models) > 0:
            best_model = max(best_models, key=best_models.get)
        else:
            best_model = str(model_path)

        if args.config.endswith('pretrain.json'):
            save_path = 'data/models/{model_type}/pretrained/model_{fold}.pt'.format(model_type=config.model, fold=fold)
        elif args.config.endswith('filetune.json'):
            save_path = 'data/models/{model_type}/pretrained2/model_{fold}.pt'.format(model_type=config.model, fold=fold)
        else:
            save_path = 'data/models/{model_type}/model_{fold}.pt'.format(model_type=config.model, fold=fold)

        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(best_model, str(save_path))
        print("Best model {} saved to {}".format(best_model,str(save_path)))


    report_each = 10
    log = root.joinpath('train_{config}_{fold}.log'.format(config=args.config[:-5], fold=fold)).open('at', encoding='utf8')
    valid_losses = []

    # threshold to start save best models
    if args.config.endswith('filetune.json'):
        BEST_IOU = 0.84
    else:
        BEST_IOU = 0.82

    best_models = {}
    best_iou = 0

    print("Criteries before training starts:")
    validation(model, criterion, valid_loader, num_classes)

    for epoch in range(epoch, n_epochs + 1):
        if config.scheduler != 'CyclicScheduler':
            lr_scheduler.step(epoch)

        model.train()

        if args.freeze_bn:
            model = freeze_bn(model)

        tq = tqdm.tqdm(total=(len(train_loader) * config.batch_size), ascii=True)
        tq.set_description('Epoch {0}, lr {1:.6f}'.format(epoch, get_learning_rate(optimizer)))
        losses = []
        tl = train_loader
        try:
            mean_loss = 0

            for i, (inputs, targets) in enumerate(tl):
                if config.scheduler == 'CyclicScheduler':
                    if step % config.lr_steps[0] == 0:
                        lr_scheduler.last_epoch = 0
                    lr_scheduler.step()

                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)

                optimizer.step()
                step += 1
                tq.update(batch_size)
                losses.append(loss.data.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss, lr=lr_scheduler.get_lr()[0])

            write_event(log, step, loss=mean_loss)
            tq.close()
            save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader, num_classes)
            valid_metrics['lr'] = lr_scheduler.get_lr()[0]
            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)

            miou = valid_metrics['mean_iou2']
            if miou > BEST_IOU and miou > best_iou:
                miou = valid_metrics['mean_iou2']
                save_best_model(model, epoch + 1, step, miou)
                best_iou = valid_metrics['mean_iou2']
                print("Best model saved.")

            # will find best model every 10 epoch
            if epoch % 10 == 0:
                best_iou = 0
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            save_best_model_for_future_use()
            print('done.')
            return

    # at the end of the training



    save_best_model_for_future_use()


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    return model
