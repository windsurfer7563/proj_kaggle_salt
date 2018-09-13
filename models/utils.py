import json
from datetime import datetime
from pathlib import Path

import random
import numpy as np

import torch
import tqdm
from torch import optim

optimizers = {
    'adam': optim.Adam,
    'rmsprop': optim.RMSprop,
    'sgd': optim.SGD
}
optimizer_parameters = {
    'adam': {},
    'sgd':{'momentum':0.9, 'weight_decay': 0.0001},
    'rmsprop': {}
}


from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau, LambdaLR

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
    return lr


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

    #init_optimizer = lambda lr: optimizers[config['optimizer']](filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
    #                                                         **optimizer_parameters[config['optimizer']])

    lr = config['initial_lr'] if args.warmup != 1 else config['warmap_lr']
    lr_enc = lr if args.warmup != 1 else lr*0.1

    base_params = list(map(id, model.encoder.parameters()))
    logits_params = filter(lambda p: id(p) not in base_params, model.parameters())
    params = [{"params": logits_params, "lr": lr},
              {"params": model.encoder.parameters(), "lr": lr_enc}]

    init_optimizer = lambda params: optimizers[config['optimizer']](params, **optimizer_parameters[config['optimizer']])


    optimizer = init_optimizer(params)

    #lr_scheduler = ExponentialLR(optimizer, config['lr_gamma'])
    lr_scheduler = MultiStepLR(optimizer, milestones=config['lr_steps'], gamma=0.1)
    #lr_scheduler = ReduceLROnPlateau(optimizer, factor = 0.5, patience=6, verbose = True, mode='max')

    if torch.cuda.is_available():
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()



    root = Path(args.root)
    model_path = root / 'model_{fold}.pt'.format(fold=fold)
    best_model_path = root / 'model_best_{fold}.pt'.format(fold=fold)
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        #if step > 1500:
        #    lr_scheduler.load_state_dict(state['lr_scheduler'])

        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        'lr_scheduler': lr_scheduler.state_dict(),
    }, str(model_path))

    save_best_model = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        }, str(best_model_path))



    report_each = 10
    log = root.joinpath('train_{fold}.log'.format(fold=fold)).open('at', encoding='utf8')
    valid_losses = []
    best_iou = 0.797




    for epoch in range(epoch, n_epochs + 1):
        lr_scheduler.step()
        model.train()

        #random.seed()

        tq = tqdm.tqdm(total=(len(train_loader) * config['batch_size']), ascii=True)
        #tq.set_description('Epoch {}, lr {}'.format(epoch, lr_scheduler.get_lr()))
        tq.set_description('Epoch {}, lr {}'.format(epoch, get_learning_rate(optimizer)))
        losses = []
        tl = train_loader
        try:
            mean_loss = 0

            for i, (inputs, targets) in enumerate(tl):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                loss.backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)
                losses.append(loss.data.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)
            write_event(log, step, loss=mean_loss)
            tq.close()
            save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader, num_classes)
            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)

            #if epoch > config['warmup']:
            #    lr_scheduler.step()

            #lr_scheduler.step(valid_metrics['mean_iou'])
            if valid_metrics['mean_iou'] > best_iou:
                save_best_model(epoch + 1)
                best_iou = valid_metrics['mean_iou']
                print("Best model saved.")
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return
