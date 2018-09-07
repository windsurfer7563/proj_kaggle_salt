import json
from datetime import datetime
from pathlib import Path

import random
import numpy as np

import torch
import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#def variable(x, volatile=False):
#    if isinstance(x, (list, tuple)):
#       return [variable(y, volatile=volatile) for y in x]
#    return cuda(Variable(x, volatile=volatile))


#def cuda(x):
#    return x.cuda(async=True) if torch.cuda.is_available() else x


def write_event(log, step: int, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def train(args, model, criterion, train_loader, valid_loader, validation, init_optimizer, n_epochs=None, fold=None, num_classes=None):
    SEED = 47
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    np.random.seed(SEED)

    torch.backends.cudnn.deterministic = True


    lr = args.lr
    n_epochs = n_epochs or args.n_epochs
    optimizer = init_optimizer(lr)

    root = Path(args.root)
    model_path = root / 'model_{fold}.pt'.format(fold=fold)
    best_model_path = root / 'model_best_{fold}.pt'.format(fold=fold)
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        optimizer = state['optimizer']
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        'optimizer': optimizer.state_dict(),
    }, str(model_path))

    save_best_model = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(best_model_path))



    report_each = 10
    log = root.joinpath('train_{fold}.log'.format(fold=fold)).open('at', encoding='utf8')
    valid_losses = []
    best_iou = 0.77
    for epoch in range(epoch, n_epochs + 1):
        # TODO code for best model(based on validation score) savings
        model.train()
        random.seed()

        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size), ascii=True)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
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
            if valid_metrics['mean iou'] > best_iou:
                save_best_model(epoch + 1)
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return
