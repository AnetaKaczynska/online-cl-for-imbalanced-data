import torch
from torch import nn, optim
from sklearn.metrics import accuracy_score
from .utils import val_acc_per_subset


def train(model, train_dl, test_dl, epochs_per_set=1, lr=1e-3, buffer=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    table = []

    for task_id, task in enumerate(train_dl):
        a = 1 / (task_id + 1)
        model.train()
        for input, target in task:
            input = input.float().cuda()
            target = target.cuda()
            output = model(input)
            for step in range(5):
                optimizer.zero_grad()
                output = model(input)
                loss_s = criterion(output, target)
                loss_r = 0
                if buffer:
                    m_input, m_target = buffer.sample(len(input))
                    if m_input is not None and m_target is not None:
                        m_input = m_input.float().cuda()
                        m_target = m_target.cuda()
                        m_output = model(m_input)
                        loss_r = criterion(m_output, m_target)
                    else:
                        loss_r = 0
                loss = a * loss_s + (1 - a) * loss_r
                loss.backward()
                optimizer.step()
            if buffer:
              buffer.update_memory(input, target)

        model.eval()
        if (i+1) % epochs_per_set == 0:
            predictions = []
            targets = []
            with torch.no_grad():
                for input, target in test_dl:
                    input = input.cuda()
                    output = model(input)
                    predicted = torch.argmax(output, 1)
                    predictions += predicted.cpu()
                    targets += target

            val_acc = accuracy_score(targets, predictions)
            subset = task_id #[2 * task_id, 2 * task_id + 1]
            table.append([subset] + val_acc_per_subset(targets, predictions) + [val_acc])

    return table