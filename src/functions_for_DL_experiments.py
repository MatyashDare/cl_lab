import torch, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
from functions_for_ML_experiments import visualize, count_metrics


def train_epochs(model, train_dataloader, val_dataloader,
                 num_exp,
                 scheduler, loss, optimizer,
                 device, num_epochs=10):
    metrics_dicts = []
    train_losses = []
    val_losses = []

    model.train()
    for epoch in tqdm(range(num_epochs)):

        train_loss_list = []
        for _, (t, f, l, p) in enumerate(train_dataloader):
            f = f.to(device).requires_grad_(True)
            t = t.to(dtype=torch.float32, device=device)
            logits = model(f, l).to(dtype=torch.float32, device=device)
            cur_loss = loss(logits, t)
            cur_loss.backward()

            optimizer.step()

            optimizer.zero_grad()

            if scheduler:
                scheduler.step(cur_loss)

            train_loss_list.append(cur_loss.item())


        val_loss_list = []
        with torch.no_grad():
            for _, (t, f, l, p) in enumerate(val_dataloader):
                f = f.to(device)
                t = t.to(dtype=torch.float32, device=device)
                logits = model(f, l).to(dtype=torch.float32, device=device)
                val_loss = loss(logits, t)
                val_loss_list.append(val_loss.item())
                tl = round(np.mean(train_loss_list), 4)
                vl = round(np.mean(val_loss_list), 4)

        print(f"Epoch: {epoch} | Train loss: {tl}| Val loss: {vl}")
        train_losses.append(tl)
        val_losses.append(vl)


    path = f'../experiments_dl/exp{num_exp}/'
    if not os.path.exists(path):
        os.mkdir(path)
        if not os.path.exists(path+'images'):
            os.mkdir(path+'images')
    plot_losses(train_losses, val_losses, path)
    torch.save(model.state_dict(), path+'model.pt')
    
    return model, path 


def test_preds(model, test_dataloader, parameters, device, path):
    model.eval()

    test_metric_list = []
    all_preds = []
    all_correct = []
    all_paths = []
    
    with torch.inference_mode():
        for i, (t, f, l, p) in enumerate(test_dataloader):

                f = f.to(device)
                t = t.to(dtype=torch.float32, device=device).argmax(dim=1)
                logits = model(f, l).to(dtype=torch.float32, device=device)
                test_pred = logits.argmax(dim=1)

                all_preds.extend(list(test_pred.to('cpu').numpy()))
                all_correct.extend(list((t.to('cpu').numpy())))
                all_paths.extend(p)

    visualize(all_correct,
              all_preds,
              labels_sorted=[0,1],
              algorithm=str(parameters),
              image_path=path)
    
    metrics_dict = count_metrics(all_correct,
              all_preds, str(parameters))
    

    cr = classification_report(all_correct,
              all_preds)
    print(cr)

    df_preds = pd.DataFrame({'paths': all_paths, 'alc': all_correct, 'preds': all_preds})
    df_preds.to_csv(path+'preds.csv', index=False)

    df_cr = pd.DataFrame(classification_report(all_correct,
              all_preds, output_dict=True)).transpose()
    df_cr.to_csv(path+'classification_report.csv', index=False)
    
    
    return metrics_dict


def plot_losses(train_losses, val_losses, path):
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title('Losses')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig(f'{path}losses.png')
    plt.show()
    