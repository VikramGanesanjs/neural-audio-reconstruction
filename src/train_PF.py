from pf_dataset import PFDataset, patient_data
from torch.utils.data import DataLoader, random_split
from model import VariationalAutoEncoder
import torch
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm


def train_PF():
    SEGMENT_LENGTH = 1.5 #(seconds)

    pf_dataset = PFDataset(patient_data, SEGMENT_LENGTH)

    # Training Dataset Setup
    train_dataset, _, _ = random_split(pf_dataset, [0.6, 0.2, 0.2])

    #HYPERPARAMETER
    batch_size = 16

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = VariationalAutoEncoder(pf_dataset.get_data_shape(), pf_dataset.get_data_shape(), pf_dataset.get_label_shape())

    #HYPERPARAMETER
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    #HYPERPARAMETER
    lmbda = 5

    #HYPERPARAMETER
    theta = 5

    #HYPERPARAMETER
    epochs = 3

    losses = []


    for epoch in range(epochs):
        summed_loss = 0
        for batch_idx, ((x, loc_cat, seg_idx, pat_id), x_label) in enumerate(tqdm(train_loader)):
            x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
            optim.zero_grad()
            x_label = x_label.float()
            x_hat, x_reg, mu, log_var, z = model(x)
            loss = model.loss_function(x, x_hat, x_reg, x_label, mu, log_var, lmbda=lmbda, theta=theta)
            summed_loss += loss.item()
            losses.append(loss.item()/batch_size)

            loss.backward()
            optim.step()
        print("\tEpoch", epoch + 1, "\tAverage Loss: ", summed_loss/(batch_idx*batch_size))
        losses.append(summed_loss/(batch_idx * batch_size))

    epochs_arr = [i+1 for i in range(len(losses))]
    plt.plot(epochs_arr, losses)

    torch.save(model.state_dict(), "model_weights_{}".format(datetime.now()))




if __name__ == "__main__":
    train_PF()
 