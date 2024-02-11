import torch.nn as nn
import torch

class UnFlatten(nn.Module):
    def __init__(self, output_shape):
        super(UnFlatten, self).__init__()
        self.output_shape = output_shape
    def forward(self, x):
        return x.view(x.shape[0], *self.output_shape)


class VariationalAutoEncoder(nn.Module):
    # input_shape and output_shape should be the same for lossless encoding and decoding
    # h_dim is the shape of the flattened-data after going through the encoder
    # regression_output_shape is the shape of the output of the regression model mapping the latent representation to the audio
    def __init__(self, input_shape, output_shape, regression_output_shape, h_dim=324):
        super(VariationalAutoEncoder, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.h_dim = h_dim
        self.regression_output_shape = regression_output_shape

        # setting the latent dimension to the time segment that the network is trained on
        self.latent_dim = input_shape[1]

        # encoder which will give the latent-representation of the EEG data
        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_shape[-1], 32, 4),
            nn.ReLU(),
            nn.Conv2d(32, 16, 4),
            nn.ReLU(),
            nn.Conv2d(16, 8, 4),
            nn.ReLU(),
            nn.Conv2d(8, 4, 4),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Flatten(),
        )
         # latent mean and variance
        self.mean_layer = nn.Linear(self.h_dim, self.latent_dim)
        self.logvar_layer = nn.Linear(self.h_dim, self.latent_dim)

        self.unflatten = nn.Sequential(
            nn.Linear(self.latent_dim, torch.prod(torch.tensor(self.output_shape))),
            nn.ReLU(),
            UnFlatten(self.output_shape),

        )
        # decoder which will return the original EEG data
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 8, 4),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 16, 4),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 32, 4),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.input_shape[-1], 4),
            nn.Flatten(),
            nn.Linear(336960, torch.prod(torch.tensor(self.output_shape))),
            nn.ReLU(),
            UnFlatten(self.output_shape),
        )

        # regression model that will map the latent-space representation to the 128 bin spectrogram
        self.audio_regression = nn.Sequential(
            nn.Linear(self.latent_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, torch.prod(torch.tensor(self.regression_output_shape))),
            nn.ReLU(),
            UnFlatten(self.regression_output_shape),
        )

    # encodes input data and returns mean and variance vectors
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    # reparameterization sampling trick
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var*epsilon
        return z

    # returns both the decoding and audio regression of the latent vector
    def decode(self, x):
        x_2 = self.unflatten(x)
        x_2 = x_2.view(x_2.shape[0], 1, x_2.shape[1], x_2.shape[2])
        return self.decoder(x_2), self.audio_regression(x)


    def forward(self, x):
        x = x.float()
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat, x_reg = self.decode(z)
        x_hat = x_hat.view(x_hat.shape[0], 1, x_hat.shape[1], x_hat.shape[2])
        return x_hat, x_reg, mean, logvar, z

# combines reproduction loss, KL-Divergence, as well as the loss from the regression
# lambda: HYPERPARAMETER, weight of the regression loss
def loss_function(x, x_hat, x_reg, x_reg_label, mean, log_var, lmbda=10, theta=5):
    x = x.to(torch.float32)
    reproduction_loss = nn.functional.mse_loss(x_hat, x)



    regression_loss = nn.functional.mse_loss(x_reg, x_reg_label)

    KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + (lmbda * regression_loss) + (theta * KLD)

def test_loss(x_reg, x_reg_label):
    return nn.functional.mse_loss(x_reg, x_reg_label)