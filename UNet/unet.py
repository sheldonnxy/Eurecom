import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    """UNet architecture for radio map estimation and uncertainty mapping."""

    def __init__(self, in_channels=2, out_channels=1):
        super(UNet, self).__init__()

        # buffer for loss
        self.l_train_loss = []
        self.l_validation_loss = []
        self.l_epochs = []
        self.l_train_loss_mean = []
        self.l_train_loss_sigma = []
        self.l_val_loss_mean = []
        self.l_val_loss_sigma = []

        # metrics
        self.optimizer = None
        self.train_loss = None
        self.val_loss = None
        self.loss_metric = None
        self.b_posterior_target = False
        self.alpha = None
        self._b_trainable_combined_layers = True
        self._b_trainable_mean_layers = True
        self._b_trainable_std_layers = True
        self.train_loss_mean = None
        self.train_loss_sigma = None
        self.val_loss_sigma = None
        self.val_loss_mean = None

        # Encoder (downsampling)
        self.enc1 = self.double_conv(in_channels, 64)
        self.enc2 = self.double_conv(64, 128)
        self.enc3 = self.double_conv(128, 256)
        self.enc4 = self.double_conv(256, 512)

        # Decoder (upsampling)
        self.dec4 = self.double_conv(512 + 256, 256)
        self.dec3 = self.double_conv(256 + 128, 128)
        self.dec2 = self.double_conv(128 + 64, 64)
        self.dec1 = self.double_conv(64, 32)

        # Final output layers
        self.final_conv_mu = nn.Conv2d(32, out_channels, kernel_size=1)
        self.final_conv_sigma = nn.Conv2d(32, out_channels, kernel_size=1)

        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Decoder
        d4 = self.dec4(torch.cat([self.upsample(e4), e3], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e1], dim=1))
        d1 = self.dec1(self.upsample(d2))

        # Final output
        mu = self.final_conv_mu(d1)
        sigma = F.elu(self.final_conv_sigma(d1)) + 1  # ELU activation + 1 for non-negative sigma

        return mu, sigma

    def loss_function(self, y_true, y_predict):
        y_true = y_true.permute(0, 3, 1, 2)

        if self.loss_metric is None or self.loss_metric == 'NLL':
            return -torch.mean(y_predict.log_prob(y_true))
        elif self.loss_metric == 'MSE':
            if self.b_posterior_target:
                power_mse = F.mse_loss(y_true[..., 0][..., None], y_predict.loc)
                std_mse = F.mse_loss(y_true[..., 1][..., None], y_predict.scale)
            else:
                power_mse = F.mse_loss(y_true, y_predict.loc)
                std_mse = F.mse_loss(5*torch.ones_like(y_true), y_predict.scale)
            return self.l_loss_weightage[0] * power_mse + self.l_loss_weightage[1] * std_mse
        elif self.loss_metric == 'Custom':
            t_delta = (torch.square(y_true - y_predict.loc))
            loss_sigma = torch.mean(torch.square(torch.sqrt(t_delta) - y_predict.scale))
            loss_mean = torch.mean(t_delta)
            loss = self.alpha * loss_sigma + (1 - self.alpha) * loss_mean
            return loss, loss_sigma, loss_mean
        else:
            raise NotImplementedError

    def train_step(self, data, labels):
        self.train()
        mu_pred, sigma_pred = self(data)
        predictions = torch.distributions.Normal(mu_pred, sigma_pred)

        if self.loss_metric == "Custom":
            loss, loss_mean, loss_sigma = self.loss_function(labels, predictions)
            self.train_loss_mean(loss_mean.item())
            self.train_loss_sigma(loss_sigma.item())
        else:
            loss = self.loss_function(labels, predictions)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_loss(loss.item())

    def test_step(self, data, labels):
        self.eval()
        with torch.no_grad():
            mu_pred, sigma_pred = self(data)
            predictions = torch.distributions.Normal(mu_pred, sigma_pred)

        if self.loss_metric == "Custom":
            val_loss, val_loss_mean, val_loss_sigma = self.loss_function(labels, predictions)
            self.val_loss_mean(val_loss_mean.item())
            self.val_loss_sigma(val_loss_sigma.item())
        else:
            val_loss = self.loss_function(labels, predictions)

        self.val_loss(val_loss.item())

