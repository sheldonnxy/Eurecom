import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConvolutionalNeuralNetwork(nn.Module):
    """Convolutional Neural Network class to get the posterior Gaussian distribution of the predicted power map."""

    def __init__(self):
        super(FullyConvolutionalNeuralNetwork, self).__init__()

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
        
        # self.l_loss_weightage = [1.0, 1.1]
        self.alpha = None
        self._b_trainable_combined_layers = True # if False freeze the combined layers
        self._b_trainable_mean_layers = True    # if False freeze the mean block layers
        self._b_trainable_std_layers = True     # if False freeze the std. block layers
        self.train_loss_mean = None
        self.train_loss_sigma = None
        self.val_loss_sigma = None
        self.val_loss_mean = None



    def loss_function(self, y_true, y_predict):
        """
        Args:
            if self.b_posterior_target:
                -`y_true`: is N x H x W x C tensor with channels C=2, where first channel
                    contains a posterior mean target and the other contains
                    a posterior standard deviation.
            else:
                -`y_true`: is N x H x W x C tensor tha contains C= num_sources true maps as a target.
                    if channels are combined then C = 1.

            -`y_predict`: is N x H x W x 1 PyTorch normal distribution whose 'loc' is the estimated
                    posterior mean and 'scale' is the estimated standard deviation.

        """
        y_true = y_true.permute(0, 3, 1, 2)
        # y_true = y_true[..., None] # to make the shape equivalent to that of the output of a neural estimator
        # y_predict = self.flatten(y_predict)

        if self.loss_metric is None or self.loss_metric == 'NLL':
            'The loss function is a Negative Log Likelihood'
            return -torch.mean(y_predict.log_prob(y_true))

        elif self.loss_metric == 'MSE':
            'The loss function is Mean Square Error'
            if self.b_posterior_target:
                "A target contains a posterior mean and a posterior standard deviation"
                power_mse = F.mse_loss(y_true[..., 0][..., None], y_predict.loc)
                std_mse = F.mse_loss(y_true[..., 1][..., None], y_predict.scale)
            else:
                power_mse = F.mse_loss(y_true, y_predict.loc)
                std_mse = F.mse_loss(5*torch.ones(y_true.shape), y_predict.scale)
            return self.l_loss_weightage[0] * power_mse + \
                   self.l_loss_weightage[1] * std_mse #[power_mse, std_mse]
        elif self.loss_metric == 'Custom':
            """The loss function is inspired from the paper
            Efficient estimation of conditional variance
            functions in stochastic regression.
            If self.alpha = 0, then freeze combined and standard dev. block layers.
            Elif self.alpha =1, then freeze combined and mean block layers.
            Else do not freeze any layers."""

            t_delta = (torch.square(y_true - y_predict.loc))
            # loss_sigma = torch.mean(torch.square(t_delta - torch.square(y_predict.scale)))
            loss_sigma = torch.mean(torch.square(torch.sqrt(t_delta) - y_predict.scale))
            loss_mean = torch.mean(t_delta)
            loss = self.alpha * loss_sigma + (1 - self.alpha) * loss_mean
            return loss, loss_sigma, loss_mean
        else:
            raise NotImplementedError
        


    def train_step(self, data, labels):
        """Perform a single training step for the neural network.

        Args:
            data: input data tensor
            labels: true labels tensor
        """
        # set the model to training mode
        self.train()

        # apply the neural network to the input data to obtain the predicted mean and standard deviation
        mu_pred, sigma_pred = self(data)
        predictions = torch.distributions.Normal(mu_pred, sigma_pred)

        # calculate the loss function for the neural network
        if self.loss_metric == "Custom":
            loss, loss_mean, loss_sigma = self.loss_function(labels, predictions)
            self.train_loss_mean(loss_mean.item())
            self.train_loss_sigma(loss_sigma.item())
        else:
            loss = self.loss_function(labels, predictions)

        # compute the gradients of the loss function with respect to the trainable variables of the neural network
        self.optimizer.zero_grad()
        loss.backward()

        # update the trainable variables using the optimizer
        self.optimizer.step()

        # update the training loss of the neural network
        self.train_loss(loss.item())



    def test_step(self, data, labels):
        """Perform a single testing step for the neural network.

        Args:
            data: input data tensor
            labels: true labels tensor
        """
        # set the model to evaluation mode
        self.eval()

        # apply the neural network to the input data to obtain the predicted mean and standard deviation
        with torch.no_grad():
            mu_pred, sigma_pred = self(data)
            predictions = torch.distributions.Normal(mu_pred, sigma_pred)

        # calculate the loss function for the neural network
        if self.loss_metric == "Custom":
            val_loss, val_loss_mean, val_loss_sigma = self.loss_function(labels, predictions)
            self.val_loss_mean(val_loss_mean.item())
            self.val_loss_sigma(val_loss_sigma.item())
        else:
            val_loss = self.loss_function(labels, predictions)

        # update the validation loss of the neural network
        self.val_loss(val_loss.item())



        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.train_loss = torch.nn.MSELoss()
        self.val_loss = torch.nn.MSELoss()
        self.train_loss_mean = torch.nn.MSELoss()
        self.train_loss_sigma = torch.nn.MSELoss()
        self.val_loss_mean = torch.nn.MSELoss()
        self.val_loss_sigma = torch.nn.MSELoss()

        # loss metric
        if self.loss_metric == 'Custom':
            assert alpha is not None
            self.alpha = alpha
            if alpha == 0:
                print("alpha 0= ", alpha)
                # freeze combined lsayers and std. deviation layers
                self.combined_layers.requires_grad = False
                self.std_deviation_layers.requires_grad = False
                self.mean_layers.requires_grad = True

            elif alpha == 1:
                print("alpha 1 =", alpha)
                # freeze combined layers and mean layers
                self.combined_layers.requires_grad = False
                self.std_deviation_layers.requires_grad = True
                self.mean_layers.requires_grad = False
            else:
                print("alpha 0. 5")
                self.combined_layers.requires_grad = True
                self.std_deviation_layers.requires_grad = True
                self.mean_layers.requires_grad = True

        if self.loss_metric is None:
            self.loss_metric = loss_metric

        # posterior target variable
        self.b_posterior_target = b_posterior_target



    def train_step(self, data, labels):
        """Perform a single training step for the neural network.

        Args:
            data: input data tensor
            labels: true labels tensor
        """
        # set the model to training mode
        self.train()

        # apply the neural network to the input data to obtain the predicted mean and standard deviation
        mu_pred, sigma_pred = self(data)
        predictions = torch.distributions.Normal(mu_pred, sigma_pred)

        # calculate the loss function for the neural network
        if self.loss_metric == "Custom":
            train_loss_mean, train_loss_sigma = self.loss_function(labels, predictions)
            loss = self.alpha * train_loss_mean + (1 - self.alpha) * train_loss_sigma
            self.train_loss_mean(train_loss_mean.item())
            self.train_loss_sigma(train_loss_sigma.item())
        else:
            loss = self.train_loss(labels, predictions.mean)

        # compute the gradients of the loss function with respect to the trainable variables of the neural network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update the training loss of the neural network
        self.train_loss(loss.item())



    def test_step(self, data, labels):
        """Perform a single testing step for the neural network.

        Args:
            data: input data tensor
            labels: true labels tensor
        """
        # set the model to evaluation mode
        self.eval()

        # apply the neural network to the input data to obtain the predicted mean and standard deviation
        with torch.no_grad():
            mu_pred, sigma_pred = self(data)
            predictions = torch.distributions.Normal(mu_pred, sigma_pred)

        # calculate the loss function for the neural network
        if self.loss_metric == "Custom":
            val_loss_mean, val_loss_sigma = self.loss_function(labels, predictions)
            val_loss = self.alpha * val_loss_mean + (1 - self.alpha) * val_loss_sigma
            self.val_loss_mean(val_loss_mean.item())
            self.val_loss_sigma(val_loss_sigma.item())
        else:
            val_loss = self.val_loss(labels, predictions.mean)

        # update the validation loss of the neural network
        self.val_loss(val_loss.item())