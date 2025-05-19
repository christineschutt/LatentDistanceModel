import networkx as nx
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions import Normal
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

class LDM(torch.nn.Module):
    """
    Latent Distance Model (LDM) for predicting ordinal categories
    from a drug-side effect adjacency matrix using a probit link function.

    Inputs:
    ------
    Aij: adjacency matrix

    embedding_dim : int, the number of dimensions for the embedding space

    device : computational device

    n_epochs : int, the number of epochs the models trains for

    lr : learning rate

    seed : int
    """
    def __init__(self, Aij, embedding_dim, device, n_epochs, lr, seed=None):
        super(LDM, self).__init__()
        self.Aij = Aij.to(device)
        self.device = device
        self.n_drugs, self.n_effects = Aij.shape
        self.n_ordinal_classes = Aij.max().int().item() +1

        #set seed
        self.seed = seed
        self.__set_seed(seed)

        #Variables for the learning process
        self.n_epochs = n_epochs
        self.lr = lr

        #parameters to be learned (latent representations)
        self.gamma = nn.Parameter(torch.randn(1, device=device))
        self.beta = nn.Parameter(torch.randn(self.n_effects, device=device))
        self.w = torch.nn.Parameter(torch.randn(self.n_drugs, embedding_dim, device=device))  # Latent embeddings for drugs
        self.v = torch.nn.Parameter(torch.randn(self.n_effects, embedding_dim, device=device))  # Latent embeddings for side effects

        # Parameters to be learned (thresholds)
        self.beta_thilde = nn.Parameter(torch.randn((self.n_ordinal_classes-1), device=device))
        self.a = nn.Parameter(torch.rand(1, device=device))
        self.b = nn.Parameter(torch.rand(1, device=device))

        #Weighting
        self.values, self.counts = torch.unique(self.Aij, return_counts=True)
        self.freqs = torch.tensor(self.counts, dtype=torch.float32, device= self.device)
        self.class_weights = 1.0 / (self.freqs + 1e-6)  # Avoid div by 0
        self.class_weights = self.class_weights / self.class_weights.sum()

    def __set_seed(self, seed):
        """Set random seed for reproducibility."""
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def get_embeddings(self):
        """Return learned embeddings for drugs and side effects."""
        return self.w, self.v

    def get_thresholds(self):
        """
        Compute ordinal thresholds ensuring positive increasing increments.
        """
        # Ensure thresholds remain positive and increasing
        deltas = torch.softmax(self.beta_thilde, dim = 0)  # Ensure positive increments
        thresholds = torch.cumsum(deltas, dim=0)* self.a - self.b
        return torch.cat([torch.tensor([-float("inf")], device=self.device), thresholds, torch.tensor([float("inf")], device=self.device)])
    
    def probit(self):
        """
        Compute the ordinal probability matrix using the probit link function.
        Returns:
            probit_matrix: [C x D x E] tensor of probabilities for each ordinal class. (C: n classes, D: n drugs, E : n effects)
            latent_var: [D x E] matrix latent scores.
        """
        normal_dist = Normal(0, 1) # Noise contaminated by normal distribution
        probit_matrix = torch.zeros((self.n_ordinal_classes, self.n_drugs, self.n_effects), device=self.device)
        thresholds = self.get_thresholds()
    
        #Linear term (\beta^T x_{i,j})
        linear_term = torch.matmul(self.Aij, self.beta.unsqueeze(1))

        # Distance term -|w_i - v_j|
        dist = -torch.norm(self.w.unsqueeze(1) - self.v.unsqueeze(0), dim=2)

        # Latent variable \beta^T x_{i,j} + \alpha(u_i - u_j)
        latent_var = self.gamma + linear_term + dist
        
        for y in range(self.n_ordinal_classes):
            z1 = latent_var - thresholds[y]
            z2 = latent_var - thresholds[y+1]
            probit_matrix[y, :, :] = normal_dist.cdf(z1) - normal_dist.cdf(z2)
        return probit_matrix, latent_var

    
    def predict_categories(self):
        """
        Predict the ordinal category for each drug-side effect pair.
        Returns:
            predictions: [D x E] tensor of predicted ordinal categpries.
            probit_matrix: [C x D x E] tensor of class probabilities.
        """
        probit_matrix, _ = self.probit()  # Call probit to get probabilities
        return torch.argmax(probit_matrix, dim=0), probit_matrix
    
    def ordinal_cross_entropy_loss(self):
        # Compute the predicted probabilities using the probit function
        probit_matrix, _ = self.probit() 
        # Initialize loss variable
        loss = 0.0

        # Convert Aij to a one-hot encoded tensor
        one_hot_target = torch.zeros(self.n_drugs, self.n_effects, self.n_ordinal_classes, device=self.device)
        one_hot_target.scatter_(-1, self.Aij.unsqueeze(-1).long(), 1)  # One-hot encoding

        # Compute the log-likelihood loss
        prob = probit_matrix  
        loss = -torch.mean(torch.log(torch.sum(prob * one_hot_target.permute(2, 0, 1), dim=0) + 1e-8))
        return loss

    def train(self):
        """
        Training loop. Optimize model parameters using Adam and minimize ordinal cross-entropy.
        Returns:
            epoch_losses: list of loss values for each epoch.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        epoch_losses = []

        for epoch in range(self.n_epochs):
            optimizer.zero_grad()  # Reset gradients
            loss = self.ordinal_cross_entropy_loss()  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update parameters
            
            
            epoch_losses.append(loss.item())
            print(f"Epoch {epoch}/{self.n_epochs}, Loss: {loss.item():.4f}")

        return epoch_losses 

    def get_params(self):
        return self.beta, self.w.detach().cpu().numpy(), self.v.detach().cpu().numpy(), self.beta_thilde.detach().cpu().numpy()
