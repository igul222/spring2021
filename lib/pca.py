import torch

class PCA:
    """Pytorch PCA implementation."""
    def __init__(self, X, n_components, whiten=False):
        mean = X.mean(dim=0, keepdim=True)

        X = X - mean
        covariance = torch.einsum('nx,ny->xy', X, X) / float(X.shape[0])

        _, eigenvals, components = torch.svd(covariance)
        components = components[:, :n_components]
        eigenvals = eigenvals[:n_components]

        self.mean = mean
        self.components = components
        self.magnitudes = eigenvals.sqrt()
        self._whiten = whiten

    def forward(self, X):
        X = (X - self.mean) @ self.components
        if self._whiten:
            X = X / (1e-8 + self.magnitudes[None,:])
        return X

    def inverse(self, X):
        if self._whiten:
            X = X * self.magnitudes[None,:]
        X = (X @ self.components.T) + self.mean
        return X