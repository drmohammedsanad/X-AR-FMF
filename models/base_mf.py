import numpy as np

class BaseMF:
    """
    Standard Matrix Factorization (baseline)
    """

    def __init__(self, num_users, num_items, k=20, lr=0.01, reg=0.01):
        self.num_users = num_users
        self.num_items = num_items
        self.k = k
        self.lr = lr
        self.reg = reg

        # Initialize latent vectors
        self.U = np.random.normal(0, 0.01, (num_users, k))
        self.V = np.random.normal(0, 0.01, (num_items, k))

    def train(self, data, epochs=10):
        for _ in range(epochs):
            for u, i, r in data:
                pred = self.predict(u, i)
                err = r - pred

                # Gradient updates
                self.U[u] += self.lr * (err * self.V[i] - self.reg * self.U[u])
                self.V[i] += self.lr * (err * self.U[u] - self.reg * self.V[i])

    def predict(self, u, i):
        return np.dot(self.U[u], self.V[i])