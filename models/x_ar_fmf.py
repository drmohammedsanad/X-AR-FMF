import numpy as np

class XARFMF:
    """
    X-AR-FMF: Explainable Adaptive Fair Matrix Factorization

    Components:
    - Matrix Factorization (U, V)
    - Fairness regularization (user + item)
    - Explainability via centroid deviation
    """

    def __init__(self, num_users, num_items, k=20, alpha=0.05, beta=0.05, lr=0.01):
        self.num_users = num_users
        self.num_items = num_items
        self.k = k

        self.alpha = alpha   # user fairness strength
        self.beta = beta     # item fairness strength
        self.lr = lr

        # Initialize latent factors
        self.U = np.random.normal(0, 0.01, (num_users, k))
        self.V = np.random.normal(0, 0.01, (num_items, k))

    # --------------------------------------------------
    # TRAINING
    # --------------------------------------------------
    def train(self, data, user_groups, item_groups, epochs=10):

        for epoch in range(epochs):

            # Compute centroids at each epoch
            user_centroids = self.compute_user_centroids(user_groups)
            item_centroids = self.compute_item_centroids(item_groups)

            for u, i, r in data:

                pred = np.dot(self.U[u], self.V[i])
                error = r - pred

                # ------------------------------
                # Update user factors
                # ------------------------------
                self.U[u] += self.lr * (
                    error * self.V[i]
                    - self.alpha * (self.U[u] - user_centroids[user_groups[u]])
                )

                # ------------------------------
                # Update item factors
                # ------------------------------
                self.V[i] += self.lr * (
                    error * self.U[u]
                    - self.beta * (self.V[i] - item_centroids[item_groups[i]])
                )

    # --------------------------------------------------
    # CENTROID COMPUTATION
    # --------------------------------------------------
    def compute_user_centroids(self, user_groups):

        centroids = {}

        for g in set(user_groups.values()):
            idx = [u for u in user_groups if user_groups[u] == g]

            # Safety check
            if len(idx) == 0:
                centroids[g] = np.zeros(self.k)
            else:
                centroids[g] = np.mean(self.U[idx], axis=0)

        return centroids


    def compute_item_centroids(self, item_groups):

        centroids = {}

        for g in set(item_groups.values()):
            idx = [i for i in item_groups if item_groups[i] == g]

            # Safety check
            if len(idx) == 0:
                centroids[g] = np.zeros(self.k)
            else:
                centroids[g] = np.mean(self.V[idx], axis=0)

        return centroids

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------
    def predict(self, u, i):
        return np.dot(self.U[u], self.V[i])

    # --------------------------------------------------
    # EXPLAINABILITY
    # --------------------------------------------------
    def explanation_score(self, i, item_groups, item_centroids):
        """
        Measures FAIRNESS contribution to recommendation

        Higher score = stronger fairness adjustment
        """

        v_i = self.V[i]
        centroid = item_centroids[item_groups[i]]

        score = self.beta * np.linalg.norm(v_i - centroid) ** 2

        return score

    def explain(self, u, i, item_groups, item_centroids, threshold=0.01):
        """
        Generates human-readable explanation
        """

        score = self.explanation_score(i, item_groups, item_centroids)

        if score > threshold:
            return "Fairness-driven recommendation"
        else:
            return "Preference-driven recommendation"