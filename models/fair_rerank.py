import numpy as np

class FairRerank:
    """
    Post-processing fairness method
    Penalizes dominant groups in ranking
    """

    def __init__(self, lambda_fair=0.1):
        self.lambda_fair = lambda_fair

    def rerank(self, scores, item_groups):
        """
        scores: list of predicted scores
        item_groups: dict {item_id: group}
        """

        adjusted_scores = []

        for i, score in enumerate(scores):
            group = item_groups.get(i, 0)

            # Penalize dominant group
            penalty = self.lambda_fair * group

            adjusted_scores.append(score - penalty)

        # Return ranked indices (descending)
        return np.argsort(adjusted_scores)[::-1]