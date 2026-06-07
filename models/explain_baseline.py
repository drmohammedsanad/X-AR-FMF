import numpy as np

class ExplainBaseline:
    """
    Simple post-hoc explanation baseline
    Uses magnitude of interaction as importance
    """

    def explain(self, user_vec, item_vec):
        """
        Estimate importance score
        """

        # Element-wise interaction importance
        importance = np.abs(user_vec * item_vec)

        return np.sum(importance)
