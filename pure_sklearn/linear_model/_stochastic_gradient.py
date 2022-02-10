"""
Stochastic Gradient Descent (SGD)
"""

from ._base import LinearClassifierMixinPure
from ..base import safe_log
from ..utils import check_types, check_version


class SGDClassifierPure(LinearClassifierMixinPure):
    """
    Pure python implementation of `SGDClassifier`.

    Args:
        estimator (sklearn estimator): fitted `SGDClassifier` object
    """

    def __init__(self, estimator):
        check_version(estimator)
        super().__init__(estimator=estimator)
        check_types(self)

    def _check_proba(self):
        if self.loss not in ["log", "modified_huber"]:
            raise AttributeError(
                "probability estimates are not available for loss=%r" % self.loss
            )

    @property
    def predict_proba(self):
        """Probability estimates.
        This method is only available for log loss and modified Huber loss.
        Multiclass probability estimates are derived from binary (one-vs.-rest)
        estimates by simple normalization, as recommended by Zadrozny and
        Elkan.
        """
        self._check_proba()
        return self._predict_proba

    def _predict_proba(self, X):
        if self.loss == "log":
            return self._predict_proba_lr(X)
        elif self.loss == "modified_huber":
            import numpy as np
            binary = len(self.classes_) == 2
            scores = self.decision_function(X)
            scores = np.array(scores)

            if binary:
                prob2 = np.ones((scores.shape[0], 2))
                prob = prob2[:, 1]
            else:
                prob = scores

            np.clip(scores, -1, 1, prob)
            prob += 1.0
            prob /= 2.0

            if binary:
                prob2[:, 0] -= prob
                prob = prob2
            else:
                # the above might assign zero to all classes, which doesn't
                # normalize neatly; work around this to produce uniform
                # probabilities
                prob_sum = prob.sum(axis=1)
                all_zero = prob_sum == 0
                if np.any(all_zero):
                    prob[all_zero, :] = 1
                    prob_sum[all_zero] = len(self.classes_)

                # normalize
                prob /= prob_sum.reshape((prob.shape[0], -1))

            return [a for a in prob]
        else:
            raise NotImplementedError("predict_(log_)proba only supported when"
                                      " loss='log' or 'modified_huber' "
                                      "(%r given)" % self.loss)

    @property
    def predict_log_proba(self):
        """
        Log of probability estimates.
        This method is only available for log loss and modified Huber loss.
        When loss="modified_huber", probability estimates may be hard zeros
        and ones, so taking the logarithm is not possible.
        See ``predict_proba`` for details.
        """
        self._check_proba()
        return self._predict_log_proba

    def _predict_log_proba(self, X):
        return [list(map(safe_log, a)) for a in self.predict_proba(X)]
