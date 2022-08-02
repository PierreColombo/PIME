# TODO à faire pour malik

def unit_tests():

    n_dims = 20
    n_points = 1000
    n_classes = 10
    n_tries = 10
    classifier = LogisticRegression()
    entropy_estimator = AlphaEntropy(name='voldemort', alpha=2.0)

    for i in range(n_tries):

        # 1) General case : make sure that the mi value is in the good range
        mi_estimator = LinearEstimator(classifier, entropy_estimator)
        x = torch.randn(n_points, n_dims)
        y = torch.randint(n_classes, (n_points,))

        predicted_mi = mi_estimator.predict(x, y)

        assert predicted_mi >= 0 and predicted_mi <= math.log(n_classes)

        # 2) Make sure that the mi is 0 in a trivial case
        class TrivialClassifier:

            def __init__(self, num_classes: int):
                self.num_classes = num_classes

            def fit(self, *args, **kwargs):
                pass

            def predict_proba(self, X):
                return np.ones((X.shape[0], self.num_classes)) / self.num_classes

        mi_estimator = LinearEstimator(TrivialClassifier(n_classes), entropy_estimator)
        predicted_mi = mi_estimator.predict(x, y)
        assert math.isclose(predicted_mi.item(), 0., abs_tol=1e-10), predicted_mi.item()
