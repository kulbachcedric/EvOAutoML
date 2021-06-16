from pipelinehelper import PipelineHelper
from sklearn import clone
from sklearn.pipeline import Pipeline, _fit_transform_one
from sklearn.utils import _print_elapsed_time
from sklearn.utils.validation import check_memory


class OnlinePipeline(Pipeline):

    def partial_fit_predict(self, X, y):
        """ partial_fit_predict

        Partial fits and transforms data in all but last step, then partial
        fits and predicts in the last step

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            All the samples we want to predict the label for.

        y: An array_like object of length n_samples
            Contains the true class labels for all the samples in X

        Returns
        -------
        list
            The predicted class label for all the samples in X.

        """
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is None:
                pass
            if hasattr(transform, "partial_fit_transform"):
                Xt = transform.partial_fit_transform(Xt, y)
            else:
                Xt = transform.partial_fit(Xt, y).transform(Xt)

        if hasattr(self._final_estimator, "partial_fit_predict"):
            return self._final_estimator.partial_fit_predict(Xt, y)
        else:
            return self._final_estimator.partial_fit(Xt, y).predict(Xt)

    def partial_fit_transform(self, X, y=None):
        """ partial_fit_transform

        Partial fits and transforms data in all but last step, then
        partial_fit in last step

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The data upon which the transforms/estimator will create their
            model.

        y: An array_like object of length n_samples
            Contains the true class labels for all the samples in X

        Returns
        -------
        Pipeline
            self

        """
        raise NotImplementedError

    def partial_fit(self, X, y, classes):
        """ partial_fit_predict

                Partial fits and transforms data in all but last step, then partial
                fits and predicts in the last step

                Parameters
                ----------
                X: numpy.ndarray of shape (n_samples, n_features)
                    All the samples we want to predict the label for.

                y: An array_like object of length n_samples
                    Contains the true class labels for all the samples in X

                Returns
                -------
                list
                    The predicted class label for all the samples in X.

                """
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is None:
                pass
            if hasattr(transform, "partial_fit_transform"):
                Xt = transform.partial_fit_transform(Xt, y)
            else:
                Xt = transform.partial_fit(Xt, y).transform(Xt)


        return self._final_estimator.partial_fit(Xt, y)


class OnlinePipelineHelper(PipelineHelper):

    def partial_fit(self, X,y,**kwargs):
        """Fits the selected model."""
        if self.selected_model is None or self.selected_model == 'passthrough':
            return self
        else:
            try:
                res = self.selected_model.partial_fit(X, y, kwargs)
            except Exception as e:
                res = self.selected_model
            return res
