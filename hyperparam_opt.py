"""
The :mod:`ml_utils_ts.calibration` implements methods for hyperparameters optimization relying on the *Hyperopt*
library (Bayesian optimization or random search). It implements a functional version *bayesian_tuning*
and a sklearn-style class version *BayesianSearchCV*.

These methods support both time-series data (pandas objects with DatetimeIndex) and standard data
(array-like or dataframe) depending on the underlying model provided and cross-validation generator used.
"""

import logging
from typing import List
import inspect
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from hyperopt import Trials, STATUS_OK
from hyperopt import fmin
from hyperopt import tpe
from hyperopt.fmin import generate_trials_to_calculate
from sklearn.base import MetaEstimatorMixin, BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted


from typing import Callable


logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


__score__metrics = ["f1_score", "accuracy_score", "auc", "recall_score",
                    "precision_score", "fbeta_score", "r2_score",
                    "explained_variance_score"]

__loss__metrics = ["log_loss", "hinge_loss", "brier_score_loss", "zero_one_loss", "neg_log_loss",
                   "mean_squared_error", "root_mean_square_error", "mean_absolute_error",
                   "mean_poisson_deviance", "mean_gamma_deviance"]


def set_params(model, **params):
    if hasattr(model, 'set_params'):
        model.set_params(**params)
    else:
        for param, val in params.items():
            setattr(model, param, val)


def _apply_metric(metric, y_true, y_pred, sample_weight=None, **kw):

    if 'sample_weight' in inspect.signature(
            metric).parameters.keys():

        return metric(y_true, y_pred, sample_weight=sample_weight, **kw)
    else:
        return metric(y_true, y_pred, **kw)


def build_sample_weight(weight_feature, scaling='sum_to_one'):

    if scaling is None or weight_feature is None:
        return weight_feature
    elif scaling == 'minmax':
        scaler = MinMaxScaler()
        weight_feature = np.absolute(weight_feature)
        weights = scaler.fit_transform(weight_feature.reshape(-1, 1))
        return np.ravel(weights)
    elif scaling == 'sum_to_one':
        return (np.absolute(weight_feature) / np.absolute(weight_feature).sum())\
               * len(weight_feature)


def build_loss_fun(metric_fun: Callable):
    """
     Transform a score or loss function into a loss metric
     able to be passed in cv_methods for hyperparameters
     optimization purpose

    Parameters
    ----------
    metric_fun: Callable
        either a metric of module sklearn.metric
        or function with the signature
        func(y_true, y_pred, sample_weight=None) -> (eval_name, eval_result, is_higher_better)
        or func(y_true, y_pred, sample_weight=None) -> (eval_name, eval_result, is_higher_better)

    Returns
    -------
    loss_fun: callable
        a metric method of signature
        loss_fun(y_test, y_pred, sample_weight=None)
    """
    if metric_fun.__name__ in __loss__metrics:
        return metric_fun

    elif metric_fun.__name__ in __score__metrics:
        def loss_metric(y_true, y_pred, sample_weight=None):
            return - metric_fun(y_true, y_pred, sample_weight=sample_weight)
        loss_metric.__name__ = metric_fun.__name__
        return loss_metric

    else:
        def loss_metric(y_test, y_pred, sample_weight=None):
            name, val, is_h_b = metric_fun(y_test, y_pred,
                                           sample_weight=sample_weight)
            return (-1 if is_h_b else 1) * val
        loss_metric.__name__ = metric_fun.__name__
        return loss_metric


def kfold_cv(model, X, y,
             metric: Callable[[np.ndarray, np.ndarray], float],
             cv_gen=KFold(),
             sample_weight: np.ndarray = None,
             fit_params: dict = None,
             n_jobs: int = 1) -> List:
    '''
    Perform k-fold cross-validation

    Parameters
    ----------
    model:
        ML model object implementing fit and predict

    X : array-like or pd.DataFrame
        X values

    y : array-like or pd.Series
        y values

    metric: callable
        method for cv scoring, etheir sklearn metric or other
        custom scoring fun matching the signature:
        func(y_true, y_pred)->float or func(y_true, y_pred, sample_weight)->float

    cv_gen: sklearn BaseCrossValidator object
        cv split generator

    sample_weight: np.array
        sample weight vector

    fit_params: dict
        dictionary of parameters passed to model.fit

    n_jobs :  int
        number of worker for parallelisation

    Returns
    -------
    scores : list
        list of all scores obtained during cross-validation
    '''

    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    def _fit(train, test):
        '''
        Process train and test data,
         fit model and compute prediction and score

        Parameters
        ----------
        train: np.array
            train index
        test: np.array
            test index

        Returns
        -------
        score: float
            the value of the score computed on the test set according to
            the metric
        '''
        sample_weight_train = build_sample_weight(sample_weight[train])
        sample_weight_score = build_sample_weight(sample_weight[test])
        fit_params['sample_weight'] = sample_weight_train

        model.fit(X[train, :], y[train], **fit_params)

        if metric.__name__ == 'log_loss':
            y_pred = model.predict_proba(X[test, :])
        else:
            y_pred = model.predict(X[test, :])

        score = _apply_metric(metric, y[test], y_pred,
                              sample_weight=sample_weight_score)
        return score

    fit_params = fit_params or {}
    fit_params = fit_params.copy()
    if sample_weight is None:
        sample_weight = np.ones((X.shape[0],))

    parallel = Parallel(n_jobs=n_jobs, max_nbytes=None)
    scores = parallel(
        delayed(_fit)(train, test)
        for train, test in cv_gen.split(X, y)
    )

    return scores


def bayesian_tuning(X, y, model,
                    param_grid,
                    metric_fun,
                    cv_gen=KFold(),
                    folds_weights=None,
                    fit_params=None,
                    static_params=None,
                    trials=Trials(),
                    optimizer=tpe.suggest,
                    nb_evals=50,
                    refit=False,
                    random_state=None,
                    n_jobs=1,
                    **kwargs):
    """
    Perform a Bayesian-style optimization of a given ML model
    hyperparameters based on iteratives cross validations and scoring,
    then store trials in an dict. X, y inputs type have to be adapted to cv_gen inputs
    (array or pd.DataFrame for sklearn CV generator or pd.DataFrame with Datetime index for
    Purged CV generators). The method use the library Hyperopt : https://github.com/hyperopt/hyperopt

    Parameters
    ----------
    X: array-like or pd.DataFrame
        X data. It should be a pandas object with DatetimeIndex if cv_gen is a PurgedFoldBase object

    y: array-like or pd.DataFrame or pd.Series
        y data. It should be a pandas object with DatetimeIndex if cv_gen is a PurgedFoldBase object

    model:
        ML model object implementing fit and predict

    param_grid: dict
        Hyperopt type grid search dictionary (see Hyperopt doc :
        https://github.com/hyperopt/hyperopt/wiki/FMin)

    metric_fun: Callable
        either a metric of module sklearn.metric
        or function with the signature
        func(y_true, y_pred, sample_weight=None) -> (eval_name, eval_result, is_higher_better)
        or func(y_true, y_pred, sample_weight=None) -> (eval_name, eval_result, is_higher_better)

    cv_gen: PurgedFoldBase or sklearn BaseCrossValidator object instance
        cross-validation generator for model hyperparameters evaluation
        at each hyperopt fmin iteration. If instance of PurgedFoldBase,
        time-indexed pandas DataFrame and Series object should be provided
        as X and y

    folds_weights : list or array-like
        optional, weights vector to apply to test fold scores. Should have the same lenght as cv_gen.n_splits

    fit_params: dict
        dictionary of parameters passed to model.fit

    static_params: dict or None
        model hyperparameter that are passed in tuning loop

    trials: instance of Trials object
        Hyperopt storage object used for hp calibration

    optimizer:
        optimizer algo used by hyperopt

    nb_evals: int
        number of iteration of optimization process

    refit: bool
        weather to train model on all data with best parameters
        once hyperparam optimization finished

    random_state: int or None
        random state of hyperopt fmin func

    n_jobs: int
        number of worker for cross-validation parallel computing (multi-threading backend)

    kwargs: dict
        additional optional arguments passed to hyperopt.fmin method

    Returns
    -------
    trials_dict:
        list of dict containing optimization info at each iteration
    """
    loss_fun = build_loss_fun(metric_fun)

    def weighted_mean(data, weights):
        """function for weights averaging on cv test fold """
        data = data.dropna(axis=1)
        wm = np.average(data.values, axis=0, weights=weights)
        res = {}
        for i in range(len(data.columns)):
            res[data.columns[i]] = wm[i]
        return res

    def objective(hyperparameters):
        """Objective function for hyperopt optimization. Returns
           the cross validation score from a set of hyperparameters."""

        global ITERATION
        ITERATION += 1

        # deal with nested param space
        for param_name in list(hyperparameters):
            if type(hyperparameters[param_name]) == dict:
                # Retrieve each sub-parameter and put it at top level key
                for sub_param in hyperparameters[param_name].keys():
                    if sub_param != param_name:
                        sub_param_val = hyperparameters[param_name].get(sub_param)
                        hyperparameters[sub_param] = sub_param_val
                # put param with nested space at top level key
                hyperparameters[param_name] = \
                    hyperparameters[param_name][param_name]

        static = static_params or {}
        all_params = {**hyperparameters, **static}
        set_params(model, **all_params)

        result_score = kfold_cv(
            model=model,
            X=X,
            y=y,
            metric=loss_fun,
            fit_params=fit_params,
            cv_gen=cv_gen,
            n_jobs=n_jobs,
        )

        result_score = pd.DataFrame({'loss': result_score})

        # compute weighted mean on test folds, default weights set to one
        if folds_weights is not None:
            weights = folds_weights
        else:
            weights = np.ones(len(result_score))
        agg_score = weighted_mean(result_score, weights)
        agg_score['hyperparameters'] = all_params
        agg_score['status'] = STATUS_OK
        agg_score['iteration'] = ITERATION

        return agg_score

    global ITERATION
    ITERATION = 0
    # Run optimization
    result = fmin(fn=objective, space=param_grid,
                  algo=optimizer, trials=trials,
                  max_evals=nb_evals, show_progressbar=True,
                  rstate=random_state, **kwargs)

    trials_list = sorted(trials.results, key=lambda x: x['loss'])

    set_params(model, **trials_list[0]['hyperparameters'])
    if refit:
        log.info(f'model trained with following hyperparameters'
                 f"\n{trials_list[0]['hyperparameters']}")
        fit_params = fit_params or {}
        model.fit(X, y, **fit_params)

    return trials_list


class BayesianSearchCV(MetaEstimatorMixin, BaseEstimator):

    """ Bayesian hyperparameters optimization wrapper class for sklearn pipeline

    Parameters
    ----------
    estimator:
        ML model object implementing fit and predict

    param_distributions: dict
        Hyperopt type grid search dictionary (see Hyperopt doc :
        https://github.com/hyperopt/hyperopt/wiki/FMin)

    scoring: Callable
        either a metric of module sklearn.metric
        or function with the signature
        func(y_true, y_pred, sample_weight=None) -> (eval_name, eval_result, is_higher_better)
        or func(y_true, y_pred, sample_weight=None) -> (eval_name, eval_result, is_higher_better)

    cv_gen: PurgedFoldBase or sklearn BaseCrossValidator object instance
        cross-validation generator for model hyperparameters evaluation
        at each hyperopt fmin iteration. If instance of PurgedFoldBase,
        time-indexed pandas DataFrame and Series object should be provided
        as X and y

    points_to_evaluate: list[dict]
        list of dictionary of hyperparameters to be evaluated

    static_params: dict or None
        model hyperparameter that are passed in tuning loop

    optimizer:
        optimizer algo used by hyperopt

    n_iter: int
        number of iteration of optimization process

    refit: bool
        weather to train model on all data with best parametres
        once hyperparam optimization finished

    random_state: int or None
        random state of hyperopt fmin func

    n_jobs: int
        number of worker for cross-validation parallel computing (multi-threading backend)

    kwargs: dict
        additional optional arguments passed to hyperopt.fmin method

    Attributes
    ----------
    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator which gave highest score
        (or smallest loss if specified) on the left out data.

    best_score_ : float
        Mean cross-validated score of the best_estimator

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.
    """

    def __init__(self, estimator, param_distributions, scoring, cv=KFold(),
                 static_params=None, points_to_evaluate=None, optimizer=tpe.suggest,
                 n_iter=50, refit=True, random_state=None, n_jobs=1, **kwargs):

        self.estimator = estimator
        self.param_distributions = param_distributions
        self.scoring = scoring
        self.cv = cv
        self.static_params = static_params
        self.points_to_evaluate = points_to_evaluate
        self.optimizer = optimizer
        self.n_iter = n_iter
        self.refit = refit
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.fmin_params = kwargs

    def _check_is_fitted(self, method_name):
        if not self.refit:
            raise NotFittedError(f'This {type(self).__name__} instance was initialized '
                                 f'with refit=False. {method_name} is '
                                 'available only after refitting on the best '
                                 'parameters. You can refit an estimator '
                                 'manually using the ``best_params_`` '
                                 'attribute')
        else:
            check_is_fitted(self, attributes='best_estimator_')

    def fit(self, X, y, **fit_params):
        """

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and n_features is the number of features.
            It should be pandas object with DatetimeIndex if cv is a PurgedFoldBase object

        y : array-like or pd.Series of shape (n_samples, n_output) or (n_samples,)
            Target relative to X for classification or regression;
            It should be pandas object with DatetimeIndex if cv is a PurgedFoldBase object

        **fit_params : dict of str -> object
            Parameters passed to the fit method of the estimator

        Returns
        -------

        """
        if self.points_to_evaluate is not None:
            trials = generate_trials_to_calculate(self.points_to_evaluate)
        else:
            trials = Trials()
        refit = False if self.combine_estimators else self.refit
        bests = bayesian_tuning(X, y, model=self.estimator,
                                param_grid=self.param_distributions,
                                metric_fun=self.scoring,
                                cv_gen=self.cv,
                                fit_params=fit_params,
                                static_params=self.static_params,
                                trials=trials,
                                optimizer=tpe.suggest,
                                nb_evals=self.n_iter,
                                refit=refit,
                                random_state=self.random_state,
                                n_jobs=self.n_jobs,
                                **self.fmin_params)
        self.trials_ = bests
        self.best_estimator_ = self.estimator
        self.best_score_ = bests[0]['loss']
        self.best_params_ = bests[0]['hyperparameters']

    def predict(self, X):

        self._check_is_fitted('predict')
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):

        self._check_is_fitted('predict_proba')
        return self.best_estimator_.predict_proba(X)

