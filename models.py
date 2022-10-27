import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

from _mcd import monte_carlo_prediction, compute_uncertainty


def build_sequential_mlp_class(n_feature=64,
                               num_classes=2,
                               layers=2,
                               activation='relu',
                               hidden_size=40,
                               hidden_size_2=None,
                               kernel_initializer='glorot_uniform',
                               dropout=0.25,
                               dropout_2=0.,
                               use_bias=True,
                               kernel_reg_l1=0,
                               kernel_reg_l2=0,
                               learning_rate=0.001,
                               beta_1=0.9,
                               beta_2=0.999,
                               batch_normalisation=False,
                               loss=categorical_crossentropy):
    """
    Function to build a sequential MLP classifier using Keras Sequential with 2 or 3 Hidden layers

    Parameters
    ----------
    n_feature : int
        input size
    num_classes : int
        output size
    activation : str
        Hidden layer dense activation
    hidden_size : int
        size of the first hidden layer
    hidden_size_2 : int or None
        size of the second hidden layer if not None. Else no additional hidden layer is added
    kernel_initializer : str
        kernel_initializer of hidden layer
    dropout : 0<= float < 1
        dropout rate of the first hidden leyer
    dropout_2 : 0<= float < 1
        dropout of the second hidden layer
    use_bias : bool
        wheather to use bias
    kernel_reg_l1 : float
        Kernel regularizer L1
    kernel_reg_l2 : float
        Kernel regularizer L2
    learning_rate : float, default=0.001
        learning rate
    beta_1 : float, default=0.9
    beta_2 : float, default=0.99
    batch_normalisation : bool, default False
        Weather to use batch normalization before each layer with non-linear activation
    loss : str or Keras.losses object
        The loss function

    Returns
    -------
    compiled model : tf.Keras.Model
    """

    x = Sequential()
    x.add(Dense(hidden_size,
                kernel_initializer=kernel_initializer,
                activation='relu',
                input_dim=n_feature,
                kernel_regularizer=l1_l2(l1=kernel_reg_l1,
                                         l2=kernel_reg_l2))
          )
    x.add(Dropout(dropout))
    if batch_normalisation and activation not in ['linear']:
        x.add(BatchNormalization())
    x.add(Dense(hidden_size,
                kernel_initializer=kernel_initializer,
                activation=activation,
                use_bias=use_bias,
                kernel_regularizer=l1_l2(l1=kernel_reg_l1,
                                         l2=kernel_reg_l2)
                ))
    x.add(Dropout(dropout))
    if layers == 3:
        if hidden_size_2 is not None:
            if batch_normalisation and activation not in ['linear']:
                x.add(BatchNormalization())
            x.add(Dense(hidden_size_2,
                        kernel_initializer=kernel_initializer,
                        activation=activation,
                        use_bias=use_bias))
            x.add(Dropout(dropout_2))
    if batch_normalisation:
        x.add(BatchNormalization())
    x.add(Dense(num_classes,
                kernel_initializer=kernel_initializer,
                use_bias=use_bias,
                activation='softmax'))

    optimizer = Adam(lr=learning_rate,
                     beta_1=beta_1,
                     beta_2=beta_2)
    x.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])
    return x


def build_sequential_mlp_reg(n_feature=64,
                             n_output=1,
                             activation='relu',
                             layers=2,
                             hidden_size=40,
                             hidden_size_2=None,
                             kernel_initializer='glorot_uniform',
                             dropout=0.,
                             dropout_2=0.,
                             use_bias=True,
                             kernel_reg_l1=0,
                             kernel_reg_l2=0,
                             learning_rate=0.001,
                             beta_1=0.9,
                             beta_2=0.999,
                             batch_normalisation=False,
                             last_activation='linear',
                             loss="mse"):
    """
    Function to build a sequential MLP regressor using Keras Sequential with 2 or 3 Hidden layers

    Parameters
    ----------
    n_feature : int
        input size
    n_output : int
        output size
    activation : str
        Hidden layer dense activation
    hidden_size : int
        size of the first hidden layer
    hidden_size_2 : int or None
        size of the second hidden layer if not None. Else no additional hidden layer is added
    kernel_initializer : str
        kernel_initializer of hidden layer
    dropout : 0<= float < 1
        dropout rate of the first hidden leyer
    dropout_2 : 0<= float < 1
        dropout of the second hidden layer
    use_bias : bool
        wheather to use bias
    kernel_reg_l1 : float
        Kernel regularizer L1
    kernel_reg_l2 : float
        Kernel regularizer L2
    learning_rate : float, default=0.001
        learning rate
    beta_1 : float, default=0.9
    beta_2 : float, default=0.99
    batch_normalisation : bool, default False
        Weather to use batch normalization before each layer with non-linear activation
    last_activation : str
        last Dense activation
    loss : str or Keras.losses object
        The loss function

    Returns
    -------
    compiled model : tf.Keras.Model
    """
    if not((layers == 2) or (layers == 3)):
        raise ValueError('layers should be 2 or 3')
    x = Sequential()
    x.add(Dense(hidden_size,
                kernel_initializer=kernel_initializer,
                activation='relu',
                input_dim=n_feature,
                kernel_regularizer=l1_l2(l1=kernel_reg_l1,
                                         l2=kernel_reg_l2)))
    x.add(Dropout(dropout))
    if batch_normalisation and activation not in ['linear']:
        x.add(BatchNormalization())
    x.add(Dense(hidden_size,
                kernel_initializer=kernel_initializer,
                activation=activation,
                use_bias=use_bias,
                kernel_regularizer=l1_l2(l1=kernel_reg_l1,
                                         l2=kernel_reg_l2)
                ))
    x.add(Dropout(dropout))

    if layers == 3:
        if hidden_size_2 is not None:
            if batch_normalisation and activation not in ['linear']:
                x.add(BatchNormalization())
            x.add(Dense(hidden_size_2,
                        kernel_initializer=kernel_initializer,
                        activation=activation,
                        use_bias=use_bias))
            x.add(Dropout(dropout_2))

    if batch_normalisation:
        x.add(BatchNormalization())

    x.add(Dense(units=n_output,
                kernel_initializer=kernel_initializer,
                activation=last_activation,
                use_bias=use_bias))

    optimizer = Adam(lr=learning_rate,
                     beta_1=beta_1,
                     beta_2=beta_2)
    x.compile(loss=loss,
              optimizer=optimizer,
              metrics=['mse'])
    return x


class MLPRegressor(BaseEstimator, RegressorMixin):

    """Class implementing a MLP regressor (which support quantile)
     implementing variational approx. (Monte-Carlo dropout).

    Args
    ----------
    scaler : Callable
        scaler used to normalize or preprocess input
    model : Keras.model
        Regression Keras model
    epochs : int
        number of epochs
    batch_size : int
        number of element in each batch
    reinitialize : bool, default True
        reset model before fitting, needed for parameters search or cv
    verbose : int
        verbosity of self.model.fit() method
    **nn_params : dict
        additional argument passed to build_sequential_mlp_reg method
        except {n_features, n_output}

    """
    def __init__(self, epochs=1, batch_size=32, scaler=StandardScaler(),
                 output_scaler=None, loss='mean_squared_error', verbose=0,
                 reinitialize=True, **nn_params):

        self._estimator_type = 'regressor'
        self.scaler = scaler
        self.output_scaler = output_scaler
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.reinitialize = reinitialize
        self.verbose = verbose
        self.nn_params = nn_params
        self.model = None
        self.history = None

    def _create_model(self, n_feature, n_output):
        tf.keras.backend.clear_session()
        return build_sequential_mlp_reg(n_feature=n_feature,
                                        n_output=n_output,
                                        loss=self.loss,
                                        **self.nn_params)

    def fit(self, X, y, sample_weight=None, **fit_params):
        """
        Fit the model.

        Args
        ----------
        X: array-like
            Feature space
        y: array-like
            target space
        sample_weight: array-like, default None
            sample weights
        fit_params: dict
            arguments passed to self.model.fit()
        """
        n_feature = X.shape[1] if len(X.shape) > 1 else 1
        n_output = y.shape[1] if len(y.shape) > 1 else 1
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values
            
        X = self.scaler.fit_transform(X)
        if self.output_scaler is not None:
            if len(y.shape) == 1: y = y.reshape(-1, 1)
            y = self.output_scaler.fit_transform(y)
        if 'validation_data' in fit_params.keys():
            X_val = self.scaler.transform(fit_params['validation_data'][0])
            y_val = fit_params['validation_data'][1]
            if self.output_scaler is not None:
                if len(y_val.shape) == 1: y_val = y_val.reshape(-1, 1)
                y_val = self.output_scaler.transform(y_val)
            fit_params['validation_data'] = (X_val, y_val)

        if self.reinitialize or self.model is None:
            self.model = self._create_model(n_feature=n_feature, n_output=n_output)
        history = self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size,
                                 verbose=self.verbose, sample_weight=sample_weight, **fit_params)
        self.history = history.history
        return self

    def summary(self):
        if self.model is None:
            pass
        else:
            self.model.summary()

    def predict(self, X, nb_path=None, conf=False, **predict_params):
        """
        Make predictions with one or several paths in the network

        Args
        ----------
        X : array-like
            Feature space
        nb_path : int or None
            number of forward path in network used for prediction
        conf : bool, default False
            weather to return confidence as tuple( MC mean, MC variance, conf inf, conf sup 95%)
        predict_params : dict
            arguments passed to self.model.predict()
        """
        if self.model is None:
            raise NotFittedError('Model not fitted')

        X = self.scaler.transform(X)

        if nb_path is not None:
            mc_samples = monte_carlo_prediction(self.model, X,
                                                forward_path=nb_path)
            if self.output_scaler is not None:
                mc_samples = np.apply_along_axis(self.output_scaler.inverse_transform, 1, mc_samples)
            mean, var, _, _, conf_inf, conf_sup = compute_uncertainty(mc_samples)
            if conf:
                return mean, var, conf_inf, conf_sup
            else:
                return mean
        else:
            pred = self.model.predict(X, **predict_params)
            if self.output_scaler is not None:
                pred = self.output_scaler.inverse_transform(pred)
            return pred

    def set_params(self, **params):
        """
        Setting model parameters.

        Args
        ----------
        params: dict
            params to be set to model
        """
        if 'epochs' in params.keys():
            self.epochs = params['epochs']
            del params['epochs']
        if 'batch_size' in params.keys():
            self.batch_size = params['batch_size']
            del params['batch_size']
        self.nn_params.update(params)
        self.model = None
        return self

    def save(self, path):
        """serialize model"""
        self.model.save(path)


def build_sequential_cnn_reg(input_shape,
                             kernel_size=(10, 10),
                             filters=32,
                             activation='relu',
                             n_output=1,
                             hidden_size=500,
                             use_bias=False,
                             bias_initializer='random_uniform',
                             loss=mean_squared_error,
                             optimizer=Adadelta(lr=0.01, rho=0.95, decay=0.0),
                             ):
    model = Sequential()
    model.add(Conv2D(filters,
                     kernel_size=kernel_size,
                     activation=activation,
                     input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(hidden_size, use_bias=True,
                    bias_initializer=bias_initializer,
                    activation=activation))
    model.add(Dense(n_output,
                    use_bias=use_bias,
                    bias_initializer=bias_initializer))
    model.compile(loss=loss,
                  optimizer=optimizer)
                  #optimizer=keras.optimizers.RMSprop(lr=0.001, rho=0.9, decay=0.0))
    return model


class CNNRegressor(BaseEstimator, RegressorMixin):

    """Class implementing a regressor using CNN structure.

    Args
    ----------
    input_shape: tuple
        input shape
    n_output: int
        number of output neurons
    epochs: int
        number of epochs
    batch_size: int
        number of element in each batch
    reinitialize: bool, default True
        reset model before fitting, needed for parameters search or cv
    loss: str or callable
        loss for the sequential model
    verbose: int
        verbosity of self.model.fit() method
    model: Keras.Sequential
        a sequential Keras model
    nn_params: any parameter of above sequential building func

    """
    def __init__(self, epochs, batch_size, input_shape=10,
                 loss='mean_squared_error', n_output=1,
                 reinitialize=True, verbose=0, **nn_params):
        self._estimator_type = 'regressor'
        self.input_shape = input_shape
        self.n_output = n_output
        self.epochs = epochs
        self.batch_size = batch_size
        self.reinitialize = reinitialize
        self.verbose = verbose
        self.loss = loss
        self.nn_params = nn_params
        self.model = self._create_model()
        self.history = None

    def _create_model(self):
        K.clear_session()
        return build_sequential_cnn_reg(input_shape=self.input_shape,
                                        n_output=self.n_output,
                                        loss=self.loss,
                                        **self.nn_params)

    def fit(self, X, y, **fit_params):
        """
        Fit the model.

        Args
        ----------
        X: array-like
            Feature space
        y: array-like
            target space
        kwargs: dict
            arguments passed to self.model.fit()
        """
        if self.reinitialize or self.model is None:
            self.model = self._create_model()
        history = self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size,
                                 verbose=self.verbose, **fit_params)
        self.history = history
        return self

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    def set_params(self, **params):
        if 'epochs' in params.keys():
            self.epochs = params['epochs']
            del params['epochs']
        if 'batch_size' in params.keys():
            self.batch_size = params['batch_size']
            del params['batch_size']
        self.nn_params.update(params)
        self.model = self._create_model()
        return self
