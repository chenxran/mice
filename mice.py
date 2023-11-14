import numpy as np
from statsmodels.regression.linear_model import OLS
from collections import defaultdict
from typing import Tuple, List


class OurMICEData:
    def __init__(self, data: np.ndarray, perturbation_method='gaussian',
                 k_pmm=20, history_callback=None):

        # if data.columns.dtype != np.dtype('O'):
        #     msg = "MICEData data column names should be string type"
        #     raise ValueError(msg)

        self.regularized = dict()

        # Drop observations where all variables are missing.  This
        # also has the effect of copying the data frame.
        # self.data = data.dropna(how='all')
        mask = np.isnan(data).all(axis=1)
        # Invert the mask to keep rows where not all elements are NaN
        self.data = data[~mask]

        # shape of data
        self.nrows, self.ncols = self.data.shape

        self.history_callback = history_callback
        self.history = []
        self.predict_kwds = {}

        # Assign the same perturbation method for all variables.
        # Can be overridden when calling 'set_imputer'.
        self.perturbation_method = defaultdict(lambda:
                                               perturbation_method)

        # Map from variable name to indices of observed/missing
        # values.
        self.ix_obs = {}
        self.ix_miss = {}
        for i in range(self.ncols):
            ix_obs, ix_miss = self._split_indices(self.data[:, i])
            self.ix_obs[i] = ix_obs
            self.ix_miss[i] = ix_miss

        # Most recent model instance and results instance for each variable.
        self.models = {}
        self.results = {}

        # Map from variable names to the conditional formula.
        self.conditional_formula = {}

        # Map from variable names to init/fit args of the conditional
        # models.
        self.init_kwds = defaultdict(dict)
        self.fit_kwds = defaultdict(dict)

        # Map from variable names to the model class.
        self.model_class = {}

        # Map from variable names to most recent params update.
        self.params = {}

        # Set default imputers.
        for vname in range(self.ncols):
            self.set_imputer(vname)

        # The order in which variables are imputed in each cycle.
        # Impute variables with the fewest missing values first.
        vnames = list(range(self.ncols))
        nmiss = [len(self.ix_miss[v]) for v in vnames]
        nmiss = np.asarray(nmiss)
        ii = np.argsort(nmiss)
        ii = ii[sum(nmiss == 0):]
        self._cycle_order = [vnames[i] for i in ii]

        self._initial_imputation()

        self.k_pmm = k_pmm

        # set random seed
        np.random.seed(12345)

    def next_sample(self):
        """
        Returns the next imputed dataset in the imputation process.

        Returns
        -------
        data : array_like
            An imputed dataset from the MICE chain.

        Notes
        -----
        `MICEData` does not have a `skip` parameter.  Consecutive
        values returned by `next_sample` are immediately consecutive
        in the imputation chain.

        The returned value is a reference to the data attribute of
        the class and should be copied before making any changes.
        """

        self.update_all(1)
        return self.data

    def _initial_imputation(self):
        """
        Use a PMM-like procedure for initial imputed values.

        For each variable, missing values are imputed as the observed
        value that is closest to the mean over all observed values.
        """
        # Changed for pandas 2.0 copy-on-write behavior to use a single
        # in-place fill
        # change to code where self.data is np.ndarray
        imp_values = np.zeros((self.nrows, self.ncols))
        for i in range(self.ncols):
            di = self.data[:, i] - np.nanmean(self.data[:, i])
            di = np.abs(di)
            ix = np.nanargmin(di)
            imp_values[:, i] = self.data[ix, i]
        self.data = np.where(np.isnan(self.data), imp_values, self.data)

    def _split_indices(self, vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        null = np.isnan(vec)
        ix_obs = np.flatnonzero(~null)
        ix_miss = np.flatnonzero(null)
        if len(ix_obs) == 0:
            raise ValueError("variable to be imputed has no observed values")
        return ix_obs, ix_miss

    def set_imputer(self, endog_name, formula=None, model_class=None,
                    init_kwds=None, fit_kwds=None, predict_kwds=None,
                    k_pmm=20, perturbation_method=None, regularized=False):
        """
        Specify the imputation process for a single variable.

        Parameters
        ----------
        endog_name : str
            Name of the variable to be imputed.
        formula : str
            Conditional formula for imputation. Defaults to a formula
            with main effects for all other variables in dataset.  The
            formula should only include an expression for the mean
            structure, e.g. use 'x1 + x2' not 'x4 ~ x1 + x2'.
        model_class : statsmodels model
            Conditional model for imputation. Defaults to OLS.  See below
            for more information.
        init_kwds : dit-like
            Keyword arguments passed to the model init method.
        fit_kwds : dict-like
            Keyword arguments passed to the model fit method.
        predict_kwds : dict-like
            Keyword arguments passed to the model predict method.
        k_pmm : int
            Determines number of neighboring observations from which
            to randomly sample when using predictive mean matching.
        perturbation_method : str
            Either 'gaussian' or 'bootstrap'. Determines the method
            for perturbing parameters in the imputation model.  If
            None, uses the default specified at class initialization.
        regularized : dict
            If regularized[name]=True, `fit_regularized` rather than
            `fit` is called when fitting imputation models for this
            variable.  When regularized[name]=True for any variable,
            perturbation_method must be set to boot.

        Notes
        -----
        The model class must meet the following conditions:
            * A model must have a 'fit' method that returns an object.
            * The object returned from `fit` must have a `params` attribute
              that is an array-like object.
            * The object returned from `fit` must have a cov_params method
              that returns a square array-like object.
            * The model must have a `predict` method.
        """

        if formula is None:
            main_effects = [x for x in range(self.ncols) if x != endog_name]
            # fml = endog_name + " ~ " + " + ".join(main_effects)
            self.conditional_formula[endog_name] = (endog_name, main_effects)
        else:
            raise NotImplementedError
            # fml = endog_name + " ~ " + formula
            # self.conditional_formula[endog_name] = fml

        if model_class is None:
            self.model_class[endog_name] = OLS
        else:
            self.model_class[endog_name] = model_class

        if init_kwds is not None:
            self.init_kwds[endog_name] = init_kwds

        if fit_kwds is not None:
            self.fit_kwds[endog_name] = fit_kwds

        if predict_kwds is not None:
            self.predict_kwds[endog_name] = predict_kwds

        if perturbation_method is not None:
            self.perturbation_method[endog_name] = perturbation_method

        self.k_pmm = k_pmm
        self.regularized[endog_name] = regularized

    def _store_changes(self, col, vals):
        """
        Fill in dataset with imputed values.

        Parameters
        ----------
        col : str
            Name of variable to be filled in.
        vals : ndarray
            Array of imputed values to use for filling-in missing values.
        """

        ix = self.ix_miss[col]
        if len(ix) > 0:
            self.data[ix, col] = vals
            # self.data.iloc[ix, self.data.columns.get_loc(col)] = np.atleast_1d(vals)

    def update_all(self, n_iter=1):
        """
        Perform a specified number of MICE iterations.

        Parameters
        ----------
        n_iter : int
            The number of updates to perform.  Only the result of the
            final update will be available.

        Notes
        -----
        The imputed values are stored in the class attribute `self.data`.
        """

        for k in range(n_iter):
            for vname in self._cycle_order:
                self.update(vname)

        if self.history_callback is not None:
            hv = self.history_callback(self)
            self.history.append(hv)

    def get_split_data(self, vname: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict, dict]:
        """
        Return endog and exog for imputation of a given variable.

        Parameters
        ----------
        vname : str
           The variable for which the split data is returned.

        Returns
        -------
        endog_obs : DataFrame
            Observed values of the variable to be imputed.
        exog_obs : DataFrame
            Current values of the predictors where the variable to be
            imputed is observed.
        exog_miss : DataFrame
            Current values of the predictors where the variable to be
            Imputed is missing.
        init_kwds : dict-like
            The init keyword arguments for `vname`, processed through Patsy
            as required.
        fit_kwds : dict-like
            The fit keyword arguments for `vname`, processed through Patsy
            as required.
        """

        endog, exog = self.conditional_formula[vname]
        # add constant for exog in the first column

        # endog, exog = patsy.dmatrices(formula, self.data,
        #                               return_type="dataframe")


        # Rows with observed endog
        ixo = self.ix_obs[vname]
        endog_obs = self.data[:, [endog]]
        exog_obs = self.data[:, exog]
        endog_obs = np.require(endog_obs[ixo, :], requirements="W")
        exog_obs = np.require(exog_obs[ixo, :], requirements="W")
        exog_obs = np.insert(exog_obs, 0, 1, axis=1)

        # endog_obs = np.require(endog.iloc[ixo], requirements="W")
        # exog_obs = np.require(exog.iloc[ixo, :], requirements="W")

        # Rows with missing endog
        ixm = self.ix_miss[vname]
        # exog_miss = np.require(exog.iloc[ixm, :], requirements="W")
        exog_miss = self.data[:, exog]
        exog_miss = np.require(exog_miss[ixm, :], requirements="W")
        exog_miss = np.insert(exog_miss, 0, 1, axis=1)

        predict_obs_kwds = {}
        # if vname in self.predict_kwds:
        #     kwds = self.predict_kwds[vname]
        #     predict_obs_kwds = self._process_kwds(kwds, ixo)

        predict_miss_kwds = {}
        # if vname in self.predict_kwds:
        #     kwds = self.predict_kwds[vname]
        #     predict_miss_kwds = self._process_kwds(kwds, ixo)

        return (endog_obs, exog_obs, exog_miss, predict_obs_kwds,
                predict_miss_kwds)

    def _process_kwds(self, kwds, ix):
        kwds = kwds.copy()
        # for k in kwds:
        #     v = kwds[k]
        #     if isinstance(v, PatsyFormula):
        #         mat = patsy.dmatrix(v.formula, self.data,
        #                             return_type="dataframe")
        #         mat = np.require(mat, requirements="W")[ix, :]
        #         if mat.shape[1] == 1:
        #             mat = mat[:, 0]
        #         kwds[k] = mat
        return kwds

    def get_fitting_data(self, vname):
        """
        Return the data needed to fit a model for imputation.

        The data is used to impute variable `vname`, and therefore
        only includes cases for which `vname` is observed.

        Values of type `PatsyFormula` in `init_kwds` or `fit_kwds` are
        processed through Patsy and subset to align with the model's
        endog and exog.

        Parameters
        ----------
        vname : str
           The variable for which the fitting data is returned.

        Returns
        -------
        endog : DataFrame
            Observed values of `vname`.
        exog : DataFrame
            Regression design matrix for imputing `vname`.
        init_kwds : dict-like
            The init keyword arguments for `vname`, processed through Patsy
            as required.
        fit_kwds : dict-like
            The fit keyword arguments for `vname`, processed through Patsy
            as required.
        """

        # Rows with observed endog
        ix = self.ix_obs[vname]

        # formula = self.conditional_formula[vname]
        # endog, exog = patsy.dmatrices(formula, self.data,
        #                               return_type="dataframe")

        # endog = np.require(endog.iloc[ix, 0], requirements="W")
        # exog = np.require(exog.iloc[ix, :], requirements="W")

        endog, exog = self.conditional_formula[vname]
        
        endog = np.require(self.data[ix, :][:, endog], requirements="W")
        exog = np.require(self.data[ix, :][:, exog], requirements="W")
        exog = np.insert(exog, 0, 1, axis=1)

        init_kwds = self._process_kwds(self.init_kwds[vname], ix)
        fit_kwds = self._process_kwds(self.fit_kwds[vname], ix)

        return endog, exog, init_kwds, fit_kwds


    def _boot_kwds(self, kwds, rix):
        # Try to identify any auxiliary arrays (e.g. status vector in
        # (PHReg) that need to be bootstrapped along with exog and endog.
        for k in kwds:
            v = kwds[k]

            # This is only relevant for ndarrays
            if not isinstance(v, np.ndarray):
                continue

            # Handle 1d vectors
            if (v.ndim == 1) and (v.shape[0] == len(rix)):
                kwds[k] = v[rix]

            # Handle 2d arrays
            if (v.ndim == 2) and (v.shape[0] == len(rix)):
                kwds[k] = v[rix, :]

        return kwds

    def _perturb_bootstrap(self, vname):
        """
        Perturbs the model's parameters using a bootstrap.
        """

        endog, exog, init_kwds, fit_kwds = self.get_fitting_data(vname)

        m = len(endog)
        rix = np.random.randint(0, m, m)
        endog = endog[rix]
        exog = exog[rix, :]

        init_kwds = self._boot_kwds(init_kwds, rix)
        fit_kwds = self._boot_kwds(fit_kwds, rix)

        klass = self.model_class[vname]
        self.models[vname] = klass(endog, exog, **init_kwds)

        if vname in self.regularized and self.regularized[vname]:
            self.results[vname] = (
                self.models[vname].fit_regularized(**fit_kwds))
        else:
            self.results[vname] = self.models[vname].fit(**fit_kwds)

        self.params[vname] = self.results[vname].params

    def _perturb_gaussian(self, vname):
        """
        Gaussian perturbation of model parameters.

        The normal approximation to the sampling distribution of the
        parameter estimates is used to define the mean and covariance
        structure of the perturbation distribution.
        """

        endog, exog, init_kwds, fit_kwds = self.get_fitting_data(vname)

        klass = self.model_class[vname]
        self.models[vname] = klass(endog, exog, **init_kwds)
        self.results[vname] = self.models[vname].fit(**fit_kwds)

        cov = self.results[vname].cov_params()
        mu = self.results[vname].params
        self.params[vname] = np.random.multivariate_normal(mean=mu, cov=cov)

    def perturb_params(self, vname):

        if self.perturbation_method[vname] == "gaussian":
            self._perturb_gaussian(vname)
        elif self.perturbation_method[vname] == "boot":
            self._perturb_bootstrap(vname)
        else:
            raise ValueError("unknown perturbation method")

    def impute(self, vname):
        # Wrap this in case we later add additional imputation
        # methods.
        self.impute_pmm(vname)

    def update(self, vname):
        """
        Impute missing values for a single variable.

        This is a two-step process in which first the parameters are
        perturbed, then the missing values are re-imputed.

        Parameters
        ----------
        vname : str
            The name of the variable to be updated.
        """

        self.perturb_params(vname)
        self.impute(vname)

    # work-around for inconsistent predict return values
    def _get_predicted(self, obj):

        if isinstance(obj, np.ndarray):
            return obj
        elif hasattr(obj, 'predicted_values'):
            return obj.predicted_values
        else:
            raise ValueError(
                "cannot obtain predicted values from %s" % obj.__class__)

    def impute_pmm(self, vname):
        """
        Use predictive mean matching to impute missing values.

        Notes
        -----
        The `perturb_params` method must be called first to define the
        model.
        """

        k_pmm = self.k_pmm

        endog_obs, exog_obs, exog_miss, predict_obs_kwds, predict_miss_kwds = (
            self.get_split_data(vname))

        # Predict imputed variable for both missing and non-missing
        # observations
        model = self.models[vname]
        pendog_obs = model.predict(self.params[vname], exog_obs,
                                   **predict_obs_kwds)
        pendog_miss = model.predict(self.params[vname], exog_miss,
                                    **predict_miss_kwds)

        pendog_obs = self._get_predicted(pendog_obs)
        pendog_miss = self._get_predicted(pendog_miss)

        # Jointly sort the observed and predicted endog values for the
        # cases with observed values.
        ii = np.argsort(pendog_obs)
        endog_obs = endog_obs[ii]
        pendog_obs = pendog_obs[ii]

        # Find the closest match to the predicted endog values for
        # cases with missing endog values.
        ix = np.searchsorted(pendog_obs, pendog_miss)

        # Get the indices for the closest k_pmm values on
        # either side of the closest index.
        ixm = ix[:, None] + np.arange(-k_pmm, k_pmm)[None, :]

        # Account for boundary effects
        msk = np.nonzero((ixm < 0) | (ixm > len(endog_obs) - 1))
        ixm = np.clip(ixm, 0, len(endog_obs) - 1)

        # Get the distances
        dx = pendog_miss[:, None] - pendog_obs[ixm]
        dx = np.abs(dx)
        dx[msk] = np.inf

        # Closest positions in ix, row-wise.
        dxi = np.argsort(dx, 1)[:, 0:k_pmm]

        # Choose a column for each row.
        ir = np.random.randint(0, k_pmm, len(pendog_miss))

        # Unwind the indices
        jj = np.arange(dxi.shape[0])
        ix = dxi[(jj, ir)]
        iz = ixm[(jj, ix)]

        imputed_miss = np.array(endog_obs[iz]).squeeze()
        self._store_changes(vname, imputed_miss)
