""" Commonly used functions."""

import xarray as xr
import numpy as np
import scipy.signal
import scipy.stats

def interp_coords(var: xr.DataArray, val: float, dim: str):
    """Interpolate coordinate linearly on data value.
    
    Searches for the first occurence where `val` is in between
    two data points and retuns the coordinate of `val` infered by
    linear linterpolation between those two data points.
    
    Arguments
    ---------
    var: xarray.DataArray
        Data to use as abscissa in interpolation
    val: float
        Value to interpolate to
    dim: str
        Name of the coordinate to use as ordinate. Must be 1D.
    """
    return xr.apply_ufunc(
        _interp_coords_np,
        var[dim], var, val,
        input_core_dims=[[dim], [dim], []],
        output_core_dims=[[]],
        dask='parallelized',
        output_dtypes=[var.dtype]
    )


def _interp_coords_np(y, x, x0):
    """Interpolate coordinate linearly on data value."""
    mask = (x > x0).astype(int)

    threshold_crossing = np.abs(np.diff(mask, axis=-1))

    # only account for first threshold crossing
    cs = (
        np.cumsum(threshold_crossing, axis=-1)
        * np.arange(
            1, threshold_crossing.shape[-1] + 1,
            dtype=int
        )
    )
    # take care about locations where there is no crossing
    cs[np.all(cs==0, axis=-1)] = 1
    cs = cs * np.arange(1, threshold_crossing.shape[-1] + 1, dtype=int)
    ind = np.where(cs != 0, cs, threshold_crossing.shape[-1]**2+1)
    bool_index = (
        ind == np.min(ind, axis=-1)[..., np.newaxis]
    )
    
    is_crossing = (threshold_crossing.max(axis=-1) != 0)

    x1 = x[..., :-1][bool_index]
    x2 = x[..., 1:][bool_index]
    
    y1 = np.broadcast_to(y, x.shape)[..., :-1][bool_index]
    y2 = np.broadcast_to(y, x.shape)[..., 1:][bool_index]
    
    dy = y2 - y1
    dx = x2 - x1

    y_interp = y1 + (x0 - x1) * dy / np.where(dx != 0, dx, 1.)
    y_interp = y_interp.reshape(x.shape[:-1])

    # mask slices where target value does not occur
    y_interp = np.where(
        is_crossing, y_interp, np.nan
    )
    return y_interp


def detrend(a, dim="time", **kwargs):
    """Detrend `a` along given dimension.
    
    If there is any Nan value within a 1D slice along that dimension,
    the output for that slice will be set to all NaNs. Attributes of
    the input DataArray will be preserved.
    
    Any other keyword arguments are forwarded to `xarray.apply_ufunc`.
    """
    out = xr.where(
        (~a.isnull()).all(dim),
        xr.apply_ufunc(
            scipy.signal.detrend,
            a,
            input_core_dims=[[dim]], output_core_dims=[[dim]],
            **kwargs
        ),
        np.nan,
    )
    out.attrs.update(a.attrs)
    return out


def monthly_anomalies(a: xr.DataArray, dim: str="time"):
    """Compute monthly anomalies from seasonal cycle.
    
    Resample to monthly data and remove monthly seasonal cycle
    estimated from the full time series. The time stamps of the
    result is fixed to the 15th of each month.
    
    Arguments
    ---------
    a: xr.DataArray
        Data to process
    dim: str="time"
        Name of time dimension
    """
    monthly = a.resample(
        **{dim: "1M"}, label="left", loffset="15d"
    ).mean(dim)
    
    monthly_grouped = monthly.groupby(f"{dim}.month")
    return (monthly_grouped - monthly_grouped.mean(dim)).drop("month")


def corr_with_ttest(
    x: xr.DataArray, y: xr.DataArray, dim: str="time", sig: float=0.95,
    t_bound_kwargs: dict=None
):
    """Compute Pearson correlation of two time series and the
    rejection region for a t-test that the correlation is significantly
    different from zero at the given level of significance.
    
    The test is taking serial correlation into account by modifying the
    effective degrees of freedom according to Eq. 31 in Bretherton (1999).
    
    Arguments
    ---------
    x: xr.DataArray
        First time series
    y: xr.DataArray
        Second time series
    dim: str = "time"
        Name of dimension along which to compute correlation
    sig: float = 0.95
        Significance level of the t-test
        
    Returns
    -------
    Tuple of the form (correlation, lower, upper), where lower and upper mark
    the boundaries of the two-sided t-test rejection region.
    """
    alpha = (1 - sig) / 2.
    N = len(x[dim])
    
    if t_bound_kwargs is None:
        t_bound_kwargs = dict()
    
    corr = xr.corr(x, y, dim=dim)
    
    # estimate effective degrees of freedom
    x_lag1 = _lag1_autocorrelation(x, dim=dim)
    y_lag1 = _lag1_autocorrelation(y, dim=dim)
    N_star = N * (1 - x_lag1 * y_lag1) / (1 + x_lag1 * y_lag1)
    
    return (
        corr,
        _rho_bounds(_t_bound(N_star - 2, alpha, **t_bound_kwargs), N_star),
        _rho_bounds(_t_bound(N_star - 2, 1 - alpha, **t_bound_kwargs), N_star)
    )


def _t_bound(dof: xr.DataArray, alpha: float, dim=("lon",)):
    return xr.apply_ufunc(
        scipy.stats.t.ppf,
        alpha,
        dof,
        input_core_dims=[[],dim],
        output_core_dims=[dim],
        dask='parallelized',
        output_dtypes=[type(alpha)],
        dask_gufunc_kwargs=dict(allow_rechunk=True),
    )

def _rho_bounds(tval: xr.DataArray, n: xr.DataArray):
    return np.sqrt(
            (tval**2 / (n - 2)) / (1 + (tval**2 / (n - 2)))
    )


def _lag1_autocorrelation(x: xr.DataArray, dim="time"):
    x1 = x.isel({dim: slice(None, -1)})
    x2 = x.isel({dim: slice(1, None)})
    dummy_coord = np.arange(len(x1[dim]))
    x1 = x1.assign_coords({dim: dummy_coord})
    x2 = x2.assign_coords({dim: dummy_coord})
    return xr.corr(x1, x2, dim=dim)


def reg_slope(x: xr.DataArray, y: xr.DataArray, dim: str):
    """Compute regression slope along dim.
    
    Arguments
    ---------
    x: xarray.DataArray
        x values to use in the regression.
    y: float
        y values to use in the regression.
    dim: str
        Name of the coordinate Axis long which the regression is computed.
    """
    return xr.apply_ufunc(
        _slope,
        x, y,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[x.dtype],
        dask_gufunc_kwargs=dict(allow_rechunk=True),
    )


def _slope(x, y):
    return scipy.stats.linregress(x, y).slope


def standardize(data, dim="time"):
    """Compute z-score of data."""
    mu = data.mean(dim=dim)
    # use unbiased estimator of sample std
    sigma=data.std(dim=dim, ddof=1)
    return (data-mu)/sigma


def running_variance(ds: xr.Dataset, win_length: int, dim: str="time", detrended: bool=True, **kwargs):
    """Compute running variance over windows of length win_length.
    
    Windows are detrended by default befor the computation of variance.
    
    Arguments
    ---------
    ds: xr.Dataset
        Dataset of which the variance shall be computed.
    win_length: int
        Size of running window in units of data points along dimension dim.
    dim: str="time"
        Name of the dimension along which the variance will be computed.
    detrended: bool=True
        Whether windows shall be detrended before computing the variance.
    **kwargs:
        Additional keyword arguments are passed to utils.detrend if detrending
        the windows.
        
    Returns
    -------
    Dataset of running variance. The time coordinates mark the center of the
    window over which the variance has been computed.
    """
    ds_win = ds.rolling(**{dim: win_length, "center": True}).construct("window_dim")
    
    if detrended:
        ds_dropped = ds_win.dropna(dim)
        ds_win = detrend(ds_dropped, "window_dim", **kwargs)
    
    # use unbiased estimator of the sample variance
    ds_var = ds_win.var(dim="window_dim", ddof=1)
    return ds_var