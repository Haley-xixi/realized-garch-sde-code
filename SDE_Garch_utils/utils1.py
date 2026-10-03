# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
import datetime as dt
from numpy import ix_
from scipy import stats
import statsmodels.api as sm
from numpy.random import rand
from matplotlib import pyplot as plt
from data import distributions as dists
from collections import defaultdict
import torch

#####################################
# Helper Fn
def def_value():
    return {}

# ======================= Utilities =======================
def set_seed_all(seed: int = 1234):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import random, os as _os
        random.seed(seed)
        _os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_rv(filepath = 'data/rv.pkl'):
    """ load rv data from pickle """
    df = pd.read_pickle(filepath)
    df.index = df.index.set_levels(pd.to_datetime(df.index.levels[-1], utc=True).tz_localize(None), level=-1)
    df.index.rename(['Ticker', 'Date'], inplace=True)
    return df

def get_dataset(data, expand_dims=True, scale='train', st=dt.datetime(2004,1,1), en=dt.datetime(2022,1,1)):
    rm_ls = ['rv5', 'bv', 'medrv', 'rk_parzen', 'rsv']

    df = data.copy()

    df['log_price'] = np.log(df['close_price'])
    df.dropna(subset=['log_price'], inplace=True)

    df[rm_ls] *= (10**4)

    df['rv5_week'] = df['rv5'] + df['rv5'].shift() + df['rv5'].shift(2) + df['rv5'].shift(3) + df['rv5'].shift(4)
    df['rv5_month'] = (
        df['rv5'] + df['rv5'].shift() + df['rv5'].shift(2) + df['rv5'].shift(3) + df['rv5'].shift(4) +
        df['rv5'].shift(5) + df['rv5'].shift(6) + df['rv5'].shift(7) + df['rv5'].shift(8) + df['rv5'].shift(9) +
        df['rv5'].shift(10) + df['rv5'].shift(11) + df['rv5'].shift(12) + df['rv5'].shift(13) + df['rv5'].shift(14) +
        df['rv5'].shift(15) + df['rv5'].shift(16) + df['rv5'].shift(17) + df['rv5'].shift(18) + df['rv5'].shift(19) +
        df['rv5'].shift(20) + df['rv5'].shift(21)
    )

    df = df[st:en]
    n_all = df.shape[0]
    n_train = int(n_all * 0.8)

    if (df['rv5'] <= 0).sum() > 0:
        df.loc[df['rv5'] <= 0, 'rv5'] = df.loc[df['rv5'] > 0, 'rv5'].min()

    df['logRV'] = np.log(df['rv5'].clip(lower=1e-8))

    c = 1
    df[rm_ls] *= c

    df_train = df[:n_train]
    df_test = df[n_train:]

    if expand_dims:
        y_all = df['log_price'].to_numpy().reshape(-1,1)
        y_train = df_train['log_price'].to_numpy().reshape(-1,1)
        y_test  = df_test['log_price'].to_numpy().reshape(-1,1)

        rv_all = df['rv5'].to_numpy().reshape(-1,1)
        rv_train = df_train['rv5'].to_numpy().reshape(-1,1)
        rv_test = df_test['rv5'].to_numpy().reshape(-1,1)

        logRV_all = df['logRV'].to_numpy().reshape(-1,1)
        logRV_train = df_train['logRV'].to_numpy().reshape(-1,1)
        logRV_test = df_test['logRV'].to_numpy().reshape(-1,1)

        rv5_week_train = df_train['rv5_week'].to_numpy().reshape(-1,1)
        rv5_month_train = df_train['rv5_month'].to_numpy().reshape(-1,1)
        rv5_week_test = df_test['rv5_week'].to_numpy().reshape(-1,1)
        rv5_month_test = df_test['rv5_month'].to_numpy().reshape(-1,1)
    else:
        y_all = df['log_price'].to_numpy()
        y_train = df_train['log_price'].to_numpy()
        y_test  = df_test['log_price'].to_numpy()

        rv_all = df['rv5'].to_numpy()
        rv_train = df_train['rv5'].to_numpy()
        rv_test = df_test['rv5'].to_numpy()

        logRV_all = df['logRV'].to_numpy()
        logRV_train = df_train['logRV'].to_numpy()
        logRV_test = df_test['logRV'].to_numpy()

        rv5_week_train = df_train['rv5_week'].to_numpy()
        rv5_month_train = df_train['rv5_month'].to_numpy()
        rv5_week_test = df_test['rv5_week'].to_numpy()
        rv5_month_test = df_test['rv5_month'].to_numpy()

    rm_test = df_test[rm_ls]
    close_test = df_test['close_price'].to_numpy()

    return {
        'y_train': y_train,
        'y_test': y_test,
        'rv_train': rv_train,
        'rv_test': rv_test,
        'logRV_train': logRV_train,
        'logRV_test': logRV_test,
        'y_all': y_all,
        'rv_all': rv_all,
        'logRV_all': logRV_all,
        'rm_test': rm_test,
        'date': df.index,
        'rv5_week_train': rv5_week_train,
        'rv5_month_train': rv5_month_train,
        'rv5_week_test': rv5_week_test,
        'rv5_month_test': rv5_month_test,
        'close_test': close_test
    }

def compute_rv_scale(
    returns_train,
    rv_train,
    eps=1e-12,
):
    """
    Scale realized variance to match the in-sample
    unconditional variance level of returns:

        c_hat = sum(y_t^2) / sum(RV_t)

    Training data only.
    """
    returns_train = np.asarray(
        returns_train,
        dtype=np.float64,
    ).reshape(-1)

    rv_train = np.asarray(
        rv_train,
        dtype=np.float64,
    ).reshape(-1)

    mask = (
        np.isfinite(returns_train)
        & np.isfinite(rv_train)
    )

    returns_train = returns_train[mask]
    rv_train = rv_train[mask]

    if len(returns_train) == 0:
        raise ValueError(
            "No valid observations available "
            "for RV scaling."
        )

    numerator = np.sum(
        returns_train ** 2
    )

    denominator = np.sum(
        rv_train
    )

    if denominator <= eps:
        raise ValueError(
            "Sum of training RV is too small "
            "to compute the scaling factor."
        )

    return float(
        numerator / denominator
    )


def scale_realized_variance(
    rv,
    scale_factor,
):
    """
    Apply a pre-computed scaling factor to RV.
    """
    rv = np.asarray(
        rv,
        dtype=np.float64,
    )

    return scale_factor * rv

def build_data_dict(df, scale='train'):
    """ build dataset dict for all tickers"""
    dataset_dict = defaultdict(def_value)
    ticker_ls = df.index.levels[0]
    for ticker in ticker_ls:
        ticker_ = ticker[1:]
        dataset_dict[ticker_]['train'] = get_dataset(df.loc[ticker], expand_dims=True, scale=scale)
        dataset_dict[ticker_]['test'] = get_dataset(df.loc[ticker], expand_dims=False, scale=scale)
    return dataset_dict

def get_dataset_opttrading(data, st=dt.datetime(2004,1,1), en=dt.datetime(2022,1,1)):

    data =  data.copy()
    data['log_price'] = np.log(data['close_price'])
    data = data[st:en]

    n_all = data.shape[0]
    n_train = int(n_all/2)

    data_train = data[:n_train]
    data_test = data[n_train:]

    y_test  = data_test['log_price'].to_numpy()

    return y_test

#####################################
# Stat Fn
def get_stat(smc, data_dict, alpha1=0.05, alpha2=0.01, rv=False):

    y_train, y_test, rv_train, rv_test = data_dict['test']['y_train'], data_dict['test']['y_test'], data_dict['test']['rv_train'], data_dict['test']['rv_test']

    if rv:
        var_train = np.sqrt(rv_train)
        var_test  = np.sqrt(rv_test)
    else:
        var_train = np.sqrt(smc.pre.var_ls)
        var_test = np.sqrt(smc.var_ls)

    # align
    y_train = y_train[:var_train.shape[0]]
    y_test = y_test[:var_test.shape[0]]

    stat = {}
    #
    stat['train','var','mean'] = var_train.mean()
    stat['train','var','std']  = var_train.std()
    stat['train','var','skew'] = stats.skew(var_train)
    stat['train','var','kurt'] = stats.kurtosis(var_train)
    stat['train','stat','PPS'] = -np.average(stats.norm.logpdf(y_train, loc=0, scale = var_train))
    stat['train','stat','Mar.llik'] = 0 if rv else get_mllik(smc.pre)

    # test
    stat['test','var','mean'] = var_test.mean()
    stat['test','var','std']  = var_test.std()
    stat['test','var','skew'] = stats.skew(var_test)
    stat['test','var','kurt'] = stats.kurtosis(var_test)
    stat['test','stat','PPS'] = -np.average(stats.norm.logpdf(y_test, loc=0, scale=var_test))

    var, _ = stats.norm.interval(1-alpha2*2, loc=0, scale=var_test)
    es = y_test.mean() + var_test * esn(alpha2)
    stat['test', 'stat', 'Vrate_1%'], stat['test', 'stat', 'Qloss_1%'] = check_var_fc(var, y_test, alpha2)
    stat['test', 'stat', 'JointLoss_1%'] = jointloss(es, var, y_test, alpha2)

    var, _ = stats.norm.interval(1-alpha1*2, loc=0, scale=var_test)
    es = y_test.mean() + var_test * esn(alpha1)
    stat['test', 'stat', 'Vrate_5%'], stat['test', 'stat', 'Qloss_5%'] = check_var_fc(var, y_test, alpha1)
    stat['test', 'stat', 'JointLoss_5%'] = jointloss(es, var, y_test, alpha1)

    return stat

def get_stat_mse(var, rm, vol=False, type='mse'):

    # align
    if rm.shape[0] != var.shape[0]:
        rm  = rm[1:]

    if vol:
        var = np.sqrt(var)
        rm  = np.sqrt(rm)

    if type == 'mse':
        return np.square(rm-var).mean(axis=0)
    elif type == 'rmse':
        return np.sqrt(np.square(rm-var).mean(axis=0))
    elif type == 'mad':
        return np.abs(rm-var).mean(axis=0)
#####################################
# VaR Fn
# ES residuals
def es(r, p):
    var = np.quantile(r, p)
    return np.mean(r[r < var])

def esn(p): # Theoretical ES value
    ninv = stats.norm.ppf(p)
    return -stats.norm.pdf(ninv) / p

def es_resid(es, var, s, r):
    xi = r[r < var] - es[r < var]
    return (xi, (xi / s[r < var]))

def ttest(x, mu):
    n = len(x)
    xbar = np.mean(x)
    s = np.std(x, ddof=1)
    t = (xbar - mu) / (s / np.sqrt(n))
    pval = 2 * stats.t.sf(np.abs(t), df=(n - 1))
    return pval, t

# Unconditional coverage test
def uctest(hit, a):
    n = len(hit)
    p = np.sum(hit) / n
    z = (p - a) / np.sqrt(a * (1 - a) / n)
    pval = 2 * stats.norm.sf(np.abs(z))
    return pval, p

# Independence test
def indtest(hits):
    n = len(hits)

    r5 = hits[1:]
    r51 = hits[:-1]
    i11 = r5*r51
    i01 = r5*(1-r51)
    i10 = (1-r5)*r51
    i00 = (1-r5)*(1-r51)

    t00 = np.sum(i00)
    t01 = np.sum(i01)
    t10 = np.sum(i10)
    t11 = np.sum(i11)
    p01 = t01/(t00+t01)
    p11 = t11/(t10+t11)
    p1 = (t01+t11)/n

    ll1 = t00 * np.log(1-p01) + (p01>0) * t01 * np.log(p01) + t10 * np.log(1-p11)
    if p11>0:
        ll1=ll1+t11*np.log(p11)

    ll0=(t10+t00)*np.log(1-p1)+(t01+t11)*np.log(p1)

    lrind=2*(ll1-ll0)
    pcc=1-stats.chi2.cdf(lrind,1)
    return pcc, lrind

# Dynamic quantile test
def dqtest(y,f,a,lag):
    # f
    n = len(y)
    hits = ((y<f)*1)*(1-a)
    hits = (hits)*1+(y>f)*(-a)
    q=2+lag

    if np.sum((y<f)*1) > 0:
        ns = n - lag
        xmat = np.column_stack([np.ones((ns,1)), f[lag:n+1]])
        for k in range(1,lag+1):
            lk = lag-k
            xmat = np.column_stack([xmat, hits[lk:n-k]])

        hx = np.dot((hits[lag:n+1]), xmat)
        xtx = np.linalg.lstsq(np.matmul(xmat.T, xmat), np.eye(q), rcond = None)[0]
        dq = np.dot(np.dot(hx, xtx), hx.T)
        dq = dq/(a*(1-a))
        pdq = 1 - stats.chi2.cdf(dq,q)
    else:
        pdq = np.nan
        dq = np.nan
    return pdq, dq

# Quantile loss function
def qloss(q,r,p):
    # q->var, r->y_test, p->alpha
    q = np.array(q)
    x1 = r[r > q]
    x2 = r[r < q]
    f1 = q[r > q]
    f2 = q[r < q]
    l = p * np.sum(x1-f1) + (1-p) * np.sum(f2-x2)
    return l

def qloss_series(var, y_test, alpha):
    # q->var, r->y_test, p->alpha
    return (y_test - var)*(alpha-(y_test<=var))

# Joint loss function
def jointloss(es,q,r,p):
    m = len(r)
    q = np.array(q)
    es = np.array(es)
    i1 = (r < q).astype(int)
    aes = es ** (-1) * (p-1)
    ees = (r-q) * (p - i1)
    l =  np.sum(-np.log(aes)) - np.sum(ees / es) / p
    l = l / m
    return l

def jointloss_series(es,q,r,p):
    m = len(r)
    q = np.array(q)
    es = np.array(es)
    i1 = (r < q).astype(int)
    aes = es ** (-1) * (p-1)
    ees = (r-q) * (p - i1)
    l =  -np.log(aes) - ees / es / p
    return l

# Accuracy checks for VaR
def check_var_fc(var_fc, r, p):
    hit = r < var_fc
    n_hit = np.sum(hit)
    pval_uc, p_hat = uctest(hit, p)
    # pval_ind = indtest(hit)[0]
    # pval_dq = dqtest(r, var_fc, p, 4)[0]
    qtl_loss = qloss(var_fc, r, p)
    return [p_hat, qtl_loss]

# Accuracy checks for ES
def check_es_fc(es, var, s, r):
    # s -> sigma
    hit = r < var
    n_hit = np.sum(hit)
    xi, xis = es_resid(es, var, s, r)
    t_xi = ttest(xi, 0)[1]
    t_xis = ttest(xis, 0)[1]
    p_xis = ttest(xis, 0)[0]
    return [n_hit, np.mean(xi), t_xi, np.mean(xis), t_xis, p_xis]

# More accuracy checks for ES - 1 day
def check_es_fc_ex(es, var, s, r, p):
    # s->sigma
    xi = es_resid(es, var, s, r)[0]
    rmse = np.sqrt(np.mean(xi ** 2))
    mad = np.mean(np.abs(xi))
    jloss = jointloss(es, var, r, p)
    lst = check_var_fc(es, r, p)
    lst.append(jloss)
    lst.append(rmse)
    lst.append(mad)
    return lst

def check_es_fc_ex_10(es, var, s, r, p):
    hit = r < var
    n_hit = np.sum(hit)
    xi, xis = es_resid(es, var, s, r)
    rmse = np.sqrt(np.mean(xi ** 2))
    mad = np.mean(np.abs(xi))
    t_xi = ttest(xi, 0)[1]
    t_xis = ttest(xis, 0)[1]
    p_xis = ttest(xis, 0)[0]
    jl = jointloss(es, var, r, p)
    return [n_hit, np.mean(xi), t_xi, np.mean(xis), t_xis, p_xis, jl, rmse, mad]

def get_stat_rv(inst_key, data_dict_test, scale='all'):

    Y_train, Y_test, RV_train, RV_test,_ = data_dict_test[inst_key]

    if scale == 'all':
        c = (np.square(Y_train).mean()+np.square(Y_test).mean())/(RV_train.mean()+RV_test.mean())
    elif scale == 'train':
        c = (np.square(Y_train).mean())/(RV_train.mean())
    elif scale == 'test':
        c = (np.square(Y_test).mean())/(RV_test.mean())
    else:
        c = 1

    Y = Y_test[1:]
    RV = RV_test[1:]*c
    var = RV
    res = Y/np.sqrt(var)

    Y_train = Y_train[1:]
    RV_train = RV_train[1:]*c
    var_train = RV_train
    res_train = Y_train/np.sqrt(var_train)
    stat = {}

    # train
    stat['train','var','mean'] = var_train.mean()
    stat['train','var','std']  = var_train.std()
    stat['train','var','skew'] = stats.skew(var_train)
    stat['train','var','kurt'] = stats.kurtosis(var_train)
    stat['train','res','std'] = res_train.std()
    stat['train','res','skew'] = stats.skew(res_train)
    stat['train','res','kurt'] = stats.kurtosis(res_train)
    stat['train','res','kurt'] = stats.kurtosis(res_train)
    stat['train','res','lb'] = sm.stats.acorr_ljungbox(res_train, lags=[10], return_df=True).to_numpy()[0][1]

    stat['train','stat','mse1'] = 1
    stat['train','stat','mse2'] = 1
    stat['train','stat','qlike'] = 1
    stat['train','stat','PPS'] = -np.average(stats.norm.logpdf(Y_train, loc=0, scale = np.sqrt(var_train)))
    stat['train','stat','Mar.llik'] = 1

    # test
    stat['test','var','mean'] = var.mean()
    stat['test','var','std']  = var.std()
    stat['test','var','skew'] = stats.skew(var)
    stat['test','var','kurt'] = stats.kurtosis(var)
    stat['test','res','std'] = res.std()
    stat['test','res','skew'] = stats.skew(res)
    stat['test','res','kurt'] = stats.kurtosis(res)
    stat['test','res','lb'] = sm.stats.acorr_ljungbox(res, lags=[10], return_df=True).to_numpy()[0][1]

    stat['test','stat','mse1'] = 1
    stat['test','stat','mse2'] = 1
    stat['test','stat','qlike'] = 1
    stat['test','stat','PPS'] = -np.average(stats.norm.logpdf(Y, loc=0, scale = np.sqrt(var)))
    return stat

def get_stat_2(smc, inst_key, data_dict_test):

    Y_train, Y_test, RV_train, RV_test,_ = data_dict_test[inst_key]
    Y = Y_test[1:]
    RV = RV_test[1:,1]
    var = smc.var_ls[1:]
    res = Y/np.sqrt(var)

    Y_train = Y_train[1:]
    RV_train = RV_train[1:,1]
    var_train = smc.pre.var_ls
    res_train = Y_train/np.sqrt(var_train)
    stat = {}

    # train
    stat['train','var','mean'] = var_train.mean()
    stat['train','var','std']  = var_train.std()
    stat['train','var','skew'] = stats.skew(var_train)
    stat['train','var','kurt'] = stats.kurtosis(var_train)
    stat['train','res','std'] = res_train.std()
    stat['train','res','skew'] = stats.skew(res_train)
    stat['train','res','kurt'] = stats.kurtosis(res_train)
    stat['train','res','kurt'] = stats.kurtosis(res_train)
    stat['train','res','lb'] = sm.stats.acorr_ljungbox(res_train, lags=[10], return_df=True).to_numpy()[0][1]

    stat['train','stat','mse1'] = np.average(np.square(var_train-RV_train))
    stat['train','stat','mse2'] = np.average(np.square(np.sqrt(var_train)-np.sqrt(RV_train)))
    stat['train','stat','qlike'] = np.average(np.log(var_train)+RV_train/var_train)
    stat['train','stat','llik'] = -np.average(stats.norm.logpdf(Y_train, loc=0, scale = np.sqrt(var_train)))
    stat['train','stat','mllik'] = get_mllik(smc.pre)

    # test
    stat['test','var','mean'] = var.mean()
    stat['test','var','std']  = var.std()
    stat['test','var','skew'] = stats.skew(var)
    stat['test','var','kurt'] = stats.kurtosis(var)
    stat['test','res','std'] = res.std()
    stat['test','res','skew'] = stats.skew(res)
    stat['test','res','kurt'] = stats.kurtosis(res)
    stat['test','res','lb'] = sm.stats.acorr_ljungbox(res, lags=[10], return_df=True).to_numpy()[0][1]

    stat['test','stat','mse1'] = np.average(np.square(var-RV))
    stat['test','stat','mse2'] = np.average(np.square(np.sqrt(var)-np.sqrt(RV)))
    stat['test','stat','qlike'] = np.average(np.log(var)+RV/var)
    stat['test','stat','llik'] = -np.average(stats.norm.logpdf(Y, loc=0, scale = np.sqrt(var)))
    return stat

def get_stat_norv(smc, data):

    Y_train, Y_test = data
    Y = Y_test[1:]
    var = smc.var_ls[1:]
    res = Y/np.sqrt(var)

    Y_train = Y_train[1:]
    var_train = smc.pre.var_ls
    res_train = Y_train/np.sqrt(var_train)
    stat = {}

    # train
    stat['train','var','mean'] = var_train.mean()
    stat['train','var','std']  = var_train.std()
    stat['train','var','skew'] = stats.skew(var_train)
    stat['train','var','kurt'] = stats.kurtosis(var_train)
    stat['train','res','std'] = res_train.std()
    stat['train','res','skew'] = stats.skew(res_train)
    stat['train','res','kurt'] = stats.kurtosis(res_train)
    stat['train','res','kurt'] = stats.kurtosis(res_train)
    stat['train','res','lb'] = sm.stats.acorr_ljungbox(res_train, lags=[10], return_df=True).to_numpy()[0][1]

    stat['train','stat','llik'] = -np.average(stats.norm.logpdf(Y_train, loc=0, scale = np.sqrt(var_train)))
    stat['train','stat','mllik'] = get_mllik(smc.pre)

    # test
    stat['test','var','mean'] = var.mean()
    stat['test','var','std']  = var.std()
    stat['test','var','skew'] = stats.skew(var)
    stat['test','var','kurt'] = stats.kurtosis(var)
    stat['test','res','std'] = res.std()
    stat['test','res','skew'] = stats.skew(res)
    stat['test','res','kurt'] = stats.kurtosis(res)
    stat['test','res','lb'] = sm.stats.acorr_ljungbox(res, lags=[10], return_df=True).to_numpy()[0][1]
    stat['test','stat','llik'] = -np.average(stats.norm.logpdf(Y, loc=0, scale = np.sqrt(var)))

    alpha = 0.05
    lower_a, upper_a = stats.norm.interval(1-alpha, loc=0, scale=np.sqrt(var))
    lower_a2, upper_a2 = stats.norm.interval(1-alpha*2, loc=0, scale=np.sqrt(var))
    stat['test','stat','#vio'] = np.sum(((Y<lower_a)|(Y>upper_a)))
    stat['test','stat','QS'] = np.average((Y-lower_a2)*(alpha-(Y<lower_a2)))
    stat['test','stat','pct_hit'] = np.sum(Y<lower_a2)/Y.shape[0]
    pct_hit_up = np.sum(Y>upper_a2)/Y.shape[0]

    return stat

def get_stat_orginal(smc, Y_test, RV_test, alpha=0.01):
    Y = np.squeeze(Y_test)
    RV = np.squeeze(RV_test)
    var = np.array(smc.var_ls)
    PPS = -np.average(stats.norm.logpdf(Y, loc=0, scale = np.sqrt(var)))
    lower_a, upper_a = stats.norm.interval(1-alpha, loc=0, scale=np.sqrt(var))
    num_vio = np.sum(((Y<lower_a)|(Y>upper_a)))
    lower_a2, upper_a2 = stats.norm.interval(1-alpha*2, loc=0, scale=np.sqrt(var))
    QS = np.average((Y-lower_a2)*(0.01-(Y<lower_a2)))
    pct_hit = np.sum(Y<lower_a2)/Y.shape[0]
    pct_hit_up = np.sum(Y>upper_a2)/Y.shape[0]
    mse1 = np.average(np.square(np.sqrt(var)-np.sqrt(RV)))
    mse2 = np.average(np.square(var-RV))
    mae1 = np.average(np.absolute(np.sqrt(var)-np.sqrt(RV)))
    mae2 = np.average(np.absolute(var-RV))
    qlike = np.average(np.log(RV)+var/RV)
    # print("PPS={:3f},#vio={},QS={:3f},%Hit_l={:3f},%Hit_u={:3f}".format(PPS, num_vio, QS, pct_hit, pct_hit_up))
    # print("mse1={:4f},mse2={:4f},mae1={:4f},mae2={:4f}\n".format(mse1, mse2, mae1, mae2))
    return [PPS, num_vio, QS, pct_hit, mse1, mse2, mae1, mae2]

def get_stat_t(smc, Y_test, RV_test, alpha=0.01):
    Y = np.squeeze(Y_test)
    RV = np.squeeze(RV_test)
    var = np.array(smc.var_ls)
    nu = np.array(smc.nu_ls)
    PPS = -np.average(stats.t.logpdf(Y, nu,loc=0, scale = np.sqrt(var)))
    lower_a, upper_a = stats.t.interval(1-alpha, nu,loc=0, scale=np.sqrt(var))
    num_vio = np.sum(((Y<lower_a)|(Y>upper_a)))
    lower_a2, upper_a2 = stats.t.interval(1-alpha*2, nu,loc=0, scale=np.sqrt(var))
    QS = np.average((Y-lower_a2)*(0.01-(Y<lower_a2)))
    pct_hit_l = np.sum(Y<lower_a2)/Y.shape[0]
    pct_hit_u = np.sum(Y>upper_a2)/Y.shape[0]
    mse1 = np.average(np.square(np.sqrt(var)-np.sqrt(RV)))
    mse2 = np.average(np.square(var-RV))
    mae1 = np.average(np.absolute(np.sqrt(var)-np.sqrt(RV)))
    mae2 = np.average(np.absolute(var-RV))
    qlike = np.average(np.log(RV)+var/RV)
    print("PPS={:3f},#vio={},QS={:3f},%Hit_l={:3f},%Hit_u={:3f}".format(PPS, num_vio, QS, pct_hit_l, pct_hit_u))
    print("mse1={:4f},mse2={:4f},mae1={:4f},mae2={:4f}\n".format(mse1, mse2, mae1, mae2))
    return [PPS, num_vio, QS, pct_hit_l, mse1, mse2, mae1, mae2]

def get_mllik(smc):
    # wgts.W is normalized weights, lw in unnormalised, thus we need to use lw
    # casuing -inf due to lw is -700, np.exp(lw) will give zeros and np.log(0) = -inf
    # in smc was handled by np.exp(self.lw - self.lw.max())
    # for realrech, its a sum of two likelihood function
    mllik =0
    for w in smc.wgts_ls:
        mllik += w.lw.max()+np.log(np.average(np.exp(w.lw-w.lw.max())))
    return -mllik

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return math.sqrt(variance)

def get_theta_mean(smc):
    theta = smc.X.theta
    theta_wgts = smc.wgts.W
    theta_names = theta.dtype.names
    theta_mean = np.zeros_like(theta, shape=1)
    theta_std = np.zeros_like(theta, shape=1)
    for t in theta_names:
        theta_mean[t] = np.average(smc.X.theta[t], weights=theta_wgts)
        theta_std[t] = weighted_avg_and_std(smc.X.theta[t], weights=theta_wgts)


    return pd.concat([pd.DataFrame.from_records(theta_mean),pd.DataFrame.from_records(theta_std)])

def hurst(ts):
    ts = list(ts)
    N = len(ts)
    if N < 20:
        raise ValueError("Time series is too short! input series ought to have at least 20 samples!")

    max_k = int(np.floor(N/2))
    R_S_dict = []
    for k in range(10,max_k+1):
        R,S = 0,0
        # split ts into subsets
        subset_list = [ts[i:i+k] for i in range(0,N,k)]
        if np.mod(N,k)>0:
            subset_list.pop()
            #tail = subset_list.pop()
            #subset_list[-1].extend(tail)
        # calc mean of every subset
        mean_list=[np.mean(x) for x in subset_list]
        for i in range(len(subset_list)):
            cumsum_list = pd.Series(subset_list[i]-mean_list[i]).cumsum()
            R += max(cumsum_list)-min(cumsum_list)
            S += np.std(subset_list[i])
        R_S_dict.append({"R":R/len(subset_list),"S":S/len(subset_list),"n":k})

    log_R_S = []
    log_n = []
    print(R_S_dict)
    for i in range(len(R_S_dict)):
        R_S = (R_S_dict[i]["R"]+np.spacing(1)) / (R_S_dict[i]["S"]+np.spacing(1))
        log_R_S.append(np.log(R_S))
        log_n.append(np.log(R_S_dict[i]["n"]))

    Hurst_exponent = np.polyfit(log_n,log_R_S,1)[0]
    return Hurst_exponent

#####################################
# MCS
def bootstrap_sample(data, B, w):
    '''
    Bootstrap the input data
    data: input numpy data array
    B: boostrap size
    w: block length of the boostrap
    '''
    t = len(data)
    p = 1 / w
    indices = np.zeros((t, B), dtype=int)
    indices[0, :] = np.ceil(t * rand(1, B))
    select = np.asfortranarray(rand(B, t).T < p)
    vals = np.ceil(rand(1, np.sum(np.sum(select))) * t).astype(int)
    indices_flat = indices.ravel(order="F")
    indices_flat[select.ravel(order="F")] = vals.ravel()
    indices = indices_flat.reshape([B, t]).T
    for i in range(1, t):
        indices[i, ~select[i, :]] = indices[i - 1, ~select[i, :]] + 1
    indices[indices > t] = indices[indices > t] - t
    indices -= 1
    return data[indices]

def compute_dij(losses, bsdata):
    '''Compute the loss difference'''
    t, M0 = losses.shape
    B = bsdata.shape[1]
    dijbar = np.zeros((M0, M0))
    for j in range(M0):
        dijbar[j, :] = np.mean(losses - losses[:, [j]], axis=0)

    dijbarstar = np.zeros((B, M0, M0))
    for b in range(B):
        meanworkdata = np.mean(losses[bsdata[:, b], :], axis=0)
        for j in range(M0):
            dijbarstar[b, j, :] = meanworkdata - meanworkdata[j]

    vardijbar = np.mean((dijbarstar - np.expand_dims(dijbar, 0)) ** 2, axis=0)
    vardijbar += np.eye(M0)

    return dijbar, dijbarstar, vardijbar

def calculate_PvalR(z, included, zdata0):
    '''Calculate the p-value of relative algorithm'''
    empdistTR = np.max(np.max(np.abs(z), 2), 1)
    zdata = zdata0[ix_(included - 1, included - 1)]
    TR = np.max(zdata)
    pval = np.mean(empdistTR > TR)
    return pval

def calculate_PvalSQ(z, included, zdata0):
    '''Calculate the p-value of sequential algorithm'''
    empdistTSQ = np.sum(z ** 2, axis=1).sum(axis=1) / 2
    zdata = zdata0[ix_(included - 1, included - 1)]
    TSQ = np.sum(zdata ** 2) / 2
    pval = np.mean(empdistTSQ > TSQ)
    return pval

def iterate(dijbar, dijbarstar, vardijbar, alpha, algorithm="R"):
    '''Iteratively excluding inferior model'''
    B, M0, _ = dijbarstar.shape
    z0 = (dijbarstar - np.expand_dims(dijbar, 0)) / np.sqrt(
        np.expand_dims(vardijbar, 0)
    )
    zdata0 = dijbar / np.sqrt(vardijbar)

    excludedR = np.zeros([M0, 1], dtype=int)
    pvalsR = np.ones([M0, 1])

    for i in range(M0 - 1):
        included = np.setdiff1d(np.arange(1, M0 + 1), excludedR)
        m = len(included)
        z = z0[ix_(range(B), included - 1, included - 1)]

        if algorithm == "R":
            pvalsR[i] = calculate_PvalR(z, included, zdata0)
        elif algorithm == "SQ":
            pvalsR[i] = calculate_PvalSQ(z, included, zdata0)

        scale = m / (m - 1)
        dibar = np.mean(dijbar[ix_(included - 1, included - 1)], 0) * scale
        dibstar = np.mean(dijbarstar[ix_(range(B), included - 1, included - 1)], 1) * (
            m / (m - 1)
        )
        vardi = np.mean((dibstar - dibar) ** 2, axis=0)
        t = dibar / np.sqrt(vardi)
        modeltoremove = np.argmax(t)
        excludedR[i] = included[modeltoremove]

    maxpval = pvalsR[0]
    for i in range(1, M0):
        if pvalsR[i] < maxpval:
            pvalsR[i] = maxpval
        else:
            maxpval = pvalsR[i]

    excludedR[-1] = np.setdiff1d(np.arange(1, M0 + 1), excludedR)
    pl = np.argmax(pvalsR > alpha)
    includedR = excludedR[pl:]
    excludedR = excludedR[:pl]
    return includedR - 1, excludedR - 1, pvalsR

def MCS(losses, alpha, B, w, algorithm):
    '''Main function of the MCS'''
    t, M0 = losses.shape
    bsdata = bootstrap_sample(np.arange(t), B, w)
    dijbar, dijbarstar, vardijbar = compute_dij(losses, bsdata)
    includedR, excludedR, pvalsR = iterate(
        dijbar, dijbarstar, vardijbar, alpha, algorithm=algorithm
    )
    return includedR, excludedR, pvalsR

class ModelConfidenceSet(object):
    def __init__(self, data, alpha, B, w, algorithm="SQ", names=None):
        """
        Implementation of Econometrica Paper:
        Hansen, Peter R., Asger Lunde, and James M. Nason. "The model confidence set." Econometrica 79.2 (2011): 453-497.

        Input:
            data->pandas.DataFrame or numpy.ndarray: input data, columns are the losses of each model
            alpha->float: confidence level
            B->int: bootstrap size for computation covariance
            w->int: block size for bootstrap sampling
            algorithm->str: SQ or R, SQ is the first t-statistics in Hansen (2011) p.465, and R is the second t-statistics
            names->list: the name of each model (corresponding to each columns).

        Method:
            run(self): compute the MCS procedure

        Attributes:
            included: models that are in the model confidence sets at confidence level of alpha
            excluded: models that are NOT in the model confidence sets at confidence level of alpha
            pvalues: the bootstrap p-values of each models
        """

        if isinstance(data, pd.DataFrame):
            self.data = data.values
            self.names = data.columns.values if names is None else names
        elif isinstance(data, np.ndarray):
            self.data = data
            self.names = np.arange(data.shape[1]) if names is None else names

        if alpha < 0 or alpha > 1:
            raise ValueError(
                f"alpha must be larger than zero and less than 1, found {alpha}"
            )
        if not isinstance(B, int):
            try:
                B = int(B)
            except Exception as identifier:
                raise RuntimeError(
                    f"Bootstrap size B must be a integer, fail to convert", identifier
                )
        if B < 1:
            raise ValueError(f"Bootstrap size B must be larger than 1, found {B}")
        if not isinstance(w, int):
            try:
                w = int(w)
            except Exception as identifier:
                raise RuntimeError(
                    f"Bootstrap block size w must be a integer, fail to convert",
                    identifier,
                )
        if w < 1:
            raise ValueError(f"Bootstrap block size w must be larger than 1, found {w}")

        if algorithm not in ["R", "SQ"]:
            raise TypeError(f"Only R and SQ algorithm supported, found {algorithm}")

        self.alpha = alpha
        self.B = B
        self.w = w
        self.algorithm = algorithm

    def run(self):
        included, excluded, pvals = MCS(
            self.data, self.alpha, self.B, self.w, self.algorithm
        )

        self.included = self.names[included].ravel().tolist()
        self.excluded = self.names[excluded].ravel().tolist()
        self.pvalues = pd.Series(pvals.ravel(), index=self.excluded + self.included)
        return self

#####################################
# Plot
def single_plot(smc):
    # Initialization
    theta = smc.X.theta
    theta_wgts = smc.wgts.W
    theta_names = theta.dtype.names
    theta_mean = get_theta_mean(smc)
    # Plot
    fig, ax = plt.subplots(figsize = (10,5))
    for t in theta_names:
        ax.hist(theta[t],
                label='{}={}'.format(t, theta_mean[t]),
                alpha=0.5,
                density=True,
                weights=theta_wgts)
    ax.legend()

def var_plot(var_ls=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    for var in var_ls:
        ax.plot(var, label=nameof(var))
    ax.legend()

def get_rech_var(smc, Y=None):
    rech_var = np.zeros(Y.shape[0])
    rech_var[0] = np.var(Y)
    omega = np.zeros(Y.shape[0])
    omega[0] = rech_w_mean['beta0']
    h = np.zeros(Y.shape[0])
    theta_mean = get_theta_mean(smc)

    for n in range(Y[:-1].shape[0]):
        h[n+1] = relu(theta_mean['v0'] * omega[n] + theta_mean['v1'] * Y[n] + theta_mean['v2'] * rech_var[n] + theta_mean['w'] * h[n] + theta_mean['b'])
        omega[n+1] = theta_mean['beta0'] + rech_w_mean['beta1'] * h[n+1]
        rech_var[n+1] = omega[n+1] + theta_mean['alpha'] * np.square(Y[n]) + theta_mean['beta'] * rech_var[n]
    return rech_var, omega

def get_garch_var(smc=None, Y=None):
    theta_mean = get_theta_mean(smc)
    garch_var = [np.var(Y)]
    for n, y in enumerate(Y[:-1]):
        garch_var.append(float(theta_mean['omega'] + theta_mean['alpha'] * np.square(y) + theta_mean['beta'] * garch_var[n]))
    return garch_var

#####################################
# Helper Fn
# def get_dataset(data, rvsup_ls=None, expand_dims=False, st=dt.datetime(2004,1,1), en=dt.datetime(2022,1,1), scale='None'):

#     data =  data.copy()
#     data['close_price'] = np.log(data['close_price']).diff().dropna()
#     data['close_price'] = 100*(data['close_price']-data['close_price'].mean())
#     data = data[st:en]

#     n_all = data.shape[0]
#     n_train = int(n_all/2)

#     data_train = data[:n_train]
#     data_test = data[n_train:]

#     Y_train = data_train['close_price'].to_numpy()
#     Y_test  = data_test['close_price'].to_numpy()

#     RV_train = data_train['rv5'].to_numpy()*(10**4)
#     RV_test = data_test['rv5'].to_numpy()*(10**4)

#     if scale == 'all':
#         c = (np.square(Y_train).mean()+np.square(Y_test).mean())/(RV_train.mean()+RV_test.mean())
#     elif scale == 'train':
#         c = (np.square(Y_train).mean())/(RV_train.mean())
#     elif scale == 'test':
#         c = (np.square(Y_test).mean())/(RV_test.mean())
#     else:
#         c = 1

#     RV_train *= c
#     RV_test *= c

#     if expand_dims:
#         Y_train = np.expand_dims(Y_train, 1)
#         Y_test = np.expand_dims(Y_test, 1)
#         RV_train = np.expand_dims(RV_train, 1)
#         RV_test = np.expand_dims(RV_test, 1)

#     if rvsup_ls is not None:
#         X_train_sup = data_train[rvsup_ls].to_numpy()*(10**4)
#         X_test_sup = data_test[rvsup_ls].to_numpy()*(10**4)
#         return Y_train, Y_test, RV_train, RV_test, X_train_sup, X_test_sup, data.index
#     else:
#         return Y_train, Y_test, RV_train, RV_test, data.index


def as_dict(rec):
    # convert np record to dict as shape of (-1,1)
    return {name:rec[name].reshape((-1,1)) for name in rec.dtype.names}

#####################################
# Activation Fn
def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return x * (x > 0)

#####################################
# CSTR Fn
def garch_cstr_fn(theta):
    return (theta['alpha']>=0)&(theta['beta']>=0)&((theta['alpha']+theta['beta'])<1)&(theta['omega']>=0)

def realgarch_cstr_fn(theta):
    return (theta['omega']+theta['gamma']*theta['xi']>0)&(theta['beta']+theta['gamma']*theta['phi']<1)&(theta['beta']+theta['gamma']*theta['phi']>0)

def rech_cstr_fn(theta):
    return (theta['alpha']>=0)&(theta['beta']>=0)&((theta['alpha']+theta['beta'])<1)

def realrech_cstr_fn(theta):
    return (theta['beta']+theta['gamma']*theta['phi']<1)&(theta['beta']+theta['gamma']*theta['phi']>0)

def egarch_cstr(theta):
    return (theta['beta']>=0)&(theta['beta']<1)

#####################################
# Prior
garch_prior = dists.StructDist({'omega': dists.Uniform(0,10),
                             'alpha': dists.Uniform(0,1),
                             'beta': dists.Uniform(0,1)})

realgarch_prior = dists.StructDist({'omega': dists.Gamma(1,1),
                                     'beta': dists.Beta(10,2),
                                     'gamma': dists.Beta(2,5),
                                     'xi': dists.Gamma(1,1),
                                     'phi': dists.Gamma(1,1),
                                     'tau1': dists.Normal(0,.1),
                                     'tau2': dists.Normal(0,.1),
                                     'sigmau2': dists.Gamma(1,5)})

rech_prior = dists.StructDist({'v0': dists.Normal(0,.1),
                             'v1': dists.Normal(0,.1),
                             'v2': dists.Normal(0,.1),
                             'w': dists.Normal(0,.1),
                             'b': dists.Normal(0,.1),
                             'beta0': dists.Gamma(1,.1),
                             'beta1': dists.Gamma(1,10),
                             'alpha': dists.Uniform(0,1),
                             'beta': dists.Uniform(0,1)})

realrech_prior = dists.StructDist({'v0f': dists.Normal(0,.1),
                             'v1f': dists.Normal(0,.1),
                             'v2f': dists.Normal(0,.1),
                             'v3f': dists.Normal(0,.1),
                             'wf': dists.Normal(0,.1),
                             'bf': dists.Normal(0,.1),
                             'v0i': dists.Normal(0,.1),
                             'v1i': dists.Normal(0,.1),
                             'v2i': dists.Normal(0,.1),
                             'v3i': dists.Normal(0,.1),
                             'wi': dists.Normal(0,.1),
                             'bi': dists.Normal(0,.1),
                             'v0o': dists.Normal(0,.1),
                             'v1o': dists.Normal(0,.1),
                             'v2o': dists.Normal(0,.1),
                             'v3o': dists.Normal(0,.1),
                             'wo': dists.Normal(0,.1),
                             'bo': dists.Normal(0,.1),
                             'v0d': dists.Normal(0,.1),
                             'v1d': dists.Normal(0,.1),
                             'v2d': dists.Normal(0,.1),
                             'v3d': dists.Normal(0,.1),
                             'wd': dists.Normal(0,.1),
                             'bd': dists.Normal(0,.1),
                             'beta0': dists.Gamma(1,.1),
                             'beta1': dists.Gamma(1,10),
                             'beta': dists.Uniform(0,1),
                             'gamma': dists.Beta(2,5),
                             'xi': dists.Gamma(1,1),
                             'phi': dists.Gamma(1,1),
                             'tau1': dists.Normal(0,.1),
                             'tau2': dists.Normal(0,.1),
                             'sigmau2': dists.Gamma(1,5)})

realrech_wm_prior = dists.StructDist({'v0f': dists.Normal(0,.1),
                             'v1f': dists.Normal(0,.1),
                             'v2f': dists.Normal(0,.1),
                             'v3f': dists.Normal(0,.1),
                             'v4f': dists.Normal(0,.1),
                             'v5f': dists.Normal(0,.1),
                             'wf': dists.Normal(0,.1),
                             'bf': dists.Normal(0,.1),
                             'v0i': dists.Normal(0,.1),
                             'v1i': dists.Normal(0,.1),
                             'v2i': dists.Normal(0,.1),
                             'v3i': dists.Normal(0,.1),
                             'v4i': dists.Normal(0,.1),
                             'v5i': dists.Normal(0,.1),
                             'wi': dists.Normal(0,.1),
                             'bi': dists.Normal(0,.1),
                             'v0o': dists.Normal(0,.1),
                             'v1o': dists.Normal(0,.1),
                             'v2o': dists.Normal(0,.1),
                             'v3o': dists.Normal(0,.1),
                             'v4o': dists.Normal(0,.1),
                             'v5o': dists.Normal(0,.1),
                             'wo': dists.Normal(0,.1),
                             'bo': dists.Normal(0,.1),
                             'v0d': dists.Normal(0,.1),
                             'v1d': dists.Normal(0,.1),
                             'v2d': dists.Normal(0,.1),
                             'v3d': dists.Normal(0,.1),
                             'v4d': dists.Normal(0,.1),
                             'v5d': dists.Normal(0,.1),
                             'wd': dists.Normal(0,.1),
                             'bd': dists.Normal(0,.1),
                             'beta0': dists.Gamma(1,.1),
                             'beta1': dists.Gamma(1,10),
                             'beta': dists.Uniform(0,1),
                             'gamma': dists.Beta(2,5),
                             'xi': dists.Gamma(1,1),
                             'phi': dists.Gamma(1,1),
                             'tau1': dists.Normal(0,.1),
                             'tau2': dists.Normal(0,.1),
                             'sigmau2': dists.Gamma(1,5)})

realrechsim_prior = dists.StructDist({'v0f': dists.Normal(0,.1),
                             'v1f': dists.Normal(0,.1),
                             'v2f': dists.Normal(0,.1),
                             'v3f': dists.Normal(0,.1),
                             'wf': dists.Normal(0,.1),
                             'bf': dists.Normal(0,.1),
                             'v0i': dists.Normal(0,.1),
                             'v1i': dists.Normal(0,.1),
                             'v2i': dists.Normal(0,.1),
                             'v3i': dists.Normal(0,.1),
                             'wi': dists.Normal(0,.1),
                             'bi': dists.Normal(0,.1),
                             'v0o': dists.Normal(0,.1),
                             'v1o': dists.Normal(0,.1),
                             'v2o': dists.Normal(0,.1),
                             'v3o': dists.Normal(0,.1),
                             'wo': dists.Normal(0,.1),
                             'bo': dists.Normal(0,.1),
                             'v0d': dists.Normal(0,.1),
                             'v1d': dists.Normal(0,.1),
                             'v2d': dists.Normal(0,.1),
                             'v3d': dists.Normal(0,.1),
                             'wd': dists.Normal(0,.1),
                             'bd': dists.Normal(0,.1),
                             'beta0': dists.Gamma(1,.1),
                             'beta1': dists.Gamma(1,10),
                             'beta': dists.Uniform(0,1),
                             'gamma': dists.Beta(2,5),
                             'xi': dists.Gamma(1,1),
                             'phi': dists.Gamma(1,1),
                             'sigmau2': dists.Gamma(1,5)})

realrech_1x_prior = dists.StructDist({'v0f': dists.Normal(0,.1),
                             'v1f': dists.Normal(0,.1),
                             'v2f': dists.Normal(0,.1),
                             'v3f': dists.Normal(0,.1),
                             'wf': dists.Normal(0,.1),
                             'bf': dists.Normal(0,.1),
                             'v0i': dists.Normal(0,.1),
                             'v1i': dists.Normal(0,.1),
                             'v2i': dists.Normal(0,.1),
                             'v3i': dists.Normal(0,.1),
                             'wi': dists.Normal(0,.1),
                             'bi': dists.Normal(0,.1),
                             'v0o': dists.Normal(0,.1),
                             'v1o': dists.Normal(0,.1),
                             'v2o': dists.Normal(0,.1),
                             'v3o': dists.Normal(0,.1),
                             'wo': dists.Normal(0,.1),
                             'bo': dists.Normal(0,.1),
                             'v0d': dists.Normal(0,.1),
                             'v1d': dists.Normal(0,.1),
                             'v2d': dists.Normal(0,.1),
                             'v3d': dists.Normal(0,.1),
                             'wd': dists.Normal(0,.1),
                             'bd': dists.Normal(0,.1),
                             'beta0': dists.Gamma(1,.1),
                             'beta1': dists.Gamma(1,10),
                             'beta': dists.Uniform(0,1),
                             'gamma': dists.Beta(2,5),
                             'xi': dists.Gamma(1,1),
                             'phi': dists.Gamma(1,1),
                             'tau1': dists.Normal(0,.1),
                             'tau2': dists.Normal(0,.1),
                             'sigmau2': dists.Gamma(1,5),
                             'v4f': dists.Normal(0,.1), # for more x
                             'v4i': dists.Normal(0,.1),
                             'v4o': dists.Normal(0,.1),
                             'v4d': dists.Normal(0,.1)})

realrech_2morex_prior = dists.StructDist({'v0f': dists.Normal(0,.1),
                             'v1f': dists.Normal(0,.1),
                             'v2f': dists.Normal(0,.1),
                             'v3f': dists.Normal(0,.1),
                             'wf': dists.Normal(0,.1),
                             'bf': dists.Normal(0,.1),
                             'v0i': dists.Normal(0,.1),
                             'v1i': dists.Normal(0,.1),
                             'v2i': dists.Normal(0,.1),
                             'v3i': dists.Normal(0,.1),
                             'wi': dists.Normal(0,.1),
                             'bi': dists.Normal(0,.1),
                             'v0o': dists.Normal(0,.1),
                             'v1o': dists.Normal(0,.1),
                             'v2o': dists.Normal(0,.1),
                             'v3o': dists.Normal(0,.1),
                             'wo': dists.Normal(0,.1),
                             'bo': dists.Normal(0,.1),
                             'v0d': dists.Normal(0,.1),
                             'v1d': dists.Normal(0,.1),
                             'v2d': dists.Normal(0,.1),
                             'v3d': dists.Normal(0,.1),
                             'wd': dists.Normal(0,.1),
                             'bd': dists.Normal(0,.1),
                             'beta0': dists.Gamma(1,.1),
                             'beta1': dists.Gamma(1,10),
                             'beta': dists.Uniform(0,1),
                             'gamma': dists.Beta(2,5),
                             'xi': dists.Gamma(1,1),
                             'phi': dists.Gamma(1,1),
                             'tau1': dists.Normal(0,.1),
                             'tau2': dists.Normal(0,.1),
                             'sigmau2': dists.Gamma(1,5),
                             'v4f': dists.Normal(0,.1), # for more x
                             'v4i': dists.Normal(0,.1),
                             'v4o': dists.Normal(0,.1),
                             'v4d': dists.Normal(0,.1),
                             'v5f': dists.Normal(0,.1), # for more x
                             'v5i': dists.Normal(0,.1),
                             'v5o': dists.Normal(0,.1),
                             'v5d': dists.Normal(0,.1)})

realrech_prior_sim = dists.StructDist({'v0f': dists.Normal(0,.1),
                             'v1f': dists.Normal(0,.1),
                             'v2f': dists.Normal(0,.1),
                             'v3f': dists.Normal(0,.1),
                             'wf': dists.Normal(0,.1),
                             'bf': dists.Normal(0,.1),
                             'v0i': dists.Normal(0,.1),
                             'v1i': dists.Normal(0,.1),
                             'v2i': dists.Normal(0,.1),
                             'v3i': dists.Normal(0,.1),
                             'wi': dists.Normal(0,.1),
                             'bi': dists.Normal(0,.1),
                             'v0o': dists.Normal(0,.1),
                             'v1o': dists.Normal(0,.1),
                             'v2o': dists.Normal(0,.1),
                             'v3o': dists.Normal(0,.1),
                             'wo': dists.Normal(0,.1),
                             'bo': dists.Normal(0,.1),
                             'v0d': dists.Normal(0,.1),
                             'v1d': dists.Normal(0,.1),
                             'v2d': dists.Normal(0,.1),
                             'v3d': dists.Normal(0,.1),
                             'wd': dists.Normal(0,.1),
                             'bd': dists.Normal(0,.1),
                             'beta0': dists.Gamma(1,.1),
                             'beta1': dists.Gamma(1,10),
                             'beta': dists.Uniform(0,1),
                             'gamma': dists.Beta(2,5)})

realrech_2lstm_prior = dists.StructDist({'v0f': dists.Normal(0,.1),
                                        'v1f': dists.Normal(0,.1),
                                        'v2f': dists.Normal(0,.1),
                                        'v3f': dists.Normal(0,.1),
                                        'wf': dists.Normal(0,.1),
                                        'bf': dists.Normal(0,.1),
                                        'v0i': dists.Normal(0,.1),
                                        'v1i': dists.Normal(0,.1),
                                        'v2i': dists.Normal(0,.1),
                                        'v3i': dists.Normal(0,.1),
                                        'wi': dists.Normal(0,.1),
                                        'bi': dists.Normal(0,.1),
                                        'v0o': dists.Normal(0,.1),
                                        'v1o': dists.Normal(0,.1),
                                        'v2o': dists.Normal(0,.1),
                                        'v3o': dists.Normal(0,.1),
                                        'wo': dists.Normal(0,.1),
                                        'bo': dists.Normal(0,.1),
                                        'v0d': dists.Normal(0,.1),
                                        'v1d': dists.Normal(0,.1),
                                        'v2d': dists.Normal(0,.1),
                                        'v3d': dists.Normal(0,.1),
                                        'wd': dists.Normal(0,.1),
                                        'bd': dists.Normal(0,.1),
                                        'beta0': dists.Gamma(1,.1),
                                        'beta1': dists.Gamma(1,10),
                                        'beta': dists.Uniform(0,1),
                                        'gamma': dists.Beta(2,5),
                                        'xi': dists.Gamma(1,1),
                                        'phi': dists.Gamma(1,1),
                                        'tau1': dists.Normal(0,.1),
                                        'tau2': dists.Normal(0,.1),
                                        'sigmau2': dists.Gamma(1,5),
                                        'v0f_rv': dists.Normal(0,.1),
                                        'v1f_rv': dists.Normal(0,.1),
                                        'v2f_rv': dists.Normal(0,.1),
                                        'v3f_rv': dists.Normal(0,.1),
                                        'wf_rv': dists.Normal(0,.1),
                                        'bf_rv': dists.Normal(0,.1),
                                        'v0i_rv': dists.Normal(0,.1),
                                        'v1i_rv': dists.Normal(0,.1),
                                        'v2i_rv': dists.Normal(0,.1),
                                        'v3i_rv': dists.Normal(0,.1),
                                        'wi_rv': dists.Normal(0,.1),
                                        'bi_rv': dists.Normal(0,.1),
                                        'v0o_rv': dists.Normal(0,.1),
                                        'v1o_rv': dists.Normal(0,.1),
                                        'v2o_rv': dists.Normal(0,.1),
                                        'v3o_rv': dists.Normal(0,.1),
                                        'wo_rv': dists.Normal(0,.1),
                                        'bo_rv': dists.Normal(0,.1),
                                        'v0d_rv': dists.Normal(0,.1),
                                        'v1d_rv': dists.Normal(0,.1),
                                        'v2d_rv': dists.Normal(0,.1),
                                        'v3d_rv': dists.Normal(0,.1),
                                        'wd_rv': dists.Normal(0,.1),
                                        'bd_rv': dists.Normal(0,.1),
                                        'beta0_rv': dists.Gamma(1,.1),
                                        'beta1_rv': dists.Gamma(1,10),})

def realrech_5h_prior(n):
    return dists.StructDist({'beta': dists.Uniform(0,1),
                            'gamma': dists.Beta(2,5),
                            'w2': dists.MvNormal(np.zeros(5),1),
                            'b2': dists.Normal(0,1),
                            'w': dists.MvNormal(np.zeros(n),1),
                            'xi': dists.Gamma(1,1),
                            'phi': dists.Gamma(1,1),
                            'tau1': dists.Normal(0,.1),
                            'tau2': dists.Normal(0,.1),
                            'sigmau2': dists.Gamma(1,5)})

#####################################
# t Prior
garcht_prior = dists.StructDist({'omega': dists.Uniform(0,10),
                             'alpha': dists.Uniform(0,1),
                             'beta': dists.Uniform(0,1),
                             'nu': dists.Gamma(1,1)})

realgarcht_prior = dists.StructDist({'omega': dists.Gamma(1,1),
                                     'beta': dists.Beta(10,2),
                                     'gamma': dists.Beta(2,5),
                                     'xi': dists.Gamma(1,1),
                                     'phi': dists.Gamma(1,1),
                                     'tau1': dists.Normal(0,.1),
                                     'tau2': dists.Normal(0,.1),
                                     'sigmau2': dists.Gamma(1,5),
                                     'nu': dists.Gamma(1,1)})

recht_prior = dists.StructDist({'v0': dists.Normal(0,.1),
                             'v1': dists.Normal(0,.1),
                             'v2': dists.Normal(0,.1),
                             'w': dists.Normal(0,.1),
                             'b': dists.Normal(0,.1),
                             'beta0': dists.Gamma(1,.1),
                             'beta1': dists.Gamma(1,10),
                             'alpha': dists.Uniform(0,1),
                             'beta': dists.Uniform(0,1),
                              'nu': dists.Gamma(1,1)})

realrecht_prior = dists.StructDist({'v0f': dists.Normal(0,.1),
                             'v1f': dists.Normal(0,.1),
                             'v2f': dists.Normal(0,.1),
                             'v3f': dists.Normal(0,.1),
                             'wf': dists.Normal(0,.1),
                             'bf': dists.Normal(0,.1),
                             'v0i': dists.Normal(0,.1),
                             'v1i': dists.Normal(0,.1),
                             'v2i': dists.Normal(0,.1),
                             'v3i': dists.Normal(0,.1),
                             'wi': dists.Normal(0,.1),
                             'bi': dists.Normal(0,.1),
                             'v0o': dists.Normal(0,.1),
                             'v1o': dists.Normal(0,.1),
                             'v2o': dists.Normal(0,.1),
                             'v3o': dists.Normal(0,.1),
                             'wo': dists.Normal(0,.1),
                             'bo': dists.Normal(0,.1),
                             'v0d': dists.Normal(0,.1),
                             'v1d': dists.Normal(0,.1),
                             'v2d': dists.Normal(0,.1),
                             'v3d': dists.Normal(0,.1),
                             'wd': dists.Normal(0,.1),
                             'bd': dists.Normal(0,.1),
                             'beta0': dists.Gamma(1,.1),
                             'beta1': dists.Gamma(1,10),
                             'beta': dists.Uniform(0,1),
                             'gamma': dists.Beta(2,5),
                             'xi': dists.Gamma(1,1),
                             'phi': dists.Gamma(1,1),
                             'tau1': dists.Normal(0,.1),
                             'tau2': dists.Normal(0,.1),
                             'sigmau2': dists.Gamma(1,5),
                             'nu': dists.Gamma(1,1)})

realrecht_norv_prior = dists.StructDist({'v0f': dists.Normal(0,.1),
                                         'v1f': dists.Normal(0,.1),
                                         'v2f': dists.Normal(0,.1),
                                         'v3f': dists.Normal(0,.1),
                                         'wf': dists.Normal(0,.1),
                                         'bf': dists.Normal(0,.1),
                                         'v0i': dists.Normal(0,.1),
                                         'v1i': dists.Normal(0,.1),
                                         'v2i': dists.Normal(0,.1),
                                         'v3i': dists.Normal(0,.1),
                                         'wi': dists.Normal(0,.1),
                                         'bi': dists.Normal(0,.1),
                                         'v0o': dists.Normal(0,.1),
                                         'v1o': dists.Normal(0,.1),
                                         'v2o': dists.Normal(0,.1),
                                         'v3o': dists.Normal(0,.1),
                                         'wo': dists.Normal(0,.1),
                                         'bo': dists.Normal(0,.1),
                                         'v0d': dists.Normal(0,.1),
                                         'v1d': dists.Normal(0,.1),
                                         'v2d': dists.Normal(0,.1),
                                         'v3d': dists.Normal(0,.1),
                                         'wd': dists.Normal(0,.1),
                                         'bd': dists.Normal(0,.1),
                                         'beta0': dists.Gamma(1,.1),
                                         'beta1': dists.Gamma(1,10),
                                         'beta': dists.Normal(0,1),
                                         'gamma': dists.Beta(2,5),
                                         'nu': dists.Gamma(1,1)})

#####################################
# EGARCH Prior
egarch_prior = dists.StructDist({'omega': dists.Normal(0,1),
                                 'alpha': dists.Normal(0,1),
                                 'beta': dists.Normal(0,1),
                                 'gammaa': dists.Normal(0,0.1)})

realegarch_prior = dists.StructDist({'omega': dists.Normal(0,1),
                                    'beta': dists.Normal(0,0.1),
                                    'gamma': dists.Normal(0,1),
                                    'xi': dists.Gamma(1,1),
                                    'sigmau2': dists.Gamma(1,5),
                                    'phi': dists.Normal(0,1),
                                    'delta1': dists.Normal(0,.1),
                                    'delta2': dists.Normal(0,.1),
                                    'tau1': dists.Normal(0,.1),
                                    'tau2': dists.Normal(0,.1)})

erech_prior = dists.StructDist({'v0': dists.Normal(0,.1),
                            'v1': dists.Normal(0,.1),
                            'v2': dists.Normal(0,.1),
                            'w': dists.Normal(0,.1),
                            'b': dists.Normal(0,.1),
                            'beta0': dists.Gamma(1,.1),
                            'beta1': dists.Gamma(1,10),
                            'alpha': dists.Normal(0,1),
                            'beta': dists.Normal(0,1),
                            'gammaa': dists.Normal(0,0.1)})

realerech_prior = dists.StructDist({'v0f': dists.Normal(0,.1),
                                 'v1f': dists.Normal(0,.1),
                                 'v2f': dists.Normal(0,.1),
                                 'v3f': dists.Normal(0,.1),
                                 'wf': dists.Normal(0,.1),
                                 'bf': dists.Normal(0,.1),
                                 'v0i': dists.Normal(0,.1),
                                 'v1i': dists.Normal(0,.1),
                                 'v2i': dists.Normal(0,.1),
                                 'v3i': dists.Normal(0,.1),
                                 'wi': dists.Normal(0,.1),
                                 'bi': dists.Normal(0,.1),
                                 'v0o': dists.Normal(0,.1),
                                 'v1o': dists.Normal(0,.1),
                                 'v2o': dists.Normal(0,.1),
                                 'v3o': dists.Normal(0,.1),
                                 'wo': dists.Normal(0,.1),
                                 'bo': dists.Normal(0,.1),
                                 'v0d': dists.Normal(0,.1),
                                 'v1d': dists.Normal(0,.1),
                                 'v2d': dists.Normal(0,.1),
                                 'v3d': dists.Normal(0,.1),
                                 'wd': dists.Normal(0,.1),
                                 'bd': dists.Normal(0,.1),
                                 'beta0': dists.Gamma(1,.1),
                                 'beta1': dists.Gamma(1,10),
                                 'gamma': dists.Normal(0,1),
                                 'xi': dists.Gamma(1,1),
                                 'phi': dists.Gamma(1,1),
                                 'tau1': dists.Normal(0,.1),
                                 'tau2': dists.Normal(0,.1),
                                 'delta1': dists.Normal(0,.1),
                                 'delta2': dists.Normal(0,.1),
                                 'sigmau2': dists.Gamma(1,5),
                                 'beta': dists.Normal(0,1)})