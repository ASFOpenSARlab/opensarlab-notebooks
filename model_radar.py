"""
Retrieval of W, epsilon and s using model 01
- fixed A, B, C
- fixed alpha, beta, gamma
- linear relationship
"""

import os
import glob
import subprocess
import concurrent.futures
import time
import scipy
import seaborn as sns
import numpy as np
import pandas as pd
import oocprocess.preprocess as pp
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from osgeo import gdal
from scipy import ndimage
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from scipy.optimize import differential_evolution, minimize, shgo, dual_annealing
from scipy.stats import gaussian_kde
from ipywidgets import widgets

sns.set()


def r_squared(x, y):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2


def density_scatter_plot(
    x1,
    y1,
    x_label="W",
    y_label="W_hat",
    x_limit=None,
    y_limit=None,
    file_name="fig_00.png",
):
    # Calculate the point density
    x0 = x1[(~np.isnan(x1)) & (~np.isnan(y1))]
    y0 = y1[(~np.isnan(x1)) & (~np.isnan(y1))]
    xy = np.vstack([x0, y0])
    z0 = gaussian_kde(xy)(xy)
    idx = z0.argsort()
    x1, y1, z1 = x0[idx], y0[idx], z0[idx]

    fig0 = plt.figure(figsize=(5, 4))
    plt.scatter(x1, y1, c=z1, s=10, edgecolor=None)
    # plt.scatter(x0, y0, s=4, edgecolor="")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    x1 = plt.xlim(x_limit)
    y1 = plt.ylim(y_limit)
    plt.plot(x1, y1, "k-")
    # plt.plot(x1, np.array(y1) * 0.8, '--', color='0.6')
    # plt.plot(x1, np.array(y1) * 1.2, '--', color='0.6')
    # print('{}% of the data within the range of ±20%.'.format(
    #     np.sum((x0 * 0.8 < y0) & (y0 < x0 * 1.2) & (x0 < 100)) / x0[x0 < 100].size * 100))
    # print('{}% of the data within the range of ±10%.'.format(
    #     np.sum((x0 * 0.9 < y0) & (y0 < x0 * 1.1) & (x0 < 100)) / x0[x0 < 100].size * 100))
    # print('{}% of the data within the range of 20 Mg/ha.'.format(
    #     np.sum((x0 - 20 < y0) & (y0 < x0 + 20) & (x0 < 100)) / x0[x0 < 100].size * 100))
    # print('{}% of the data within the range of 10 Mg/ha.'.format(
    #     np.sum((x0 - 10 < y0) & (y0 < x0 + 10) & (x0 < 100)) / x0[x0 < 100].size * 100))

    # r2 = r2_score(x0[:, None], y0[:, None])
    r2 = r_squared(x0, y0)
    rmse = np.sqrt(mean_squared_error(x0[:, None], y0[:, None]))
    # print("RMSE: {:.5f}".format(rmse))
    plt.text(
        (max(x_limit) - min(x_limit)) * 0.04 + min(x_limit),
        (max(y_limit) - min(y_limit)) * 0.85 + min(y_limit),
        r"$R^2 = {:1.3f}$".format(r2) + "\n" + "$RMSE = {:1.3f}$".format(rmse),
    )
    plt.show()
    # plt.savefig(file_name, dpi=150)
    plt.close(fig0)

    return r2, rmse


def write_geotiff_with_gdalcopy(ref_file, in_array, out_name):
    in0 = gdal.Open(ref_file, gdal.GA_ReadOnly)
    try:
        os.remove(out_name)
    except OSError:
        pass
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.CreateCopy(
        out_name,
        in0,
        0,
        ["COMPRESS=LZW", "PREDICTOR=2"],
    )
    for i in range(in_array.shape[0]):
        ds.GetRasterBand(i + 1).WriteArray(in_array[i, :, :])
    ds.FlushCache()  # Write to disk.
    ds = None
    in0 = None


def data_clean(X, y, w1_noise=10, w2_noise=20):
    """

    :param X:
    :param y:
    :param w1_noise: absolute noise
    :param w2_noise: relative noise in %
    :return:
    """
    # define noise threshold
    noise_threshold = w1_noise + w2_noise * 0.01 * y

    mdl1 = RandomForestRegressor(
        n_estimators=10,
        max_depth=6,
    )
    mdl2 = KNeighborsRegressor(n_neighbors=12)

    mdl3 = MLPRegressor(learning_rate_init=0.01, hidden_layer_sizes=(100,))

    mdl_list = [mdl1, mdl2, mdl3]
    noise_list = []
    for mdl in mdl_list:
        estimators = [
            ("scale", StandardScaler()),
            ("impute", SimpleImputer()),
            ("learn", mdl),
        ]
        ppl = Pipeline(estimators)
        y_hat = ppl.fit(X, y).predict(X)
        y_c = np.int8(np.abs(y - y_hat) > noise_threshold)
        noise_list.append(y_c)

    noise0 = np.sum(np.stack(noise_list), axis=0)

    # post-preliminary, iteration
    noise = noise0
    noise_pool = []
    iter = 4
    for i in range(iter):
        Xtr = X[noise < 2, :]
        ytr = y[noise < 2]
        noise_list = []
        for mdl in mdl_list:
            estimators = [
                ("scale", StandardScaler()),
                ("impute", SimpleImputer()),
                ("learn", mdl),
            ]
            ppl = Pipeline(estimators)
            y_hat = ppl.fit(Xtr, ytr).predict(X)
            y_c = np.int8(np.abs(y - y_hat) > noise_threshold)
            noise_list.append(y_c)
            noise_pool.append(y_c)
        noise = np.sum(np.stack(noise_list), axis=0)

    noise = np.sum(np.stack(noise_pool), axis=0)
    idx = noise < 2 * iter
    X_clean = X[noise < 2 * iter, :]
    y_clean = y[noise < 2 * iter]

    # save results
    return X_clean, y_clean, idx


def data_clean_2(X, y, w1_noise=10):
    """

    :param X:
    :param y:
    :param w1_noise: relative noise in fraction
    :return:
    """
    # define noise threshold
    noise_threshold = w1_noise

    clf = StandardScaler()
    X1t = clf.fit_transform(X)
    clf = SimpleImputer()
    X1s = clf.fit_transform(X1t)

    clf = IsolationForest(
        n_estimators=300, contamination=noise_threshold, random_state=777
    )
    y1 = clf.fit_predict(X1s)

    idx = y1 == 1
    X_clean = X[idx, :]
    y_clean = y[idx]

    # save results
    return X_clean, y_clean, idx


class Retrieval:
    def __init__(self, size_p, size_t, size_s):
        """

        Args:
            size_p: num of pols
            size_t: num of temp. obs.
            size_s: num of spatial obs. (nxn)
        """
        self.size_p = size_p
        self.size_t = size_t
        self.size_s = size_s
        self.param_A = np.array([[0.18, 0.04]])
        self.param_B = np.array([[0.06, 0.12]])
        self.param_C = np.array([[0.4, 0.2]])
        self.param_D = np.array([[10, 1]])
        self.param_a = np.array([[0.14, 0.22]])
        self.param_b = np.array([[1, 1]])
        self.param_c = np.array([[0.4, 0.6]])

    def x_P(self, x):
        # x has size (size_p, )
        return np.tile(x, [1, self.size_t * self.size_s])

    def x_S(self, x):
        # x has size (size_n, size_s)
        return np.repeat(x, self.size_p * self.size_t, axis=1)

    def x_T(self, x):
        # x has size (size_t, )
        return np.tile(np.repeat(x, self.size_p, axis=1), [1, self.size_s])

    # parameter calibration
    def model_03a_fun(self, W, x):
        A = self.x_P(x[0 * self.size_p : 1 * self.size_p][None, :])
        B = self.x_P(x[1 * self.size_p : 2 * self.size_p][None, :])
        C = self.x_P(x[2 * self.size_p : 3 * self.size_p][None, :])
        a = self.x_P(x[3 * self.size_p : 4 * self.size_p][None, :])
        c = self.x_P(x[4 * self.size_p : 5 * self.size_p][None, :])
        D = self.x_P(x[5 * self.size_p : 6 * self.size_p][None, :])
        S = self.x_T(x[6 * self.size_p : 6 * self.size_p + self.size_t][None, :])

        model_fun = A * W**a * (1 - np.exp(-B * W)) + (C * W**c + D) * S * np.exp(
            -B * W
        )
        return model_fun

    def model_inverse_03a(self, y1, W1, bounds, name="tmp", init_weights=[4, 1]):
        """
        Model inversion 03a (A, B, C, D, a, c change with p, S change with t) - given y, W, retrieve A-c
        y = A*W^a (1-exp(-BW)) + (C*W^c + D)*S*exp(-BW)
        p order: HV, HH  --- modified 05/09/2018
        :param y1: input radar backscatter in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :param W1: input AGB in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :param inc_angle1: Incidence angle in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :return: Parameters
        """

        def model_inversion(instance):
            y = y1[instance, :]
            W = W1[instance, :]

            wt = np.array(init_weights)

            result = differential_evolution(
                lambda x: np.sum(
                    (
                        (
                            self.model_03a_fun(W, x).reshape(-1, self.size_p)
                            - y.reshape(-1, self.size_p)
                        )
                        * wt[-self.size_p :]
                    )
                    ** 2
                ),
                bounds,
            )
            param0 = result.x

            y_hat0 = self.model_03a_fun(W, param0)
            return y_hat0, param0

        params = []
        y_hat = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for a, b in executor.map(model_inversion, range(y1.shape[0])):
                y_hat.append(a.flatten())
                params.append(b)

        params = np.array(params)
        y_hat = np.array(y_hat)
        return params, y_hat

    def model_03a_sm_fun(self, W, sm, x):
        A = self.x_P(x[0 * self.size_p : 1 * self.size_p][None, :])
        B = self.x_P(x[1 * self.size_p : 2 * self.size_p][None, :])
        C = self.x_P(x[2 * self.size_p : 3 * self.size_p][None, :])
        a = self.x_P(x[3 * self.size_p : 4 * self.size_p][None, :])
        c = self.x_P(x[4 * self.size_p : 5 * self.size_p][None, :])
        D = self.x_P(x[5 * self.size_p : 6 * self.size_p][None, :])

        model_fun = A * W**a * (1 - np.exp(-B * W)) + (C * W**c + D) * sm * np.exp(
            -B * W
        )
        return model_fun

    def model_inverse_03a_sm(
        self, y1, W1, sm1, bounds, name="tmp", init_weights=[4, 1]
    ):
        """
        Model inversion 03a (A, B, C, D, a, c change with p) - given y, W, sm retrieve A-c
        y = A*W^a (1-exp(-BW)) + (C*W^c + D)*sm*exp(-BW)
        p order: HV, HH  --- modified 11/28/2020
        :param y1: input radar backscatter in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :param W1: input AGB in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :param sm1: input SM in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :return: Parameters
        """
        if self.size_p == 1:
            bounds = [bounds[i] for i in range(0, 12, 2)] + bounds[12:]

        def model_inversion(instance):
            y = y1[instance, :]
            W = W1[instance, :]
            sm = sm1[instance, :]

            wt = np.array(init_weights)

            result = differential_evolution(
                lambda x: np.sum(
                    (
                        (
                            self.model_03a_sm_fun(W, sm, x).reshape(-1, self.size_p)
                            - y.reshape(-1, self.size_p)
                        )
                        * wt[-self.size_p :]
                    )
                    ** 2
                ),
                bounds,
            )
            param0 = result.x

            y_hat0 = self.model_03a_sm_fun(W, sm, param0)
            return y_hat0, param0

        params = []
        y_hat = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for a, b in executor.map(model_inversion, range(y1.shape[0])):
                y_hat.append(a.flatten())
                params.append(b)

        params = np.array(params)
        y_hat = np.array(y_hat)
        return params, y_hat

    def model_03b_sm_fun(self, W, sm, x):
        A = self.x_P(x[0 * self.size_p : 1 * self.size_p][None, :])
        B = self.x_P(x[1 * self.size_p : 2 * self.size_p][None, :])
        C = self.x_P(x[2 * self.size_p : 3 * self.size_p][None, :])
        a = self.x_P(x[3 * self.size_p : 4 * self.size_p][None, :])
        c = self.x_P(x[4 * self.size_p : 5 * self.size_p][None, :])
        D = self.x_P(x[5 * self.size_p : 6 * self.size_p][None, :])
        L = self.x_S(x[6 * self.size_p : 6 * self.size_p + self.size_s][None, :])

        model_fun = A * W**a * (1 - np.exp(-B * W)) + (
            C * W**c + D
        ) * L * sm * np.exp(-B * W)
        return model_fun

    def model_inverse_03b_sm(
        self, y1, W1, sm1, bounds, name="tmp", init_weights=[4, 1]
    ):
        """
        Model inversion 03b (A, B, C, D, a, c change with p; L change with S)
        Given y, W, sm;  Retrieve A-c, L (D[0] set to 1)
        y = A*W^a (1-exp(-BW)) + (C*W^c + D)*L*sm*exp(-BW)
        p order: HV, HH  --- modified 12/02/2020
        :param y1: input radar backscatter in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :param W1: input AGB in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :param sm1: input SM in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :return: Parameters
        """
        if self.size_p == 1:
            bounds = [bounds[i] for i in range(0, 12, 2)] + bounds[12:]

        def model_inversion(instance):
            y = y1[instance, :]
            W = W1[instance, :]
            sm = sm1[instance, :]

            wt = np.array(init_weights)

            result = differential_evolution(
                lambda x: np.sum(
                    (
                        (
                            self.model_03b_sm_fun(W, sm, x).reshape(-1, self.size_p)
                            - y.reshape(-1, self.size_p)
                        )
                        * wt[-self.size_p :]
                    )
                    ** 2
                ),
                bounds,
            )
            param0 = result.x

            y_hat0 = self.model_03b_sm_fun(W, sm, param0)
            return y_hat0, param0

        params = []
        y_hat = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for a, b in executor.map(model_inversion, range(y1.shape[0])):
                y_hat.append(a.flatten())
                params.append(b)

        params = np.array(params)
        y_hat = np.array(y_hat)
        return params, y_hat

    def model_03c_sm_fun(self, W, sm, x):
        A = self.x_P(x[0 * self.size_p : 1 * self.size_p][None, :])
        B = self.x_P(x[1 * self.size_p : 2 * self.size_p][None, :])
        C = self.x_P(x[2 * self.size_p : 3 * self.size_p][None, :])
        a = self.x_P(x[3 * self.size_p : 4 * self.size_p][None, :])
        c = self.x_P(x[4 * self.size_p : 5 * self.size_p][None, :])
        D = self.x_P(x[5 * self.size_p : 6 * self.size_p][None, :])
        L = self.x_S(x[6 * self.size_p : 6 * self.size_p + self.size_s][None, :])
        E = self.x_S(
            x[6 * self.size_p + self.size_s : 6 * self.size_p + self.size_s * 2][
                None, :
            ]
        )

        model_fun = A * W**a * (1 - np.exp(-B * E * W)) + (
            C * W**c + D
        ) * L * sm * np.exp(-B * E * W)
        return model_fun

    def model_inverse_03c_sm(
        self, y1, W1, sm1, bounds, name="tmp", init_weights=[4, 1]
    ):
        """
        Model inversion 03b (A, B, C, D, a, c change with p; L, E change with S)
        Given y, W, sm;  Retrieve A-c, L, E (D[0] set to 1)
        y = A*W^a (1-exp(-BEW)) + (C*W^c + D)*L*sm*exp(-BEW)
        p order: HV, HH  --- modified 12/03/2020
        :param y1: input radar backscatter in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :param W1: input AGB in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :param sm1: input SM in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :return: Parameters
        """
        if self.size_p == 1:
            bounds = [bounds[i] for i in range(0, 12, 2)] + bounds[12:]

        def model_inversion(instance):
            y = y1[instance, :]
            W = W1[instance, :]
            sm = sm1[instance, :]

            wt = np.array(init_weights)

            result = differential_evolution(
                lambda x: np.sum(
                    (
                        (
                            self.model_03c_sm_fun(W, sm, x).reshape(-1, self.size_p)
                            - y.reshape(-1, self.size_p)
                        )
                        * wt[-self.size_p :]
                    )
                    ** 2
                ),
                bounds,
            )
            param0 = result.x

            # param0 = brute(lambda x: np.sum(
            #     ((self.model_03c_sm_fun(W, sm, x).reshape(-1, self.size_p)
            #       - y.reshape(-1, self.size_p)) * wt[-self.size_p:]) ** 2
            # ), bounds, Ns=10)

            y_hat0 = self.model_03c_sm_fun(W, sm, param0)
            return y_hat0, param0

        params = []
        y_hat = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for a, b in executor.map(model_inversion, range(y1.shape[0])):
                y_hat.append(a.flatten())
                params.append(b)

        params = np.array(params)
        y_hat = np.array(y_hat)
        return params, y_hat

    def model_03d_sm_fun(self, W, sm, x):
        A = self.x_P(x[0 * self.size_p : 1 * self.size_p][None, :])
        B = self.x_P(x[1 * self.size_p : 2 * self.size_p][None, :])
        C = self.x_P(x[2 * self.size_p : 3 * self.size_p][None, :])
        a = self.x_P(x[3 * self.size_p : 4 * self.size_p][None, :])
        c = self.x_P(x[4 * self.size_p : 5 * self.size_p][None, :])
        D = self.x_P(x[5 * self.size_p : 6 * self.size_p][None, :])
        L = self.x_S(x[6 * self.size_p : 6 * self.size_p + self.size_s][None, :])
        E = self.x_S(
            x[6 * self.size_p + self.size_s : 6 * self.size_p + self.size_s * 2][
                None, :
            ]
        )
        G = self.x_S(
            x[6 * self.size_p + self.size_s * 2 : 6 * self.size_p + self.size_s * 3][
                None, :
            ]
        )

        model_fun = (
            A * W**a * (1 - np.exp(-B * E * W))
            + (C * W**c + D) * L * sm * np.exp(-B * E * W)
            + G
        )
        return model_fun

    def model_inverse_03d_sm(
        self, y1, W1, sm1, bounds, name="tmp", init_weights=[4, 1]
    ):
        """
        Model inversion 03b (A, B, C, D, a, c change with p; L, E, G change with S)
        Given y, W, sm;  Retrieve A-c, L, E (D[0] set to 1)
        y = A*W^a (1-exp(-BEW)) + (C*W^c + D)*L*sm*exp(-BEW) + G
        p order: HV, HH  --- modified 12/03/2020
        :param y1: input radar backscatter in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :param W1: input AGB in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :param sm1: input SM in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :return: Parameters
        """
        if self.size_p == 1:
            bounds = [bounds[i] for i in range(0, 12, 2)] + bounds[12:]

        def model_inversion(instance):
            y = y1[instance, :]
            W = W1[instance, :]
            sm = sm1[instance, :]

            wt = np.array(init_weights)

            result = differential_evolution(
                lambda x: np.sum(
                    (
                        (
                            self.model_03d_sm_fun(W, sm, x).reshape(-1, self.size_p)
                            - y.reshape(-1, self.size_p)
                        )
                        * wt[-self.size_p :]
                    )
                    ** 2
                ),
                bounds,
            )
            param0 = result.x

            # param0 = brute(lambda x: np.sum(
            #     ((self.model_03c_sm_fun(W, sm, x).reshape(-1, self.size_p)
            #       - y.reshape(-1, self.size_p)) * wt[-self.size_p:]) ** 2
            # ), bounds, Ns=10)

            y_hat0 = self.model_03d_sm_fun(W, sm, param0)
            return y_hat0, param0

        params = []
        y_hat = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for a, b in executor.map(model_inversion, range(y1.shape[0])):
                y_hat.append(a.flatten())
                params.append(b)

        params = np.array(params)
        y_hat = np.array(y_hat)
        return params, y_hat

    def model_03e_sm_fun(self, W, sm, x):
        A = self.x_P(x[0 * self.size_p : 1 * self.size_p][None, :])
        B = self.x_P(x[1 * self.size_p : 2 * self.size_p][None, :])
        C = self.x_P(x[2 * self.size_p : 3 * self.size_p][None, :])
        a = self.x_P(x[3 * self.size_p : 4 * self.size_p][None, :])
        c = self.x_P(x[4 * self.size_p : 5 * self.size_p][None, :])
        D = self.x_P(x[5 * self.size_p : 6 * self.size_p][None, :])
        L = self.x_S(x[6 * self.size_p : 6 * self.size_p + self.size_s][None, :])

        model_fun = A * W**a * (1 - np.exp(-B * W)) + (
            C * W**c * (1 - L) + D * L
        ) * sm * np.exp(-B * W)
        # + C * W ** c * (1 - L) * sm * np.exp(-B * W) \
        # + D * L * sm
        return model_fun

    def model_inverse_03e_sm(
        self, y1, W1, sm1, bounds, name="tmp", init_weights=[4, 1]
    ):
        """
        Model inversion 03b (A, B, C, D, a, c change with p; L, E, G change with S)
        Given y, W, sm;  Retrieve A-c, L, E (D[0] set to 1)
        y = A*W^a (1-exp(-BEW)) + (C*W^c + D)*L*sm*exp(-BEW) + G
        p order: HV, HH  --- modified 12/03/2020
        :param y1: input radar backscatter in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :param W1: input AGB in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :param sm1: input SM in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :return: Parameters
        """
        # if self.size_p == 1:
        #     bounds = [bounds[i] for i in range(0, self.size_p * 6, self.size_p)] + bounds[self.size_p * 6:]

        def model_inversion(instance):
            y = y1[instance, :]
            W = W1[instance, :]
            sm = sm1[instance, :]

            wt = np.array(init_weights)

            result = differential_evolution(
                lambda x: np.sum(
                    (
                        (
                            self.model_03e_sm_fun(W, sm, x).reshape(-1, self.size_p)
                            - y.reshape(-1, self.size_p)
                        )
                        * wt[-self.size_p :]
                    )
                    ** 2
                ),
                bounds,
                # popsize=77, workers=1,
            )
            param0 = result.x

            # param0 = brute(lambda x: np.sum(
            #     ((self.model_03e_sm_fun(W, sm, x).reshape(-1, self.size_p)
            #       - y.reshape(-1, self.size_p)) * wt[-self.size_p:]) ** 2
            # ), bounds, Ns=10)

            # result = shgo(lambda x: np.sum(
            #     ((self.model_03e_sm_fun(W, sm, x).reshape(-1, self.size_p)
            #       - y.reshape(-1, self.size_p)) * wt[-self.size_p:]) ** 2
            # ), bounds)
            # param0 = result.x

            # result = dual_annealing(lambda x: np.sum(
            #     ((self.model_03e_sm_fun(W, sm, x).reshape(-1, self.size_p)
            #       - y.reshape(-1, self.size_p)) * wt[-self.size_p:]) ** 2
            # ), bounds)
            # param0 = result.x

            y_hat0 = self.model_03e_sm_fun(W, sm, param0)
            return y_hat0, param0

        params = []
        y_hat = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for a, b in executor.map(model_inversion, range(y1.shape[0])):
                y_hat.append(a.flatten())
                params.append(b)

        params = np.array(params)
        y_hat = np.array(y_hat)
        return params, y_hat

    def model_03g_sm_fun(self, W, sm, ks, x):
        A = self.x_P(x[0 * self.size_p : 1 * self.size_p][None, :])
        B = self.x_P(x[1 * self.size_p : 2 * self.size_p][None, :])
        C = self.x_P(x[2 * self.size_p : 3 * self.size_p][None, :])
        a = self.x_P(x[3 * self.size_p : 4 * self.size_p][None, :])
        c = self.x_P(x[4 * self.size_p : 5 * self.size_p][None, :])
        D = self.x_P(x[5 * self.size_p : 6 * self.size_p][None, :])

        model_fun = A * W**a * (1 - np.exp(-B * W)) + (
            C * W**c * (1 - ks) + D * ks
        ) * sm * np.exp(-B * W)
        # + C * W ** c * (1 - L) * sm * np.exp(-B * W) \
        # + D * L * sm
        return model_fun

    def model_inverse_03g_sm(
        self, y1, W1, sm1, ks1, bounds, name="tmp", init_weights=[4, 1]
    ):
        """
        Model inversion 03b (A, B, C, D, a, c change with p; L, E, G change with S)
        Given y, W, sm;  Retrieve A-c, L, E (D[0] set to 1)
        y = A*W^a (1-exp(-BEW)) + (C*W^c*(1-ks)+D*ks)*sm*exp(-BEW) + G
        p order: HV, HH  --- modified 12/03/2020
        :param y1: input radar backscatter in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :param W1: input AGB in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :param sm1: input SM in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :return: Parameters
        """
        # if self.size_p == 1:
        #     bounds = [bounds[i] for i in range(0, self.size_p * 6, self.size_p)] + bounds[self.size_p * 6:]

        def model_inversion(instance):
            y = y1[instance, :]
            W = W1[instance, :]
            sm = sm1[instance, :]
            ks = ks1[instance, :]

            wt = np.array(init_weights)

            result = differential_evolution(
                lambda x: np.sum(
                    (
                        (
                            self.model_03g_sm_fun(W, sm, ks, x).reshape(-1, self.size_p)
                            - y.reshape(-1, self.size_p)
                        )
                        * wt[-self.size_p :]
                    )
                    ** 2
                ),
                bounds,
                popsize=77,
                # workers=1,
            )
            param0 = result.x

            # param0 = brute(lambda x: np.sum(
            #     ((self.model_03e_sm_fun(W, sm, x).reshape(-1, self.size_p)
            #       - y.reshape(-1, self.size_p)) * wt[-self.size_p:]) ** 2
            # ), bounds, Ns=10)

            # result = shgo(lambda x: np.sum(
            #     ((self.model_03e_sm_fun(W, sm, x).reshape(-1, self.size_p)
            #       - y.reshape(-1, self.size_p)) * wt[-self.size_p:]) ** 2
            # ), bounds)
            # param0 = result.x

            # result = dual_annealing(lambda x: np.sum(
            #     ((self.model_03e_sm_fun(W, sm, x).reshape(-1, self.size_p)
            #       - y.reshape(-1, self.size_p)) * wt[-self.size_p:]) ** 2
            # ), bounds)
            # param0 = result.x

            y_hat0 = self.model_03g_sm_fun(W, sm, ks, param0)
            return y_hat0, param0

        params = []
        y_hat = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for a, b in executor.map(model_inversion, range(y1.shape[0])):
                y_hat.append(a.flatten())
                params.append(b)

        params = np.array(params)
        y_hat = np.array(y_hat)
        return params, y_hat

    def model_04a_noint(self, W, x):
        A = self.x_P(x[0 * self.size_p : 1 * self.size_p][None, :])
        B = self.x_P(x[1 * self.size_p : 2 * self.size_p][None, :])
        C = self.x_P(x[2 * self.size_p : 3 * self.size_p][None, :])
        D = self.x_P(x[3 * self.size_p : 4 * self.size_p][None, :])
        a = self.x_P(x[4 * self.size_p : 5 * self.size_p][None, :])
        S = self.x_T(x[5 * self.size_p : 5 * self.size_p + self.size_t][None, :])

        model_fun = (
            A * W**a * (1 - np.exp(-B * W * 0.1)) + D * S * np.exp(-B * W * 0.1) + C
        )
        return model_fun

    def model_inverse_04a_noint(self, y1, W1, bounds, name="tmp", init_weights=[4, 1]):
        """
        Model inversion 04a (A, B, C, D change with p, S change with t) - given y, W, retrieve A-c
        y = A*(1-exp(-BW)) + D*S*exp(-BW) + C
        p order: HV, HH  --- modified 02/22/2021
        :param y1: input radar backscatter in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :param W1: input AGB in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :param inc_angle1: Incidence angle in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :return: Parameters
        """

        def model_inversion(instance):
            y = y1[instance, :]
            W = W1[instance, :]

            wt = np.array(init_weights)

            result = differential_evolution(
                lambda x: np.sum(
                    (
                        (
                            self.model_04a_noint(W, x).reshape(-1, self.size_p)
                            - y.reshape(-1, self.size_p)
                        )
                        * wt[-self.size_p :]
                    )
                    ** 2
                ),
                bounds,
            )
            param0 = result.x

            y_hat0 = self.model_04a_noint(W, param0)
            return y_hat0, param0

        params = []
        y_hat = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for a, b in executor.map(model_inversion, range(y1.shape[0])):
                y_hat.append(a.flatten())
                params.append(b)

        params = np.array(params)
        y_hat = np.array(y_hat)
        return params, y_hat

    # Inversion with only SAR
    def sar_model_03_fun(self, x):
        A = self.x_P(x[0 * self.size_p : 1 * self.size_p][None, :])
        B = self.x_P(x[1 * self.size_p : 2 * self.size_p][None, :])
        C = self.x_P(x[2 * self.size_p : 3 * self.size_p][None, :])
        a = self.x_P(x[3 * self.size_p : 4 * self.size_p][None, :])
        c = self.x_P(x[4 * self.size_p : 5 * self.size_p][None, :])
        D = self.x_P(x[5 * self.size_p : 6 * self.size_p][None, :])
        S = self.x_T(x[6 * self.size_p : 6 * self.size_p + self.size_t][None, :])
        W = self.x_S(
            x[
                6 * self.size_p
                + self.size_t : 6 * self.size_p
                + self.size_t
                + self.size_s
            ][None, :]
        )

        model_fun = A * W**a * (1 - np.exp(-B * W)) + (C * W**c + D) * S * np.exp(
            -B * W
        )
        return model_fun

    def sar_model_inverse_03(self, param0, y1, bounds, name="tmp", init_weights=[4, 1]):
        """
        Model inversion 03a (A, B, C, D, a, c change with p, S change with t) - given y, W, retrieve A-c
        y = A*W^a (1-exp(-BW)) + (C*W^c + D)*S*exp(-BW)
        p order: HV, HH
        5x5 approach - parameter being all LOCAL
        --- modified 05/10/2018
        :param y1: input radar backscatter in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :param W1: input AGB in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :param inc_angle1: Incidence angle in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :return: Parameters
        """

        param0w = param0
        bounds0w = bounds

        def model_inversion(instance):
            y = y1[instance, :]
            param0aw = param0w[instance, :]
            bounds0aw = bounds0w[instance]

            wt = np.array(init_weights)
            result = minimize(
                lambda x: np.sum(
                    (
                        (
                            self.sar_model_03_fun(x).reshape(-1, self.size_p)
                            - y.reshape(-1, self.size_p)
                        )
                        * wt[-self.size_p :]
                    )
                    ** 2
                ),
                param0aw,
                method="L-BFGS-B",
                bounds=bounds0aw,
            )
            # result = differential_evolution(lambda x: np.sum(
            #     ((self.model_03a_fun(W, x).reshape(-1, self.size_p) - y.reshape(-1, self.size_p)) * wt) ** 2
            # ), bounds)
            # print(instance)
            # print(result.success)
            param1 = result.x

            y_hat1 = self.sar_model_03_fun(param1)

            return y_hat1, param1

        params = []
        y_hat = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for a, b in executor.map(model_inversion, range(y1.shape[0])):
                y_hat.append(a.flatten())
                params.append(b)

        params = np.array(params)
        y_hat = np.array(y_hat)
        return params, y_hat

    def sar_model_03_sm_fun(self, x):
        A = self.x_P(x[0 * self.size_p : 1 * self.size_p][None, :])
        B = self.x_P(x[1 * self.size_p : 2 * self.size_p][None, :])
        C = self.x_P(x[2 * self.size_p : 3 * self.size_p][None, :])
        a = self.x_P(x[3 * self.size_p : 4 * self.size_p][None, :])
        c = self.x_P(x[4 * self.size_p : 5 * self.size_p][None, :])
        D = self.x_P(x[5 * self.size_p : 6 * self.size_p][None, :])
        L = self.x_S(x[6 * self.size_p : 6 * self.size_p + self.size_s][None, :])
        S = self.x_T(
            x[
                6 * self.size_p
                + self.size_s : 6 * self.size_p
                + self.size_s
                + self.size_t
            ][None, :]
        )
        W = self.x_S(
            x[
                6 * self.size_p
                + self.size_s
                + self.size_t : 6 * self.size_p
                + self.size_s
                + self.size_t
                + self.size_s
            ][None, :]
        )
        model_fun = A * W**a * (1 - np.exp(-B * W)) + (
            C * W**c + D
        ) * L * S * np.exp(-B * W)
        return model_fun

    def sar_model_03_smc_fun(self, x):
        A = self.x_P(x[0 * self.size_p : 1 * self.size_p][None, :])
        B = self.x_P(x[1 * self.size_p : 2 * self.size_p][None, :])
        C = self.x_P(x[2 * self.size_p : 3 * self.size_p][None, :])
        a = self.x_P(x[3 * self.size_p : 4 * self.size_p][None, :])
        c = self.x_P(x[4 * self.size_p : 5 * self.size_p][None, :])
        D = self.x_P(x[5 * self.size_p : 6 * self.size_p][None, :])
        L = self.x_S(x[6 * self.size_p : 6 * self.size_p + self.size_s][None, :])
        E = self.x_S(
            x[6 * self.size_p + self.size_s : 6 * self.size_p + self.size_s * 2][
                None, :
            ]
        )
        S = self.x_T(
            x[
                6 * self.size_p
                + self.size_s * 2 : 6 * self.size_p
                + self.size_s * 2
                + self.size_t
            ][None, :]
        )
        W = self.x_S(
            x[
                6 * self.size_p
                + self.size_s * 2
                + self.size_t : 6 * self.size_p
                + self.size_s * 2
                + self.size_t
                + self.size_s
            ][None, :]
        )
        model_fun = A * W**a * (1 - np.exp(-B * E * W)) + (
            C * W**c + D
        ) * L * S * np.exp(-B * E * W)
        return model_fun

    def sar_model_03_smd_fun(self, x):
        A = self.x_P(x[0 * self.size_p : 1 * self.size_p][None, :])
        B = self.x_P(x[1 * self.size_p : 2 * self.size_p][None, :])
        C = self.x_P(x[2 * self.size_p : 3 * self.size_p][None, :])
        a = self.x_P(x[3 * self.size_p : 4 * self.size_p][None, :])
        c = self.x_P(x[4 * self.size_p : 5 * self.size_p][None, :])
        D = self.x_P(x[5 * self.size_p : 6 * self.size_p][None, :])
        L = self.x_S(x[6 * self.size_p : 6 * self.size_p + self.size_s][None, :])
        E = self.x_S(
            x[6 * self.size_p + self.size_s : 6 * self.size_p + self.size_s * 2][
                None, :
            ]
        )
        G = self.x_S(
            x[6 * self.size_p + self.size_s * 2 : 6 * self.size_p + self.size_s * 3][
                None, :
            ]
        )
        S = self.x_T(
            x[
                6 * self.size_p
                + self.size_s * 3 : 6 * self.size_p
                + self.size_s * 3
                + self.size_t
            ][None, :]
        )
        W = self.x_S(
            x[
                6 * self.size_p
                + self.size_s * 3
                + self.size_t : 6 * self.size_p
                + self.size_s * 3
                + self.size_t
                + self.size_s
            ][None, :]
        )
        model_fun = (
            A * W**a * (1 - np.exp(-B * E * W))
            + (C * W**c + D) * L * S * np.exp(-B * E * W)
            + G
        )
        return model_fun

    def sar_model_03_sme_fun(self, x):
        A = self.x_P(x[0 * self.size_p : 1 * self.size_p][None, :])
        B = self.x_P(x[1 * self.size_p : 2 * self.size_p][None, :])
        C = self.x_P(x[2 * self.size_p : 3 * self.size_p][None, :])
        a = self.x_P(x[3 * self.size_p : 4 * self.size_p][None, :])
        c = self.x_P(x[4 * self.size_p : 5 * self.size_p][None, :])
        D = self.x_P(x[5 * self.size_p : 6 * self.size_p][None, :])
        L = self.x_S(x[6 * self.size_p : 6 * self.size_p + self.size_s][None, :])
        S = self.x_T(
            x[
                6 * self.size_p
                + self.size_s : 6 * self.size_p
                + self.size_s
                + self.size_t
            ][None, :]
        )
        W = self.x_S(
            x[
                6 * self.size_p
                + self.size_s
                + self.size_t : 6 * self.size_p
                + self.size_s
                + self.size_t
                + self.size_s
            ][None, :]
        )
        model_fun = A * W**a * (1 - np.exp(-B * W)) + (
            C * W**c * (1 - L) + D * L
        ) * S * np.exp(-B * W)
        # + C * W ** c * (1 - L) * S * np.exp(-B * W) \
        # + D * L * S
        return model_fun

    def sar_model_inverse_03_sm(
        self, param0, y1, bounds, name="tmp", init_weights=[1, 1]
    ):
        """
        Model inversion 03e_sm (A, B, C, D, a, c change with p; L change with S)
        Given y, W, sm;  Retrieve A-c, L (D[0] set to 1)
        y = A*W^a (1-exp(-BW)) + (C*W^c + D)*L*sm*exp(-BW)
        p order: HV, HH  --- modified 12/03/2020
        :param y1: input radar backscatter in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :return: Parameters
        """

        if self.size_p == 1:
            bounds = [bounds[i] for i in range(0, 12, 2)] + bounds[12:]
            param0 = [param0[i] for i in range(0, 12, 2)] + param0[12:]

        param0w = param0
        bounds0w = bounds

        def model_inversion(instance):
            y = y1[instance, :]
            param0aw = param0w[instance, :]
            bounds0aw = bounds0w[instance]

            wt = np.array(init_weights)
            result = minimize(
                lambda x: np.sum(
                    (
                        (
                            self.sar_model_03_sm_fun(x).reshape(-1, self.size_p)
                            - y.reshape(-1, self.size_p)
                        )
                        * wt[-self.size_p :]
                    )
                    ** 2
                ),
                param0aw,
                method="L-BFGS-B",
                bounds=bounds0aw,
            )
            # result = differential_evolution(lambda x: np.sum(
            #     ((self.model_03a_fun(W, x).reshape(-1, self.size_p) - y.reshape(-1, self.size_p)) * wt) ** 2
            # ), bounds)
            # print(instance)
            # print(result.success)
            param1 = result.x

            y_hat1 = self.sar_model_03_sm_fun(param1)

            return y_hat1, param1

        params = []
        y_hat = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for a, b in executor.map(model_inversion, range(y1.shape[0])):
                y_hat.append(a.flatten())
                params.append(b)

        if self.size_p == 1:
            param1tmp = param0
            param1tmp[0:12:2] = params[:6]
            param1tmp[12:] = params[6:]
            params = param1tmp

        params = np.array(params)
        y_hat = np.array(y_hat)
        return params, y_hat

    def sar_model_inverse_03_smc(
        self, param0, y1, bounds, name="tmp", init_weights=[1, 1]
    ):
        """
        Model inversion 03e_sm (A, B, C, D, a, c change with p; L, E change with S)
        Given y, W, sm;  Retrieve A-c, L, E (D[0] set to 1)
        y = A*W^a (1-exp(-BEW)) + (C*W^c + D)*L*sm*exp(-BEW)
        p order: HV, HH  --- modified 12/03/2020
        :param y1: input radar backscatter in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :return: Parameters
        """

        if self.size_p == 1:
            bounds = [bounds[i] for i in range(0, 12, 2)] + bounds[12:]
            param0 = [param0[i] for i in range(0, 12, 2)] + param0[12:]

        param0w = param0
        bounds0w = bounds

        def model_inversion(instance):
            y = y1[instance, :]
            param0aw = param0w[instance, :]
            bounds0aw = bounds0w[instance]

            wt = np.array(init_weights)
            result = minimize(
                lambda x: np.sum(
                    (
                        (
                            self.sar_model_03_smc_fun(x).reshape(-1, self.size_p)
                            - y.reshape(-1, self.size_p)
                        )
                        * wt[-self.size_p :]
                    )
                    ** 2
                ),
                param0aw,
                method="L-BFGS-B",
                bounds=bounds0aw,
            )
            # result = differential_evolution(lambda x: np.sum(
            #     ((self.model_03a_fun(W, x).reshape(-1, self.size_p) - y.reshape(-1, self.size_p)) * wt) ** 2
            # ), bounds)
            # print(instance)
            # print(result.success)
            param1 = result.x

            y_hat1 = self.sar_model_03_smc_fun(param1)

            return y_hat1, param1

        params = []
        y_hat = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for a, b in executor.map(model_inversion, range(y1.shape[0])):
                y_hat.append(a.flatten())
                params.append(b)

        if self.size_p == 1:
            param1tmp = param0
            param1tmp[0:12:2] = params[:6]
            param1tmp[12:] = params[6:]
            params = param1tmp

        params = np.array(params)
        y_hat = np.array(y_hat)
        return params, y_hat

    def sar_model_inverse_03_smd(
        self, param0, y1, bounds, name="tmp", init_weights=[1, 1]
    ):
        """
        Model inversion 03e_sm (A, B, C, D, a, c change with p; L, E, G change with S)
        Given y, W, sm;  Retrieve A-c, L,E,G (D[0] set to 1)
        y = A*W^a (1-exp(-BEW)) + (C*W^c + D)*L*sm*exp(-BEW) + G
        p order: HV, HH  --- modified 12/04/2020
        :param y1: input radar backscatter in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :return: Parameters
        """

        if self.size_p == 1:
            bounds = [bounds[i] for i in range(0, 12, 2)] + bounds[12:]
            param0 = [param0[i] for i in range(0, 12, 2)] + param0[12:]

        param0w = param0
        bounds0w = bounds

        def model_inversion(instance):
            y = y1[instance, :]
            param0aw = param0w[instance, :]
            bounds0aw = bounds0w[instance]

            wt = np.array(init_weights)
            result = minimize(
                lambda x: np.sum(
                    (
                        (
                            self.sar_model_03_smd_fun(x).reshape(-1, self.size_p)
                            - y.reshape(-1, self.size_p)
                        )
                        * wt[-self.size_p :]
                    )
                    ** 2
                ),
                param0aw,
                method="L-BFGS-B",
                bounds=bounds0aw,
            )
            # result = differential_evolution(lambda x: np.sum(
            #     ((self.model_03a_fun(W, x).reshape(-1, self.size_p) - y.reshape(-1, self.size_p)) * wt) ** 2
            # ), bounds)
            # print(instance)
            # print(result.success)
            param1 = result.x

            y_hat1 = self.sar_model_03_smd_fun(param1)

            return y_hat1, param1

        params = []
        y_hat = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for a, b in executor.map(model_inversion, range(y1.shape[0])):
                y_hat.append(a.flatten())
                params.append(b)

        if self.size_p == 1:
            param1tmp = param0
            param1tmp[0:12:2] = params[:6]
            param1tmp[12:] = params[6:]
            params = param1tmp

        params = np.array(params)
        y_hat = np.array(y_hat)
        return params, y_hat

    def sar_model_inverse_03_sme(
        self, param0, y1, bounds, name="tmp", init_weights=[1, 1]
    ):
        """
        Model inversion 03e_sm (A, B, C, D, a, c change with p; L change with S)
        Given y, W, sm;  Retrieve A-c, L (D[0] set to 1)
        y = A*W^a (1-exp(-BW)) + (C*W^c*(1-L) + D*L) *sm*exp(-BW)
        p order: HV, HH  --- modified 12/04/2020
        :param y1: input radar backscatter in ndarray with dimension: [n_sample, size_p*size_t*size_s].
        :return: Parameters
        """

        # if self.size_p == 1:
        #     bounds = [bounds[i] for i in range(0, self.size_p*6, self.size_p)] + bounds[self.size_p*6:]
        #     param0 = [param0[i] for i in range(0, self.size_p*6, self.size_p)] + param0[self.size_p*6:]

        param0w = param0
        bounds0w = bounds

        def model_inversion(instance):
            y = y1[instance, :]
            param0aw = param0w[instance, :]
            bounds0aw = bounds0w[instance]

            wt = np.array(init_weights)

            result = minimize(
                lambda x: np.sum(
                    (
                        (
                            self.sar_model_03_sme_fun(x).reshape(-1, self.size_p)
                            - y.reshape(-1, self.size_p)
                        )
                        * wt[-self.size_p :]
                    )
                    ** 2
                ),
                param0aw,
                method="L-BFGS-B",
                bounds=bounds0aw,
            )

            # result = dual_annealing(lambda x: np.sum(
            #     ((self.sar_model_03_sme_fun(x).reshape(-1, self.size_p)
            #       - y.reshape(-1, self.size_p)) * wt[-self.size_p:]) ** 2
            # ), bounds0aw)

            # result = differential_evolution(lambda x: np.sum(
            #     ((self.sar_model_03_sme_fun(x).reshape(-1, self.size_p) - y.reshape(-1, self.size_p)) * wt) ** 2
            # ), bounds0aw, popsize=127)
            # print(instance)
            # print(result.success)

            param1 = result.x
            y_hat1 = self.sar_model_03_sme_fun(param1)

            return y_hat1, param1

        params = []
        y_hat = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for a, b in executor.map(model_inversion, range(y1.shape[0])):
                y_hat.append(a.flatten())
                params.append(b)

        if self.size_p == 1:
            param1tmp = param0
            param1tmp[0 : self.size_p * 6 : self.size_p] = params[:6]
            param1tmp[self.size_p * 6 :] = params[6:]
            params = param1tmp

        params = np.array(params)
        y_hat = np.array(y_hat)
        return params, y_hat

    def sar_model_04a_fun(self, x):
        A = self.x_P(x[0 * self.size_p : 1 * self.size_p][None, :])
        B = self.x_P(x[1 * self.size_p : 2 * self.size_p][None, :])
        C = self.x_P(x[2 * self.size_p : 3 * self.size_p][None, :])
        D = self.x_P(x[3 * self.size_p : 4 * self.size_p][None, :])
        a = self.x_P(x[4 * self.size_p : 5 * self.size_p][None, :])
        S = self.x_T(x[5 * self.size_p : 5 * self.size_p + self.size_t][None, :])
        W = self.x_S(
            x[
                5 * self.size_p
                + self.size_t : 5 * self.size_p
                + self.size_t
                + self.size_s
            ][None, :]
        )

        model_fun = (
            A * W**a * (1 - np.exp(-B * W * 0.1)) + D * S * np.exp(-B * W * 0.1) + C
        )
        return model_fun

    def sar_model_inverse_04a_noint(
        self, param0, y1, bounds, name="tmp", init_weights=[4, 1]
    ):
        """ """

        param0w = param0
        bounds0w = bounds

        def model_inversion(instance):
            y = y1[instance, :]
            param0aw = param0w[instance, :]
            bounds0aw = bounds0w[instance]

            wt = np.array(init_weights)
            result = minimize(
                lambda x: np.sum(
                    (
                        (
                            self.sar_model_04a_fun(x).reshape(-1, self.size_p)
                            - y.reshape(-1, self.size_p)
                        )
                        * wt[-self.size_p :]
                    )
                    ** 2
                ),
                param0aw,
                method="L-BFGS-B",
                bounds=bounds0aw,
            )
            # result = differential_evolution(lambda x: np.sum(
            #     ((self.model_03a_fun(W, x).reshape(-1, self.size_p) - y.reshape(-1, self.size_p)) * wt) ** 2
            # ), bounds)
            # print(instance)
            # print(result.success)
            param1 = result.x

            y_hat1 = self.sar_model_04a_fun(param1)

            return y_hat1, param1

        params = []
        y_hat = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for a, b in executor.map(model_inversion, range(y1.shape[0])):
                y_hat.append(a.flatten())
                params.append(b)

        params = np.array(params)
        y_hat = np.array(y_hat)
        return params, y_hat


class ModelSimulation:
    def __init__(self, z0_file, z1_file, valid_min=0, out_name="tmp"):
        """

        :param z0_file:
        :param z1_file:
        :param valid_min:
        :param out_name:
        """
        self.basemap_file = z0_file
        self.mask_file = z1_file
        self.valid_min = valid_min
        self.out_name = out_name
        self.out_agb = "{}_agb_0.tif".format(self.out_name)

    def build_noisy_agb(self, minmax=(5, 155), noise_W=30):
        """

        :param minmax:
        :param noise_W:
        :return:
        """
        self.out_agb = "{}_agb_0.tif".format(self.out_name)
        in0 = gdal.Open(self.basemap_file, gdal.GA_ReadOnly)
        y_dim = in0.RasterYSize
        x_dim = in0.RasterXSize
        W0 = (
            noise_W * np.random.randn(y_dim, x_dim)
            + np.linspace(minmax[0], minmax[1], y_dim)[:, None]
        )
        W0[W0 < 0] = 0

        driver = gdal.GetDriverByName("GTiff")
        ds = driver.CreateCopy(self.out_agb, in0, 0, ["COMPRESS=LZW", "PREDICTOR=2"])
        ds.GetRasterBand(1).WriteArray(W0)
        ds.FlushCache()  # Write to disk.
        ds = None

        return self.out_agb

    def build_noisy_alos(self, param0, m_t=2, noise_H=0.04, noise_V=0.01):
        """

        :param param0:
        :param m_t:
        :param noise_H:
        :param noise_V:
        :return:
        """
        out_radar_list = [
            "{}_alossim_t{}_{}.tif".format(self.out_name, i, j)
            for i in range(m_t)
            for j in ("HV", "HH")
        ]
        in0 = gdal.Open(self.basemap_file, gdal.GA_ReadOnly)
        y_dim = in0.RasterYSize
        x_dim = in0.RasterXSize
        if not os.path.exists("tmp"):
            os.mkdir("tmp")
        df0 = pp.array_reshape_rolling(
            [self.out_agb],
            self.basemap_file,
            name=f"tmp/tmp_agb_{os.path.basename(self.out_name)}",
            m=0,
            n=0,
            valid_min=self.valid_min,
        )
        W1 = df0.iloc[:, 1:].values
        n_obs = W1.shape[0]

        param0c = np.tile(param0, (n_obs, 1))
        param0c = np.concatenate((param0c, W1), axis=1)

        model_00 = Retrieval(2, m_t, 1)
        y_hat = []
        for i in range(n_obs):
            y_hat.append(model_00.sar_model_03_fun(param0c[i, :]))
        y_hat0 = np.concatenate(y_hat, axis=0)
        y_hat1 = y_hat0.reshape([n_obs, m_t, 2])
        y_hv = y_hat1[:, :, 0] + noise_V * np.random.randn(n_obs, m_t)
        y_hh = y_hat1[:, :, 1] + noise_H * np.random.randn(n_obs, m_t)
        y_hat1 = np.stack((y_hv, y_hh), axis=-1)
        y_hat1 = y_hat1.reshape([n_obs, -1])

        for i in range(y_hat1.shape[1]):
            array0 = in0.GetRasterBand(1).ReadAsArray()
            array0[array0 > self.valid_min] = y_hat1[:, i]
            driver = gdal.GetDriverByName("GTiff")
            ds = driver.CreateCopy(
                out_radar_list[i], in0, 0, ["COMPRESS=LZW", "PREDICTOR=2"]
            )
            ds.GetRasterBand(1).WriteArray(array0)
            ds.FlushCache()  # Write to disk.
            ds = None
        in0 = None

        return out_radar_list

    def data_noisy_w_vs_alos(
        self,
        m_t=1,
        w_noise=30,
        h_noise=0.04,
        v_noise=0.01,
        s_noise=0.001,
        mask_file=None,
        n_1=1,
    ):
        if mask_file is None:
            mask_file = self.mask_file
        # define W0 (AGB) raster
        w_file = self.build_noisy_agb(noise_W=w_noise)

        # retrieve HH/HV from W1
        in_path = os.path.dirname(self.mask_file)
        df0 = pd.read_csv("{}/model_sim_param0_s13.csv".format(in_path))
        param0 = df0.iloc[0, 1:].values
        # param0 = np.append(param0, np.repeat(param0[-1], m_t - 1))
        param0 = np.append(
            param0, np.repeat(param0[-1], m_t - 1) + np.random.randn(m_t - 1) * s_noise
        )

        out_radar_list = self.build_noisy_alos(
            param0, m_t=m_t, noise_H=h_noise, noise_V=v_noise
        )
        # print(out_radar_list)

        if not os.path.exists("tmp"):
            os.mkdir("tmp")
        df_agb = pp.array_reshape_rolling(
            [self.out_agb],
            mask_file,
            name=f"tmp/tmp_w_{os.path.basename(self.out_name)}",
            m=n_1,
            n=n_1,
            valid_min=self.valid_min,
        )
        W1 = df_agb.iloc[:, 1:].values

        df_alos = pp.array_reshape_rolling(
            out_radar_list,
            mask_file,
            name=f"tmp/tmp_alos_{os.path.basename(self.out_name)}",
            m=n_1,
            n=n_1,
            valid_min=self.valid_min,
        )
        y_hat = df_alos.iloc[:, 1:].values
        y_hat0 = y_hat.reshape([W1.shape[0], m_t * (n_1 * 2 + 1) ** 2, 2])

        return W1, y_hat0

    def inversion_recursive_ws_sim(
        self,
        m_t=2,
        w_noise=30,
        h_noise=0.04,
        v_noise=0.01,
        s_noise=0.001,
        mask_file=None,
        n_1=0,
    ):
        """

        :return: W_mean: AGB predictions (n_obs, m_t)
        """
        if mask_file is None:
            mask_file = self.mask_file
        n_2 = n_1 * 2 + 1
        in0 = gdal.Open(mask_file, gdal.GA_ReadOnly)
        array0 = in0.GetRasterBand(1).ReadAsArray()
        struct2 = ndimage.generate_binary_structure(2, 2)
        array0 = ndimage.binary_dilation(
            array0, structure=struct2, iterations=n_2
        ).astype(array0.dtype)
        mask_ws1 = "{}_mask_ws1.tif".format(self.out_name)
        driver = gdal.GetDriverByName("GTiff")
        ds = driver.CreateCopy(mask_ws1, in0, 0, ["COMPRESS=LZW", "PREDICTOR=2"])
        ds.GetRasterBand(1).WriteArray(array0)
        ds.FlushCache()  # Write to disk.
        ds = None
        in0 = None

        W1, y_hat0 = self.data_noisy_w_vs_alos(
            m_t=m_t,
            w_noise=w_noise,
            h_noise=h_noise,
            v_noise=v_noise,
            s_noise=s_noise,
            mask_file=mask_ws1,
            n_1=n_1,
        )
        print("Dimension of W: {}".format(W1.shape))
        print("Dimension of y: {}".format(y_hat0.shape))
        nxn = (n_1 * 2 + 1) ** 2
        n_obs = W1.shape[0]
        z1 = y_hat0.reshape((n_obs, nxn, m_t, 2))
        W1 = W1.reshape((n_obs, nxn))[:, -nxn // 2]

        W_mean, _ = self.inversion_recursive_ws(
            z1, mask_file, mask_ws1, m_t=m_t, n_1=n_1
        )

        in0 = gdal.Open(mask_ws1, gdal.GA_ReadOnly)
        in1 = gdal.Open(mask_file, gdal.GA_ReadOnly)
        array0 = in0.GetRasterBand(1).ReadAsArray()
        array0[array0 > self.valid_min] = W1
        array1 = in1.GetRasterBand(1).ReadAsArray()
        W1 = array0[array1 > self.valid_min]
        in0 = None
        in1 = None

        return W_mean, W1

    def inversion_recursive_ws(self, z1, mask_file=None, mask_ws1=None, m_t=2, n_1=0):
        """

        :return: W_mean: AGB predictions (n_obs, m_t)
        """
        if mask_file is None:
            mask_file = self.mask_file
        if mask_ws1 is None:
            n_2 = n_1 * 2 + 1
            in0 = gdal.Open(mask_file, gdal.GA_ReadOnly)
            array0 = in0.GetRasterBand(1).ReadAsArray()
            struct2 = ndimage.generate_binary_structure(2, 2)
            array0 = ndimage.binary_dilation(
                array0, structure=struct2, iterations=n_2
            ).astype(array0.dtype)
            mask_ws1 = "{}_mask_ws1.tif".format(self.out_name)
            driver = gdal.GetDriverByName("GTiff")
            ds = driver.CreateCopy(mask_ws1, in0, 0, ["COMPRESS=LZW", "PREDICTOR=2"])
            ds.GetRasterBand(1).WriteArray(array0)
            ds.FlushCache()  # Write to disk.
            ds = None
            in0 = None

        in_path = os.path.dirname(mask_file)
        df0 = pd.read_csv("{}/model_sim_param0_s13.csv".format(in_path))
        param0 = df0.iloc[0, 1:].values

        n_obs = z1.shape[0]
        nxn = z1.shape[1]
        in0 = gdal.Open(mask_ws1, gdal.GA_ReadOnly)
        in1 = gdal.Open(mask_file, gdal.GA_ReadOnly)
        W_mean = []
        t_mean = []
        kn = 1
        for i in range(m_t):
            start = time.time()
            print("Scene {} of {}".format(i + 1, m_t))
            z = z1[:, :, i, :]
            mean_hv = z[:, :, 0]
            param0w = (100 * mean_hv) ** 2  # dimension: (n_obs, nxn)
            param0c = np.tile(param0, (n_obs, 1))
            param0c = np.concatenate((param0c, param0w), axis=1)

            for k in range(kn):
                # print('k = {}'.format(k))
                bound0c = [
                    [(w0 * 0.9999, w0 * 1.0001) for w0 in param0c[iw, : 13 + nxn // 2]]
                    + [
                        (w0 * 0.2, w0 * 1.8)
                        for w0 in param0c[iw, 13 + nxn // 2 : 13 + nxn // 2 + 1]
                    ]
                    + [
                        (w0 * 0.9999, w0 * 1.0001)
                        for w0 in param0c[iw, 13 + nxn // 2 + 1 :]
                    ]
                    for iw in range(param0c.shape[0])
                ]
                model_00 = Retrieval(2, 1, nxn)
                params, z_hat = model_00.sar_model_inverse_03(param0c, z, bound0c)
                w0 = params[:, -nxn // 2]

                array0 = in0.GetRasterBand(1).ReadAsArray()
                array0[array0 > self.valid_min] = w0
                if not os.path.exists("tmp"):
                    os.mkdir("tmp")
                try:
                    os.remove(
                        "tmp/tmp_agb_{}_{}_{}.tif".format(
                            os.path.basename(self.out_name), k, i
                        )
                    )
                except OSError:
                    pass
                driver = gdal.GetDriverByName("GTiff")
                ds = driver.CreateCopy(
                    "tmp/tmp_agb_{}_{}_{}.tif".format(
                        os.path.basename(self.out_name), k, i
                    ),
                    in0,
                    0,
                    ["COMPRESS=LZW", "PREDICTOR=2"],
                )
                ds.GetRasterBand(1).WriteArray(array0)
                ds.FlushCache()  # Write to disk.
                ds = None

                df_agb = pp.array_reshape_rolling(
                    [
                        "tmp/tmp_agb_{}_{}_{}.tif".format(
                            os.path.basename(self.out_name), k, i
                        )
                    ],
                    mask_ws1,
                    name="tmp/tmp_agb_nxn_{}_{}_{}".format(
                        os.path.basename(self.out_name), k, i
                    ),
                    m=n_1,
                    n=n_1,
                    valid_min=self.valid_min,
                )
                param0w = df_agb.iloc[:, 1:].values
                param0w[np.isnan(param0w)] = (100 * mean_hv[np.isnan(param0w)]) ** 2

                param0c = np.concatenate((param0c[:, :13], param0w), axis=1)

                """
                Update Parameters...
                """
                bound0c = [
                    [(w0 * 0.9999, w0 * 1.0001) for w0 in param0c[iw, :6]]
                    + [(w0 * 0.9999, w0 * 1.0001) for w0 in param0c[iw, 6:12]]
                    + [(w0 * 0.5, w0 * 1.5) for w0 in param0c[iw, 12:13]]
                    + [(w0 * 0.9999, w0 * 1.0001) for w0 in param0c[iw, 13:]]
                    for iw in range(param0c.shape[0])
                ]
                model_00 = Retrieval(2, 1, nxn)
                param0c, y_hat = model_00.sar_model_inverse_03(param0c, z, bound0c)
                density_scatter_plot(
                    z.flatten(),
                    y_hat.flatten(),
                    x_label="Measured Backscatter",
                    y_label="Predicted Backscatter",
                    x_limit=(0, 0.6),
                    y_limit=(0, 0.6),
                    file_name=(self.out_name + "_ws1_y_s{}_k{}.png").format(i, k),
                )

                array0 = in0.GetRasterBand(1).ReadAsArray()
                array0[array0 > self.valid_min] = param0c[:, -nxn // 2]

            array1 = in1.GetRasterBand(1).ReadAsArray()
            W_mean1 = array0[array1 > self.valid_min]
            W_mean.append(W_mean1)
            end = time.time()
            t1 = end - start
            t_mean.append(t1)

        t_mean = np.array(t_mean)
        W_mean = np.array(W_mean).T

        in0 = None
        in1 = None

        return W_mean, t_mean


class FieldRetrieval:
    def __init__(self, z0_file, valid_min=0, out_name="tmp"):
        """

        :param z0_file:
        :param z1_file:
        :param valid_min:
        :param out_name:
        """
        self.mask_file = z0_file
        self.valid_min = valid_min
        self.out_name = out_name
        self.out_agb = "{}_agb_0.tif".format(self.out_name)

    def data_cleaner(
        self, out_radar_list, agb_file=None, mask_file=None, w1_noise=10, w2_noise=20
    ):
        """

        :param out_radar_list: time series of HV/HH files (hv_t0, hh_t0, hv_t1, hh_t1, ...)
        :param agb_file: agb measurements, if any
        :param mask_file:
        :param n_1:
        :return:
        """
        if mask_file is None:
            mask_file = self.mask_file
        if not os.path.exists("tmp"):
            os.mkdir("tmp")

        m_t = len(out_radar_list) // 2
        df_alos = pp.array_reshape_rolling(
            out_radar_list,
            mask_file,
            name=f"tmp/tmp_alos_{os.path.basename(self.out_name)}",
            m=0,
            n=0,
            valid_min=self.valid_min,
        )
        # print(df_alos.iloc[:, 1:])
        z_alos = df_alos.iloc[:, 1:].values
        if out_radar_list[0][:3] == "./A":
            z_alos = z_alos**2 / 199526231
        z1 = z_alos.reshape([-1, m_t * 2])

        if agb_file is not None:
            df_agb = pp.array_reshape_rolling(
                [agb_file],
                mask_file,
                name=f"tmp/tmp_w_{os.path.basename(self.out_name)}",
                m=0,
                n=0,
                valid_min=self.valid_min,
            )
            W1 = df_agb.iloc[:, 1].values
        else:
            W1 = None

        x0, y0, idx = data_clean(z1, W1, w1_noise=w1_noise, w2_noise=w2_noise)

        in0 = gdal.Open(mask_file, gdal.GA_ReadOnly)
        array0 = in0.GetRasterBand(1).ReadAsArray()
        lc = array0[array0 > self.valid_min]
        lc[~idx] = self.valid_min
        array0[array0 > self.valid_min] = lc

        mask_ws0 = "{}_mask_ws0.tif".format(self.out_name)
        driver = gdal.GetDriverByName("GTiff")
        ds = driver.CreateCopy(mask_ws0, in0, 0, ["COMPRESS=LZW", "PREDICTOR=2"])
        ds.GetRasterBand(1).WriteArray(array0)
        ds.FlushCache()  # Write to disk.
        ds = None
        in0 = None

        return x0, y0, mask_ws0

    def data_cleaner_2(
        self, out_radar_list, agb_file=None, mask_file=None, w2_noise=10
    ):
        """

        :param out_radar_list: time series of HV/HH files (hv_t0, hh_t0, hv_t1, hh_t1, ...)
        :param agb_file: agb measurements, if any
        :param mask_file:
        :param n_1:
        :return:
        """
        if mask_file is None:
            mask_file = self.mask_file
        if not os.path.exists("tmp"):
            os.mkdir("tmp")

        m_t = len(out_radar_list) // 2
        df_alos = pp.array_reshape_rolling(
            out_radar_list,
            mask_file,
            name=f"tmp/tmp_alos_{os.path.basename(self.out_name)}",
            m=0,
            n=0,
            valid_min=self.valid_min,
        )
        # print(df_alos.iloc[:, 1:])
        z_alos = df_alos.iloc[:, 1:].values
        if out_radar_list[0][:3] == "./A":
            z_alos = z_alos**2 / 199526231
        z1 = z_alos.reshape([-1, m_t * 2])

        if agb_file is not None:
            df_agb = pp.array_reshape_rolling(
                [agb_file],
                mask_file,
                name=f"tmp/tmp_w_{os.path.basename(self.out_name)}",
                m=0,
                n=0,
                valid_min=self.valid_min,
            )
            W1 = df_agb.iloc[:, 1].values
        else:
            W1 = None

        # print(w2_noise/100)
        x0, y0, idx = data_clean_2(z1, W1, w1_noise=w2_noise / 100)

        in0 = gdal.Open(mask_file, gdal.GA_ReadOnly)
        array0 = in0.GetRasterBand(1).ReadAsArray()
        lc = array0[array0 > self.valid_min]
        lc[~idx] = self.valid_min
        array0[array0 > self.valid_min] = lc

        mask_ws0 = "{}_mask_ws0.tif".format(self.out_name)
        driver = gdal.GetDriverByName("GTiff")
        ds = driver.CreateCopy(mask_ws0, in0, 0, ["COMPRESS=LZW", "PREDICTOR=2"])
        ds.GetRasterBand(1).WriteArray(array0)
        ds.FlushCache()  # Write to disk.
        ds = None
        in0 = None

        return x0, y0, mask_ws0

    def inversion_setup(self, out_radar_list, agb_file=None, mask_file=None, n_1=0):
        """
        set up the inversion of AGB retrieval.
        :param out_radar_list: time series of HV/HH files (hv_t0, hh_t0, hv_t1, hh_t1, ...)
        :param agb_file: agb measurements, if any
        :param mask_file:
        :param n_1:
        :return:
            z1 (n_obs, nxn, n_time, 2_pols),
            W1 (if agb exists, central pixel of nxn),
            mask_ws1 (dilated mask depending on nxn)
        """
        if mask_file is None:
            mask_file = self.mask_file
        n_2 = n_1 * 2 + 1
        in0 = gdal.Open(mask_file, gdal.GA_ReadOnly)
        array0 = in0.GetRasterBand(1).ReadAsArray()
        struct2 = ndimage.generate_binary_structure(2, 2)
        array0 = ndimage.binary_dilation(
            array0, structure=struct2, iterations=n_2
        ).astype(array0.dtype)
        mask_ws1 = "{}_mask_ws1.tif".format(self.out_name)
        driver = gdal.GetDriverByName("GTiff")
        ds = driver.CreateCopy(mask_ws1, in0, 0, ["COMPRESS=LZW", "PREDICTOR=2"])
        ds.GetRasterBand(1).WriteArray(array0)
        ds.FlushCache()  # Write to disk.
        ds = None
        in0 = None

        if not os.path.exists("tmp"):
            os.mkdir("tmp")

        m_t = len(out_radar_list) // 2
        nxn = (n_1 * 2 + 1) ** 2
        df_alos = pp.array_reshape_rolling(
            out_radar_list,
            mask_ws1,
            name=f"tmp/tmp_alos_{os.path.basename(self.out_name)}",
            m=n_1,
            n=n_1,
            valid_min=self.valid_min,
        )
        z_alos = df_alos.iloc[:, 1:].values
        if out_radar_list[0][:3] == "./A":
            z_alos = z_alos**2 / 199526231
        z1 = z_alos.reshape([-1, nxn, m_t, 2])

        if agb_file is not None:
            df_agb = pp.array_reshape_rolling(
                [agb_file],
                mask_ws1,
                name=f"tmp/tmp_w_{os.path.basename(self.out_name)}",
                m=n_1,
                n=n_1,
                valid_min=self.valid_min,
            )
            W1 = df_agb.iloc[:, 1:].values
            W1 = W1.reshape((-1, nxn))[:, -nxn // 2][:, None]
        else:
            W1 = None

        return z1, W1, mask_ws1

    def inversion_return_valid(self, W1, mask_ws1, mask_file=None):
        """
        reshape n_obs -> n_obs2 for valid pixels using the mask file
        :param W1: reshaped input signal (n_obs, bands)
        :param mask_ws1: mask from inversion_setup (# of valid pixels: n_obs)
        :param mask_file: original mask file (# of valid pixels: n_obs2)
        :return: W0 following original mask (n_obs2, bands)
        """
        if mask_file is None:
            mask_file = self.mask_file

        in0 = gdal.Open(mask_ws1, gdal.GA_ReadOnly)
        in1 = gdal.Open(mask_file, gdal.GA_ReadOnly)
        array1 = in1.GetRasterBand(1).ReadAsArray()

        if len(W1.shape) == 1:
            W1 = W1[:, None]
        # print(W1.shape)

        W0 = []
        for i in range(W1.shape[1]):
            array0 = in0.GetRasterBand(1).ReadAsArray().astype(float)
            array0[array0 > self.valid_min] = W1[:, i]
            W = array0[array1 > self.valid_min]
            W0.append(W)
        W0 = np.array(W0).T
        # print(W0.shape)

        in0 = None
        in1 = None

        return W0

    def params_calibration(self, z1, W1, out_prefix, init_weights=[4, 1]):
        """

        :param z1: (n_obs, nxn, n_time, 2_pols[hv/hh])
        :param W1:
        :return:
        """
        mean_hv = np.nanmean(z1[:, :, :, 0].reshape([z1.shape[0], -1]), axis=1)
        mean_hh = np.nanmean(z1[:, :, :, 1].reshape([z1.shape[0], -1]), axis=1)

        mean_x = np.stack([mean_hv, mean_hh]).T
        # print(mean_x.shape)
        model_00 = Retrieval(2, 1, mean_x.shape[0])
        if len(W1.shape) == 1:
            W1 = W1[:, None]
        W = model_00.x_S(W1.T)
        y = mean_x.reshape([1, -1])
        bounds = (
            (
                (0.0001, 0.5),
                (0.0001, 0.5),
            )
            + 2 * ((0.0001, 0.9999),)
            + 2 * ((0.0001, 0.9999),)
            + 2 * ((0.0001, 1.9999),)
            + 2 * ((0.0001, 3.9999),)
            + (
                (1, 1.0001),
                (0, 25),
            )
            + 1 * ((0.0001, 0.5),)
        )
        print(y.shape)
        print(W.shape)
        params, y_hat = model_00.model_inverse_03a(y, W, bounds, init_weights=[4, 1])
        print(f"Retrieved parameters: \n {params[0]}")
        param_names = [
            "A_HV",
            "A_HH",
            "B_HV",
            "B_HH",
            "C_HV",
            "C_HH",
            "alpha_HV",
            "alpha_HH",
            "gamma_HV",
            "gamma_HH",
            "D_HV",
            "D_HH",
        ] + ["S"]
        in_path = os.path.dirname(self.mask_file)
        param0_file = f"{out_prefix}_model_sim_param0_s13.csv"
        df = pd.DataFrame(params, columns=param_names)
        df.to_csv(param0_file)
        density_scatter_plot(
            y.flatten(),
            y_hat.flatten(),
            x_label="Measured Backscatter",
            y_label="Predicted Backscatter",
            x_limit=(0, 0.6),
            y_limit=(0, 0.6),
        )
        return param0_file

    def params_calibration_cband(self, z1, W1, out_prefix, init_weights=[4, 1]):
        """

        :param z1: (n_obs, nxn, n_time, 2_pols[hv/hh])
        :param W1:
        :return:
        """
        mean_hv = np.nanmean(z1[:, :, :, 0].reshape([z1.shape[0], -1]), axis=1)
        mean_hh = np.nanmean(z1[:, :, :, 1].reshape([z1.shape[0], -1]), axis=1)

        mean_x = np.stack([mean_hv, mean_hh]).T
        # print(mean_x.shape)
        model_00 = Retrieval(2, 1, mean_x.shape[0])
        if len(W1.shape) == 1:
            W1 = W1[:, None]
        W = model_00.x_S(W1.T)
        y = mean_x.reshape([1, -1])
        bounds = (
            (
                (0.0001, 0.6),
                (0.0001, 0.6),
            )
            + 2 * ((0.0001, 0.9999),)
            + 2 * ((0.0001, 0.9999),)
            + 2 * ((-0.9999, -0.0001),)
            + 2 * ((0.0001, 3.9999),)
            + (
                (1, 1.0001),
                (0, 25),
            )
            + 1 * ((0.0001, 0.5),)
        )
        print(y.shape)
        print(W.shape)
        params, y_hat = model_00.model_inverse_03a(y, W, bounds, init_weights=[4, 1])
        print(f"Retrieved parameters: \n {params[0]}")
        param_names = [
            "A_HV",
            "A_HH",
            "B_HV",
            "B_HH",
            "C_HV",
            "C_HH",
            "alpha_HV",
            "alpha_HH",
            "gamma_HV",
            "gamma_HH",
            "D_HV",
            "D_HH",
        ] + ["S"]
        in_path = os.path.dirname(self.mask_file)
        param0_file = f"{out_prefix}_model_sim_param0_s13.csv"
        df = pd.DataFrame(params, columns=param_names)
        df.to_csv(param0_file)
        density_scatter_plot(
            y.flatten(),
            y_hat.flatten(),
            x_label="Measured Backscatter",
            y_label="Predicted Backscatter",
            x_limit=(0, 0.6),
            y_limit=(0, 0.6),
        )
        return param0_file

    def params_calibration_cband1(self, z1, W1, out_prefix, init_weights=[4, 1]):
        """

        :param z1: (n_obs, nxn, n_time, 2_pols[hv/hh])
        :param W1:
        :return:
        """
        mean_hv = np.nanmean(z1[:, :, :, 0].reshape([z1.shape[0], -1]), axis=1)
        mean_hh = np.nanmean(z1[:, :, :, 1].reshape([z1.shape[0], -1]), axis=1)

        mean_x = np.stack([mean_hv, mean_hh]).T
        # print(mean_x.shape)
        model_00 = Retrieval(2, 1, mean_x.shape[0])
        if len(W1.shape) == 1:
            W1 = W1[:, None]
        W = model_00.x_S(W1.T)
        y = mean_x.reshape([1, -1])
        bounds = (
            (
                (0.0001, 0.6),
                (0.0001, 0.6),
            )
            + 2 * ((0.0001, 0.9999),)
            + 2 * ((0.0001, 0.9999),)
            + 2 * ((-0.9999, 0.9999),)
            + 2 * ((0.0001, 3.9999),)
            + (
                (1, 1.0001),
                (0, 25),
            )
            + 1 * ((0.0001, 0.5),)
        )
        print(y.shape)
        print(W.shape)
        params, y_hat = model_00.model_inverse_03a(y, W, bounds, init_weights=[4, 1])
        print(f"Retrieved parameters: \n {params[0]}")
        param_names = [
            "A_HV",
            "A_HH",
            "B_HV",
            "B_HH",
            "C_HV",
            "C_HH",
            "alpha_HV",
            "alpha_HH",
            "gamma_HV",
            "gamma_HH",
            "D_HV",
            "D_HH",
        ] + ["S"]
        in_path = os.path.dirname(self.mask_file)
        param0_file = f"{out_prefix}_model_sim_param0_s13.csv"
        df = pd.DataFrame(params, columns=param_names)
        df.to_csv(param0_file)
        density_scatter_plot(
            y.flatten(),
            y_hat.flatten(),
            x_label="Measured Backscatter",
            y_label="Predicted Backscatter",
            x_limit=(0, 0.6),
            y_limit=(0, 0.6),
        )
        return param0_file

    def params_calibration_cband_single(self, z1, W1, out_prefix, init_weights=[4, 1]):
        """

        :param z1: (n_obs, nxn, n_time, 2_pols[hv/hh])
        :param W1:
        :return:
        """
        mean_hv = np.nanmean(z1[:, :, :, 0].reshape([z1.shape[0], -1]), axis=1)
        mean_x = mean_hv[:, None]

        model_00 = Retrieval(1, 1, mean_x.shape[0])
        if len(W1.shape) == 1:
            W1 = W1[:, None]
        W = model_00.x_S(W1.T)
        y = mean_x.reshape([1, -1])
        bounds = (
            1 * ((0.0001, 0.6),)
            + 1 * ((0.0001, 0.9999),)
            + 1 * ((0.0001, 0.9999),)
            + 1 * ((-0.9999, 0.9999),)
            + 1 * ((0.0001, 3.9999),)
            + 1 * ((1, 1.0001),)
            + 1 * ((0.0001, 0.5),)
        )
        print(y.shape)
        print(W.shape)
        params, y_hat = model_00.model_inverse_03a(y, W, bounds, init_weights=[4, 1])
        print(f"Retrieved parameters: \n {params[0]}")
        param_names = [
            "A_HV",
            "B_HV",
            "C_HV",
            "alpha_HV",
            "gamma_HV",
            "D_HV",
        ] + ["S"]
        in_path = os.path.dirname(self.mask_file)
        param0_file = f"{out_prefix}_model_sim_param0_s7.csv"
        df = pd.DataFrame(params, columns=param_names)
        df.to_csv(param0_file)
        density_scatter_plot(
            y.flatten(),
            y_hat.flatten(),
            x_label="Measured Backscatter",
            y_label="Predicted Backscatter",
            x_limit=(0, 0.6),
            y_limit=(0, 0.6),
        )
        return param0_file

    def params_calibration_cband_04a_noint(
        self, z1, W1, out_prefix, init_weights=[4, 1]
    ):
        """

        :param z1: (n_obs, nxn, n_time, 2_pols[hv/hh])
        :param W1:
        :return:
        """
        mean_hv = np.nanmean(z1[:, :, :, 0].reshape([z1.shape[0], -1]), axis=1)
        mean_hh = np.nanmean(z1[:, :, :, 1].reshape([z1.shape[0], -1]), axis=1)

        mean_x = np.stack([mean_hv, mean_hh]).T
        # print(mean_x.shape)
        model_00 = Retrieval(2, 1, mean_x.shape[0])
        if len(W1.shape) == 1:
            W1 = W1[:, None]
        W = model_00.x_S(W1.T)
        y = mean_x.reshape([1, -1])
        bounds = (
            2 * ((0.00001, 0.99),)  # A
            + 2 * ((0.0001, 0.99),)  # B
            + 2 * ((0.00001, 0.99),)  # C
            + 1 * ((1.0, 1.00001),)
            + 1 * ((0.00001, 29.9999),)  # D
            + 2 * ((0.0, 0.00001),)  # a
            + 1 * ((0.00001, 0.5),)  # S
        )
        print(y.shape)
        print(W.shape)
        params, y_hat = model_00.model_inverse_04a_noint(
            y, W, bounds, init_weights=init_weights
        )
        print(f"Retrieved parameters: \n {params[0]}")
        param_names = [
            "A_HV",
            "A_HH",
            "B_HV",
            "B_HH",
            "C_HV",
            "C_HH",
            "D_HV",
            "D_HH",
            "a_HV",
            "a_HH",
        ] + ["S"]
        in_path = os.path.dirname(self.mask_file)
        param0_file = f"{out_prefix}_model_sim_param0_s13.csv"
        df = pd.DataFrame(params, columns=param_names)
        df.to_csv(param0_file)

        x0 = y.reshape([-1, 2])
        y0 = y_hat.reshape([-1, 2])
        density_scatter_plot(
            x0[:, 0],
            y0[:, 0],
            x_label="Measured VH",
            y_label="Predicted VH",
            x_limit=(0, max(x0[:, 0]) * 1.1),
            y_limit=(0, max(x0[:, 0]) * 1.1),
        )
        density_scatter_plot(
            x0[:, 1],
            y0[:, 1],
            x_label="Measured VV",
            y_label="Predicted VV",
            x_limit=(0, max(x0[:, 1]) * 1.1),
            y_limit=(0, max(x0[:, 1]) * 1.1),
        )
        return param0_file

    def params_calibration_cband_04a_noint_nos(
        self, z1, W1, out_prefix, init_weights=[4, 1]
    ):
        """

        :param z1: (n_obs, nxn, n_time, 2_pols[hv/hh])
        :param W1:
        :return:
        """
        mean_hv = np.nanmean(z1[:, :, :, 0].reshape([z1.shape[0], -1]), axis=1)
        mean_hh = np.nanmean(z1[:, :, :, 1].reshape([z1.shape[0], -1]), axis=1)

        mean_x = np.stack([mean_hv, mean_hh]).T
        # print(mean_x.shape)
        model_00 = Retrieval(2, 1, mean_x.shape[0])
        if len(W1.shape) == 1:
            W1 = W1[:, None]
        W = model_00.x_S(W1.T)
        y = mean_x.reshape([1, -1])
        bounds = (
            1 * ((0.00001, 0.1),)  # A
            + 1 * ((0.00001, 0.4),)  # A
            # 2 * ((0.00001, 0.99),)    # A
            # + 2 * ((0.0001, 0.99),)    # B
            + 2 * ((0.0001, 0.99),)  # B
            + 2 * ((0.00001, 0.99),)  # C
            + 2 * ((1.0, 1.00001),)  # D
            + 2 * ((0.0, 0.00001),)  # a
            + 1 * ((0.0, 0.00001),)  # S
        )
        print(y.shape)
        print(W.shape)
        params, y_hat = model_00.model_inverse_04a_noint(
            y, W, bounds, init_weights=init_weights
        )
        print(f"Retrieved parameters: \n {params[0]}")
        param_names = [
            "A_HV",
            "A_HH",
            "B_HV",
            "B_HH",
            "C_HV",
            "C_HH",
            "D_HV",
            "D_HH",
            "a_HV",
            "a_HH",
        ] + ["S"]
        in_path = os.path.dirname(self.mask_file)
        param0_file = f"{out_prefix}_model_sim_param0_s13.csv"
        df = pd.DataFrame(params, columns=param_names)
        df.to_csv(param0_file)

        x0 = y.reshape([-1, 2])
        y0 = y_hat.reshape([-1, 2])
        density_scatter_plot(
            x0[:, 0],
            y0[:, 0],
            x_label="Measured VH",
            y_label="Predicted VH",
            x_limit=(0, max(x0[:, 0]) * 1.1),
            y_limit=(0, max(x0[:, 0]) * 1.1),
        )
        density_scatter_plot(
            x0[:, 1],
            y0[:, 1],
            x_label="Measured VV",
            y_label="Predicted VV",
            x_limit=(0, max(x0[:, 1]) * 1.1),
            y_limit=(0, max(x0[:, 1]) * 1.1),
        )
        return param0_file

    def params_calibration_cband_04a_noint_noc(
        self, z1, W1, out_prefix, init_weights=[4, 1]
    ):
        """

        :param z1: (n_obs, nxn, n_time, 2_pols[hv/hh])
        :param W1:
        :return:
        """
        mean_hv = np.nanmean(z1[:, :, :, 0].reshape([z1.shape[0], -1]), axis=1)
        mean_hh = np.nanmean(z1[:, :, :, 1].reshape([z1.shape[0], -1]), axis=1)

        mean_x = np.stack([mean_hv, mean_hh]).T
        # print(mean_x.shape)
        model_00 = Retrieval(2, 1, mean_x.shape[0])
        if len(W1.shape) == 1:
            W1 = W1[:, None]
        W = model_00.x_S(W1.T)
        y = mean_x.reshape([1, -1])
        bounds = (
            2 * ((0.00001, 0.99),)  # A
            + 2 * ((0.0001, 0.99),)  # B
            + 2 * ((0.0, 0.00001),)  # C
            + 1 * ((1.0, 1.00001),)
            + 1 * ((0.00001, 29.9999),)  # D
            + 2 * ((0.0, 0.00001),)  # a
            + 1 * ((0.00001, 0.5),)  # S
        )
        print(y.shape)
        print(W.shape)
        params, y_hat = model_00.model_inverse_04a_noint(
            y, W, bounds, init_weights=init_weights
        )
        print(f"Retrieved parameters: \n {params[0]}")
        param_names = [
            "A_HV",
            "A_HH",
            "B_HV",
            "B_HH",
            "C_HV",
            "C_HH",
            "D_HV",
            "D_HH",
            "a_HV",
            "a_HH",
        ] + ["S"]
        in_path = os.path.dirname(self.mask_file)
        param0_file = f"{out_prefix}_model_sim_param0_s13.csv"
        df = pd.DataFrame(params, columns=param_names)
        df.to_csv(param0_file)

        x0 = y.reshape([-1, 2])
        y0 = y_hat.reshape([-1, 2])
        density_scatter_plot(
            x0[:, 0],
            y0[:, 0],
            x_label="Measured VH",
            y_label="Predicted VH",
            x_limit=(0, max(x0[:, 0]) * 1.1),
            y_limit=(0, max(x0[:, 0]) * 1.1),
        )
        density_scatter_plot(
            x0[:, 1],
            y0[:, 1],
            x_label="Measured VV",
            y_label="Predicted VV",
            x_limit=(0, max(x0[:, 1]) * 1.1),
            y_limit=(0, max(x0[:, 1]) * 1.1),
        )
        return param0_file

    def params_calibration_cband_04a_noint_noc_walpha(
        self, z1, W1, out_prefix, init_weights=[4, 1]
    ):
        """

        :param z1: (n_obs, nxn, n_time, 2_pols[hv/hh])
        :param W1:
        :return:
        """
        mean_hv = np.nanmean(z1[:, :, :, 0].reshape([z1.shape[0], -1]), axis=1)
        mean_hh = np.nanmean(z1[:, :, :, 1].reshape([z1.shape[0], -1]), axis=1)

        mean_x = np.stack([mean_hv, mean_hh]).T
        # print(mean_x.shape)
        model_00 = Retrieval(2, 1, mean_x.shape[0])
        if len(W1.shape) == 1:
            W1 = W1[:, None]
        W = model_00.x_S(W1.T)
        y = mean_x.reshape([1, -1])
        bounds = (
            2 * ((0.00001, 0.99),)  # A
            + 2 * ((0.0001, 0.99),)  # B
            + 2 * ((0.0, 0.00001),)  # C
            + 1 * ((1.0, 1.00001),)
            + 1 * ((0.00001, 29.9999),)  # D
            + 2 * ((-0.9999, 0.9999),)  # a
            + 1 * ((0.00001, 0.5),)  # S
        )
        print(y.shape)
        print(W.shape)
        params, y_hat = model_00.model_inverse_04a_noint(
            y, W, bounds, init_weights=init_weights
        )
        print(f"Retrieved parameters: \n {params[0]}")
        param_names = [
            "A_HV",
            "A_HH",
            "B_HV",
            "B_HH",
            "C_HV",
            "C_HH",
            "D_HV",
            "D_HH",
            "a_HV",
            "a_HH",
        ] + ["S"]
        in_path = os.path.dirname(self.mask_file)
        param0_file = f"{out_prefix}_model_sim_param0_s13.csv"
        df = pd.DataFrame(params, columns=param_names)
        df.to_csv(param0_file)

        x0 = y.reshape([-1, 2])
        y0 = y_hat.reshape([-1, 2])
        density_scatter_plot(
            x0[:, 0],
            y0[:, 0],
            x_label="Measured VH",
            y_label="Predicted VH",
            x_limit=(0, max(x0[:, 0]) * 1.1),
            y_limit=(0, max(x0[:, 0]) * 1.1),
        )
        density_scatter_plot(
            x0[:, 1],
            y0[:, 1],
            x_label="Measured VV",
            y_label="Predicted VV",
            x_limit=(0, max(x0[:, 1]) * 1.1),
            y_limit=(0, max(x0[:, 1]) * 1.1),
        )
        return param0_file

    def params_calibration_cband_04a_noint_nos_walpha(
        self, z1, W1, out_prefix, init_weights=[4, 1]
    ):
        """

        :param z1: (n_obs, nxn, n_time, 2_pols[hv/hh])
        :param W1:
        :return:
        """
        mean_hv = np.nanmean(z1[:, :, :, 0].reshape([z1.shape[0], -1]), axis=1)
        mean_hh = np.nanmean(z1[:, :, :, 1].reshape([z1.shape[0], -1]), axis=1)

        mean_x = np.stack([mean_hv, mean_hh]).T
        # print(mean_x.shape)
        model_00 = Retrieval(2, 1, mean_x.shape[0])
        if len(W1.shape) == 1:
            W1 = W1[:, None]
        W = model_00.x_S(W1.T)
        y = mean_x.reshape([1, -1])
        bounds = (
            2 * ((0.00001, 0.99),)  # A
            + 2 * ((0.0001, 0.99),)  # B
            + 2 * ((0.00001, 0.99),)  # C
            + 1 * ((1.0, 1.00001),)
            + 1 * ((0.00001, 29.9999),)  # D
            + 2 * ((-0.9999, 0.9999),)  # a
            + 1 * ((0.0, 0.00001),)  # S
        )
        print(y.shape)
        print(W.shape)
        params, y_hat = model_00.model_inverse_04a_noint(
            y, W, bounds, init_weights=init_weights
        )
        print(f"Retrieved parameters: \n {params[0]}")
        param_names = [
            "A_HV",
            "A_HH",
            "B_HV",
            "B_HH",
            "C_HV",
            "C_HH",
            "D_HV",
            "D_HH",
            "a_HV",
            "a_HH",
        ] + ["S"]
        in_path = os.path.dirname(self.mask_file)
        param0_file = f"{out_prefix}_model_sim_param0_s13.csv"
        df = pd.DataFrame(params, columns=param_names)
        df.to_csv(param0_file)

        x0 = y.reshape([-1, 2])
        y0 = y_hat.reshape([-1, 2])
        density_scatter_plot(
            x0[:, 0],
            y0[:, 0],
            x_label="Measured VH",
            y_label="Predicted VH",
            x_limit=(0, max(x0[:, 0]) * 1.1),
            y_limit=(0, max(x0[:, 0]) * 1.1),
        )
        density_scatter_plot(
            x0[:, 1],
            y0[:, 1],
            x_label="Measured VV",
            y_label="Predicted VV",
            x_limit=(0, max(x0[:, 1]) * 1.1),
            y_limit=(0, max(x0[:, 1]) * 1.1),
        )
        return param0_file

    def params_calibration_cband_04a_AB(self, z1, W1, out_prefix, init_weights=[4, 1]):
        """

        :param z1: (n_obs, nxn, n_time, 2_pols[hv/hh])
        :param W1:
        :return:
        """
        mean_hv = np.nanmean(z1[:, :, :, 0].reshape([z1.shape[0], -1]), axis=1)
        mean_hh = np.nanmean(z1[:, :, :, 1].reshape([z1.shape[0], -1]), axis=1)

        mean_x = np.stack([mean_hv, mean_hh]).T
        # print(mean_x.shape)
        model_00 = Retrieval(2, 1, mean_x.shape[0])
        if len(W1.shape) == 1:
            W1 = W1[:, None]
        W = model_00.x_S(W1.T)
        y = mean_x.reshape([1, -1])
        bounds = (
            2 * ((0.00001, 0.99),)  # A
            + 2 * ((0.0001, 0.99),)  # B
            + 2 * ((0.0, 0.00001),)  # C
            + 2 * ((1.0, 1.00001),)  # D
            + 2 * ((0.0, 0.00001),)  # a
            + 1 * ((0.0, 0.00001),)  # S
        )
        print(y.shape)
        print(W.shape)
        params, y_hat = model_00.model_inverse_04a_noint(
            y, W, bounds, init_weights=init_weights
        )
        print(f"Retrieved parameters: \n {params[0]}")
        param_names = [
            "A_HV",
            "A_HH",
            "B_HV",
            "B_HH",
            "C_HV",
            "C_HH",
            "D_HV",
            "D_HH",
            "a_HV",
            "a_HH",
        ] + ["S"]
        in_path = os.path.dirname(self.mask_file)
        param0_file = f"{out_prefix}_model_sim_param0_s13.csv"
        df = pd.DataFrame(params, columns=param_names)
        df.to_csv(param0_file)

        x0 = y.reshape([-1, 2])
        y0 = y_hat.reshape([-1, 2])
        density_scatter_plot(
            x0[:, 0],
            y0[:, 0],
            x_label="Measured VH",
            y_label="Predicted VH",
            x_limit=(0, max(x0[:, 0]) * 1.1),
            y_limit=(0, max(x0[:, 0]) * 1.1),
        )
        density_scatter_plot(
            x0[:, 1],
            y0[:, 1],
            x_label="Measured VV",
            y_label="Predicted VV",
            x_limit=(0, max(x0[:, 1]) * 1.1),
            y_limit=(0, max(x0[:, 1]) * 1.1),
        )
        return param0_file

    def params_calibration_sm(self, z1, W1, sm1, out_prefix, init_weights=[4, 1]):
        """
        param calibration to take SM as another input.
        Input W1 to have the same size as z1 .
        Input sm1 to have the same size as z1 .

        :param z1: (n_obs, nxn, n_time, 2_pols[hv/hh])
        :param W1: (n_obs, nxn, n_time, 1)
        :param sm1: (n_obs, nxn, n_time, 1)
        :return:
        """
        mean_hv = z1[:, :, :, 0].reshape(-1)
        mean_hh = z1[:, :, :, 1].reshape(-1)

        mean_x = np.stack([mean_hv, mean_hh]).T
        mean_W = np.stack([W1.reshape(-1), W1.reshape(-1)]).T
        mean_sm = np.stack([sm1.reshape(-1), sm1.reshape(-1)]).T
        mean_x = mean_x[~np.isnan(sm1.reshape(-1)), :]
        mean_W = mean_W[~np.isnan(sm1.reshape(-1)), :]
        mean_sm = mean_sm[~np.isnan(sm1.reshape(-1)), :]
        print(mean_x.shape)

        model_00 = Retrieval(2, 1, mean_x.shape[0])
        y = mean_x.reshape([1, -1])
        W = mean_W.reshape([1, -1])
        sm = mean_sm.reshape([1, -1]) * 1
        bounds = (
            (
                (0.0001, 0.5),
                (0.0001, 0.5),
            )
            # + 2 * ((0.0001, 0.5),)
            + 2 * ((0.0001, 2),)
            + (
                (0.0001, 0.5),
                (0.0001, 0.5),
            )
            + 2 * ((0.0001, 0.5),)
            + 2 * ((0.5, 1.5),)
            # + ((1, 1.0001), (0.0001, 25),)
            + (
                (0.0001, 25),
                (0.0001, 25),
            )
        )
        params, y_hat = model_00.model_inverse_03a_sm(
            y, W, sm, bounds, init_weights=init_weights
        )
        print(f"Retrieved parameters: \n {params[0]}")
        param_names = [
            "A_HV",
            "A_HH",
            "B_HV",
            "B_HH",
            "C_HV",
            "C_HH",
            "alpha_HV",
            "alpha_HH",
            "gamma_HV",
            "gamma_HH",
            "D_HV",
            "D_HH",
        ]
        in_path = os.path.dirname(self.mask_file)
        param0_file = f"{out_prefix}_model_sim_param0_s13.csv"
        df = pd.DataFrame(params, columns=param_names)
        df.to_csv(param0_file)
        # density_scatter_plot(
        #     y.flatten(),
        #     y_hat.flatten(),
        #     x_label="Measured Backscatter",
        #     y_label="Predicted Backscatter",
        #     x_limit=(0, 0.6),
        #     y_limit=(0, 0.6),
        #     file_name=(f"{out_prefix}_param0_y_calib.png"),
        # )
        return param0_file

    def params_calibration_sm_03b(self, z1, W1, sm1, out_prefix, init_weights=[1, 1]):
        """
        param calibration to take SM as another input. (using 03b as new model)
        Input W1 to have the same size as z1 .
        Input sm1 to have the same size as z1 .

        :param z1: (n_obs, nxn, n_time, 2_pols[hv/hh])
        :param W1: (n_obs, nxn, n_time, 1)
        :param sm1: (n_obs, nxn, n_time, 1)
        :return:
        """
        mean_x = np.moveaxis(
            z1, [0, 1, 2, 3], [1, 0, 2, 3]
        )  # reshape to [1,size_s,size_t,size_p]
        mean_W = np.concatenate([W1, W1], axis=3)
        mean_sm = np.concatenate([sm1, sm1], axis=3)
        # mean_sm = np.concatenate([mean_x[:, :, :, 1], mean_x[:, :, :, 1]], axis=3)
        print(mean_x.shape)

        model_00 = Retrieval(2, mean_x.shape[2], mean_x.shape[1])
        y = mean_x.reshape([1, -1])
        W = mean_W.reshape([1, -1])
        sm = mean_sm.reshape([1, -1])
        bounds = (
            2 * ((0.0001, 0.99),)
            + 2 * ((0.0001, 2),)
            + 2 * ((0.0001, 100),)
            + 2 * ((0.0001, 2),)
            + 2 * ((0.0001, 2),)
            + (
                (1, 1.0001),
                (0.0001, 100),
            )
            + mean_x.shape[1] * ((0.0001, 0.5),)
        )
        params, y_hat = model_00.model_inverse_03b_sm(
            y, W, sm, bounds, init_weights=init_weights
        )
        print(f"Retrieved parameters: \n {params[0]}")
        param_names = [
            "A_HV",
            "A_HH",
            "B_HV",
            "B_HH",
            "C_HV",
            "C_HH",
            "alpha_HV",
            "alpha_HH",
            "gamma_HV",
            "gamma_HH",
            "D_HV",
            "D_HH",
        ] + [f"L_{iL}" for iL in range(mean_x.shape[1])]
        param0_file = f"{out_prefix}_model_sim_param0_s13.csv"
        df = pd.DataFrame(params, columns=param_names)
        df.to_csv(param0_file)
        density_scatter_plot(
            y.flatten(),
            y_hat.flatten(),
            x_label="Measured Backscatter",
            y_label="Predicted Backscatter",
            x_limit=(0, 0.6),
            y_limit=(0, 0.6),
            # file_name=(f"{out_prefix}_param0_y_calib.png"),
        )
        return param0_file

    def params_calibration_sm_03c(self, z1, W1, sm1, out_prefix, init_weights=[1, 1]):
        """
        param calibration to take SM as another input. (using 03b as new model)
        Input W1 to have the same size as z1 .
        Input sm1 to have the same size as z1 .

        :param z1: (n_obs, nxn, n_time, 2_pols[hv/hh])
        :param W1: (n_obs, nxn, n_time, 1)
        :param sm1: (n_obs, nxn, n_time, 1)
        :return:
        """
        mean_x = np.moveaxis(
            z1, [0, 1, 2, 3], [1, 0, 2, 3]
        )  # reshape to [1,size_s,size_t,size_p]
        mean_W = np.concatenate([W1, W1], axis=3)
        mean_sm = np.concatenate([sm1, sm1], axis=3)
        # mean_sm = np.concatenate([mean_x[:, :, :, 1], mean_x[:, :, :, 1]], axis=3)
        print(mean_x.shape)

        model_00 = Retrieval(2, mean_x.shape[2], mean_x.shape[1])
        y = mean_x.reshape([1, -1])
        W = mean_W.reshape([1, -1])
        sm = mean_sm.reshape([1, -1])
        bounds = (
            2 * ((0.0001, 0.99),)
            + 2 * ((0.0001, 2),)
            + 2 * ((0.0001, 100),)
            + 2 * ((0.0001, 2),)
            + 2 * ((0.0001, 2),)
            + (
                (1, 1.0001),
                (0.0001, 100),
            )
            + mean_x.shape[1] * ((0.0001, 0.5),)
            # + ((1, 1.0001),) + (mean_x.shape[1]-1) * ((0.0001, 0.5),)
            + mean_x.shape[1] * ((0.0001, 0.5),)
        )
        params, y_hat = model_00.model_inverse_03c_sm(
            y, W, sm, bounds, init_weights=init_weights
        )
        print(f"Retrieved parameters: \n {params[0]}")
        param_names = (
            [
                "A_HV",
                "A_HH",
                "B_HV",
                "B_HH",
                "C_HV",
                "C_HH",
                "alpha_HV",
                "alpha_HH",
                "gamma_HV",
                "gamma_HH",
                "D_HV",
                "D_HH",
            ]
            + [f"L_{iL}" for iL in range(mean_x.shape[1])]
            + [f"E_{iL}" for iL in range(mean_x.shape[1])]
        )
        param0_file = f"{out_prefix}_model_sim_param0_s13.csv"
        df = pd.DataFrame(params, columns=param_names)
        df.to_csv(param0_file)
        density_scatter_plot(
            y.flatten(),
            y_hat.flatten(),
            x_label="Measured Backscatter",
            y_label="Predicted Backscatter",
            x_limit=(0, 0.6),
            y_limit=(0, 0.6),
            # file_name=(f"{out_prefix}_param0_y_calib.png"),
        )
        return param0_file

    def params_calibration_sm_03d(self, z1, W1, sm1, out_prefix, init_weights=[1, 1]):
        """
        param calibration to take SM as another input. (using 03b as new model)
        Input W1 to have the same size as z1 .
        Input sm1 to have the same size as z1 .

        :param z1: (n_obs, nxn, n_time, 2_pols[hv/hh])
        :param W1: (n_obs, nxn, n_time, 1)
        :param sm1: (n_obs, nxn, n_time, 1)
        :return:
        """
        mean_x = np.moveaxis(
            z1, [0, 1, 2, 3], [1, 0, 2, 3]
        )  # reshape to [1,size_s,size_t,size_p]
        mean_W = np.concatenate([W1, W1], axis=3)
        mean_sm = np.concatenate([sm1, sm1], axis=3)
        # mean_sm = np.concatenate([mean_x[:, :, :, 1], mean_x[:, :, :, 1]], axis=3)
        print(mean_x.shape)

        model_00 = Retrieval(2, mean_x.shape[2], mean_x.shape[1])
        y = mean_x.reshape([1, -1])
        W = mean_W.reshape([1, -1])
        sm = mean_sm.reshape([1, -1])
        bounds = (
            2 * ((0.0001, 0.99),)
            + 2 * ((0.0001, 2),)
            + 2 * ((0.0001, 100),)
            + 2 * ((0.0001, 2),)
            + 2 * ((0.0001, 2),)
            + (
                (1, 1.0001),
                (0.0001, 100),
            )
            + mean_x.shape[1] * ((0.0001, 0.5),)
            # + ((1, 1.0001),) + (mean_x.shape[1]-1) * ((0.0001, 0.5),)
            + mean_x.shape[1] * ((0.0001, 0.5),)
            + mean_x.shape[1] * ((0.0001, 0.5),)
        )
        params, y_hat = model_00.model_inverse_03d_sm(
            y, W, sm, bounds, init_weights=init_weights
        )
        print(f"Retrieved parameters: \n {params[0]}")
        param_names = (
            [
                "A_HV",
                "A_HH",
                "B_HV",
                "B_HH",
                "C_HV",
                "C_HH",
                "alpha_HV",
                "alpha_HH",
                "gamma_HV",
                "gamma_HH",
                "D_HV",
                "D_HH",
            ]
            + [f"L_{iL}" for iL in range(mean_x.shape[1])]
            + [f"E_{iL}" for iL in range(mean_x.shape[1])]
            + [f"G_{iL}" for iL in range(mean_x.shape[1])]
        )
        param0_file = f"{out_prefix}_model_sim_param0_s13.csv"
        df = pd.DataFrame(params, columns=param_names)
        df.to_csv(param0_file)
        density_scatter_plot(
            y.flatten(),
            y_hat.flatten(),
            x_label="Measured Backscatter",
            y_label="Predicted Backscatter",
            x_limit=(0, 0.6),
            y_limit=(0, 0.6),
            # file_name=(f"{out_prefix}_param0_y_calib.png"),
        )
        return param0_file

    def params_calibration_sm_03e(
        self, z1, W1, sm1, out_prefix, init_weights=[1, 1], keys=["HV", "HH", "VV"]
    ):
        """
        param calibration to take SM as another input. (using 03b as new model)
        Input W1 to have the same size as z1 .
        Input sm1 to have the same size as z1 .

        :param z1: (n_obs, nxn, n_time, n_pols[hv/hh/vv])
        :param W1: (n_obs, nxn, n_time, 1)
        :param sm1: (n_obs, nxn, n_time, 1)
        :return:
        """
        mean_x = np.moveaxis(
            z1, [0, 1, 2, 3], [1, 0, 2, 3]
        )  # reshape to [1,size_s,size_t,size_p]
        print(mean_x.shape)
        n_pols = mean_x.shape[3]

        mean_W = np.concatenate(n_pols * [W1], axis=3)
        mean_sm = np.concatenate(n_pols * [sm1], axis=3)

        model_00 = Retrieval(n_pols, mean_x.shape[2], mean_x.shape[1])
        y = mean_x.reshape([1, -1])
        W = mean_W.reshape([1, -1])
        sm = mean_sm.reshape([1, -1])
        bounds = (
            n_pols * ((0.0001, 0.99),)
            + n_pols * ((0.0001, 0.99),)
            + n_pols * ((0.0001, 0.99),)
            + n_pols * ((0.0001, 1.99),)
            + n_pols * ((0.0001, 1.99),)
            + n_pols * ((0.0001, 19.9999),)
            + mean_x.shape[1] * ((0.0001, 0.9999),)
        )
        params, y_hat = model_00.model_inverse_03e_sm(
            y, W, sm, bounds, init_weights=init_weights
        )
        print(f"Retrieved parameters: \n {params[0]}")

        param_names = (
            [f"A_{i}" for i in keys]
            + [f"B_{i}" for i in keys]
            + [f"C_{i}" for i in keys]
            + [f"alpha_{i}" for i in keys]
            + [f"gamma_{i}" for i in keys]
            + [f"D_{i}" for i in keys]
            + [f"L_{iL}" for iL in range(mean_x.shape[1])]
        )
        param0_file = f"{out_prefix}_model_sim_param0_s13.csv"
        df = pd.DataFrame(params, columns=param_names)
        df.to_csv(param0_file)
        density_scatter_plot(
            y.flatten(),
            y_hat.flatten(),
            x_label="Measured Backscatter",
            y_label="Predicted Backscatter",
            x_limit=(0, 0.6),
            y_limit=(0, 0.6),
            # file_name=(f"{out_prefix}_param0_y_calib.png"),
        )
        return param0_file

    def params_calibration_sm_03g(
        self, z1, W1, sm1, ks1, out_prefix, init_weights=[1, 1], keys=["HV", "HH", "VV"]
    ):
        """
        param calibration assuming known ks. (from bare interp)
        Input W1 to have the same size as z1 .
        Input sm1 to have the same size as z1 .

        :param z1: (n_obs, nxn, n_time, 2_pols[hv/hh])
        :param W1: (n_obs, nxn, n_time, 1)
        :param sm1: (n_obs, nxn, n_time, 1)
        :param ks1: (n_obs, nxn, n_time, 1)
        :return:
        """
        mean_x = np.moveaxis(
            z1, [0, 1, 2, 3], [1, 0, 2, 3]
        )  # reshape to [1,size_s,size_t,size_p]
        # print(mean_x.shape)
        n_pols = mean_x.shape[3]

        mean_W = np.concatenate(n_pols * [W1], axis=3)
        mean_sm = np.concatenate(n_pols * [sm1], axis=3)
        mean_ks = np.concatenate(n_pols * [ks1], axis=3)
        # print(mean_W.shape)

        model_00 = Retrieval(n_pols, mean_x.shape[2], mean_x.shape[1])
        y = mean_x.reshape([1, -1])
        W = mean_W.reshape([1, -1])
        sm = mean_sm.reshape([1, -1])
        ks = mean_ks.reshape([1, -1])
        bounds = (
            n_pols
            * [
                (0.0001, 0.99),
            ]
            + n_pols
            * [
                (0.0001, 0.99),
            ]
            + n_pols
            * [
                (0.0001, 0.99),
            ]
            + n_pols
            * [
                (0.0001, 1.99),
            ]
            + n_pols
            * [
                (0.0001, 1.99),
            ]
            + n_pols
            * [
                (0.0001, 19.9999),
            ]
        )
        params, y_hat = model_00.model_inverse_03g_sm(
            y, W, sm, ks, bounds, init_weights=init_weights
        )
        print(f"Retrieved parameters: \n {params[0]}")
        param_names = (
            [f"A_{i}" for i in keys]
            + [f"B_{i}" for i in keys]
            + [f"C_{i}" for i in keys]
            + [f"alpha_{i}" for i in keys]
            + [f"gamma_{i}" for i in keys]
            + [f"D_{i}" for i in keys]
        )
        param0_file = f"{out_prefix}_model_sim_param0_s13.csv"
        df = pd.DataFrame(params, columns=param_names)
        df.to_csv(param0_file)
        density_scatter_plot(
            y.flatten(),
            y_hat.flatten(),
            x_label="Measured Backscatter",
            y_label="Predicted Backscatter",
            x_limit=(0, 0.6),
            y_limit=(0, 0.6),
            # file_name=(f"{out_prefix}_param0_y_calib.png"),
        )
        return param0_file

    def inversion_recursive_ws(self, z1, mask_ws1, param0_file=None):
        """

        Args:
            z1: input signal (n_obs, nxn, n_time, 2_pols)
            mask_ws1: raster mask file used to define valid pixels (# of valid: n_obs)
            param0_file: file storing the initial parameters

        Returns: W (agb), S (agb) - with the dimension (obs, n_time)

        """

        kn = 10
        if param0_file is None:
            in_path = os.path.dirname(self.mask_file)
            param0_file = "{}/model_sim_param0_s13.csv".format(in_path)
        df0 = pd.read_csv(param0_file)
        param0 = df0.iloc[0, 1:].values

        n_obs = z1.shape[0]
        nxn = z1.shape[1]
        m_t = z1.shape[2]
        n_1 = (np.sqrt(nxn) - 1) // 2
        in0 = gdal.Open(mask_ws1, gdal.GA_ReadOnly)
        W_mean = []
        S_mean = []
        for i in range(m_t):
            print("Scene {} of {}".format(i + 1, m_t))
            z = z1[:, :, i, :]
            mean_hv = z[:, :, 0]
            param0w = (100 * mean_hv) ** 2  # dimension: (n_obs, nxn)
            param0c = np.tile(param0, (n_obs, 1))
            param0c = np.concatenate((param0c, param0w), axis=1)

            for k in range(kn):
                bound0c = [
                    [(w0 - 0.0001, w0 + 0.0001) for w0 in param0c[iw, : 13 + nxn // 2]]
                    + [
                        (w0 * 0.2, w0 * 1.8)
                        for w0 in param0c[iw, 13 + nxn // 2 : 13 + nxn // 2 + 1]
                    ]
                    + [
                        (w0 - 0.0001, w0 + 0.0001)
                        for w0 in param0c[iw, 13 + nxn // 2 + 1 :]
                    ]
                    for iw in range(param0c.shape[0])
                ]
                model_00 = Retrieval(2, 1, nxn)
                params, z_hat = model_00.sar_model_inverse_03(param0c, z, bound0c)
                w0 = params[:, -nxn // 2]
                w0sum = np.sqrt(np.mean((w0 - param0c[:, -nxn // 2]) ** 2))

                array0 = in0.GetRasterBand(1).ReadAsArray()
                array0[array0 > self.valid_min] = w0
                if not os.path.exists("tmp"):
                    os.mkdir("tmp")
                try:
                    os.remove(
                        "tmp/tmp_agb_{}_{}_{}.tif".format(
                            os.path.basename(self.out_name), k, i
                        )
                    )
                except OSError:
                    pass
                driver = gdal.GetDriverByName("GTiff")
                ds = driver.CreateCopy(
                    "tmp/tmp_agb_{}_{}_{}.tif".format(
                        os.path.basename(self.out_name), k, i
                    ),
                    in0,
                    0,
                    ["COMPRESS=LZW", "PREDICTOR=2"],
                )
                ds.GetRasterBand(1).WriteArray(array0)
                ds.FlushCache()  # Write to disk.
                ds = None

                df_agb = pp.array_reshape_rolling(
                    [
                        "tmp/tmp_agb_{}_{}_{}.tif".format(
                            os.path.basename(self.out_name), k, i
                        )
                    ],
                    mask_ws1,
                    name="tmp/tmp_agb_nxn_{}_{}_{}".format(
                        os.path.basename(self.out_name), k, i
                    ),
                    m=n_1,
                    n=n_1,
                    valid_min=self.valid_min,
                )
                param0w = df_agb.iloc[:, 1:].values
                param0w[np.isnan(param0w)] = (100 * mean_hv[np.isnan(param0w)]) ** 2

                param0c = np.concatenate((param0c[:, :13], param0w), axis=1)

                """
                Update Parameters...
                """
                bound0c = [
                    [(w0 - 0.0001, w0 + 0.0001) for w0 in param0c[iw, :6]]
                    # [(w0 * 0.8, w0 * 1.2) for w0 in param0c[iw, :6]]
                    + [(w0 - 0.0001, w0 + 0.0001) for w0 in param0c[iw, 6:12]]
                    + [(w0 * 0.6, w0 * 1.4) for w0 in param0c[iw, 12:13]]
                    + [(w0 - 0.0001, w0 + 0.0001) for w0 in param0c[iw, 13:]]
                    for iw in range(param0c.shape[0])
                ]
                model_00 = Retrieval(2, 1, nxn)
                param0c, y_hat = model_00.sar_model_inverse_03(param0c, z, bound0c)
                s0 = params[:, 12]
                # density_scatter_plot(
                #     z.flatten(),
                #     y_hat.flatten(),
                #     x_label="Measured Backscatter",
                #     y_label="Predicted Backscatter",
                #     x_limit=(0, 0.6),
                #     y_limit=(0, 0.6),
                #     file_name=(self.out_name + "_ws1_y_s{}_k{}.png").format(i, k),
                # )

                # print(f'k = {k}, w_res = {w0sum}')
                if w0sum < 1:
                    # print(f"early stop of w0 at k = {k}")
                    break

            W_mean.append(w0)
            S_mean.append(s0)

        W_mean = np.array(W_mean).T
        S_mean = np.array(S_mean).T
        print("Dimension of W: {}".format(W_mean.shape))  # should be (n_obs, m_t)

        in0 = None

        return W_mean, S_mean

    def inversion_recursive_ws_04a_noint(self, z1, mask_ws1, param0_file=None):
        """

        Args:
            z1: input signal (n_obs, nxn, n_time, 2_pols)
            mask_ws1: raster mask file used to define valid pixels (# of valid: n_obs)
            param0_file: file storing the initial parameters

        Returns: W (agb), S (agb) - with the dimension (obs, n_time)

        """

        kn = 10
        if param0_file is None:
            in_path = os.path.dirname(self.mask_file)
            param0_file = "{}/model_sim_param0_s13.csv".format(in_path)
        df0 = pd.read_csv(param0_file)
        param0 = df0.iloc[0, 1:].values

        n_obs = z1.shape[0]
        nxn = z1.shape[1]
        m_t = z1.shape[2]
        n_1 = (np.sqrt(nxn) - 1) // 2
        in0 = gdal.Open(mask_ws1, gdal.GA_ReadOnly)
        W_mean = []
        S_mean = []
        for i in range(m_t):
            print("Scene {} of {}".format(i + 1, m_t))
            z = z1[:, :, i, :]
            mean_hv = z[:, :, 0]
            param0w = (100 * mean_hv) ** 2  # dimension: (n_obs, nxn)
            param0c = np.tile(param0, (n_obs, 1))
            param0c = np.concatenate((param0c, param0w), axis=1)

            n_param = 11
            for k in range(kn):
                bound0c = [
                    [
                        (w0 - 0.00001, w0 + 0.00001)
                        for w0 in param0c[iw, : n_param + nxn // 2]
                    ]
                    + [
                        (w0 * 0.2, w0 * 1.8)
                        for w0 in param0c[
                            iw, n_param + nxn // 2 : n_param + nxn // 2 + 1
                        ]
                    ]
                    + [
                        (w0 - 0.00001, w0 + 0.00001)
                        for w0 in param0c[iw, n_param + nxn // 2 + 1 :]
                    ]
                    for iw in range(param0c.shape[0])
                ]
                model_00 = Retrieval(2, 1, nxn)
                params, z_hat = model_00.sar_model_inverse_04a_noint(
                    param0c, z, bound0c
                )
                w0 = params[:, -nxn // 2]
                w0sum = np.sqrt(np.mean((w0 - param0c[:, -nxn // 2]) ** 2))

                array0 = in0.GetRasterBand(1).ReadAsArray()
                array0[array0 > self.valid_min] = w0
                if not os.path.exists("tmp"):
                    os.mkdir("tmp")
                try:
                    os.remove(
                        "tmp/tmp_agb_{}_{}_{}.tif".format(
                            os.path.basename(self.out_name), k, i
                        )
                    )
                except OSError:
                    pass
                driver = gdal.GetDriverByName("GTiff")
                ds = driver.CreateCopy(
                    "tmp/tmp_agb_{}_{}_{}.tif".format(
                        os.path.basename(self.out_name), k, i
                    ),
                    in0,
                    0,
                    ["COMPRESS=LZW", "PREDICTOR=2"],
                )
                ds.GetRasterBand(1).WriteArray(array0)
                ds.FlushCache()  # Write to disk.
                ds = None

                df_agb = pp.array_reshape_rolling(
                    [
                        "tmp/tmp_agb_{}_{}_{}.tif".format(
                            os.path.basename(self.out_name), k, i
                        )
                    ],
                    mask_ws1,
                    name="tmp/tmp_agb_nxn_{}_{}_{}".format(
                        os.path.basename(self.out_name), k, i
                    ),
                    m=n_1,
                    n=n_1,
                    valid_min=self.valid_min,
                )
                param0w = df_agb.iloc[:, 1:].values
                param0w[np.isnan(param0w)] = (100 * mean_hv[np.isnan(param0w)]) ** 2

                param0c = np.concatenate((param0c[:, :n_param], param0w), axis=1)

                """
                Update Parameters...
                """
                bound0c = [
                    [(w0 - 0.00001, w0 + 0.00001) for w0 in param0c[iw, : n_param - 1]]
                    + [
                        (w0 * 0.6, w0 * 1.4)
                        for w0 in param0c[iw, n_param - 1 : n_param]
                    ]
                    + [(w0 - 0.00001, w0 + 0.00001) for w0 in param0c[iw, n_param:]]
                    for iw in range(param0c.shape[0])
                ]
                model_00 = Retrieval(2, 1, nxn)
                param0c, y_hat = model_00.sar_model_inverse_04a_noint(
                    param0c, z, bound0c
                )
                s0 = params[:, n_param - 1]
                # density_scatter_plot(
                #     z.flatten(),
                #     y_hat.flatten(),
                #     x_label="Measured Backscatter",
                #     y_label="Predicted Backscatter",
                #     x_limit=(0, 0.6),
                #     y_limit=(0, 0.6),
                #     file_name=(self.out_name + "_ws1_y_s{}_k{}.png").format(i, k),
                # )

                # print(f'k = {k}, w_res = {w0sum}')
                if w0sum < 1:
                    # print(f"early stop of w0 at k = {k}")
                    break

            W_mean.append(w0)
            S_mean.append(s0)

        W_mean = np.array(W_mean).T
        S_mean = np.array(S_mean).T
        print("Dimension of W: {}".format(W_mean.shape))  # should be (n_obs, m_t)

        in0 = None

        return W_mean, S_mean

    def inversion_recursive_ws_single(self, z1, mask_ws1, param0_file=None):
        """

        Args:
            z1: input signal (n_obs, nxn, n_time, 1_pols)
            mask_ws1: raster mask file used to define valid pixels (# of valid: n_obs)
            param0_file: file storing the initial parameters

        Returns: W (agb), S (agb) - with the dimension (obs, n_time)

        """

        kn = 1
        if param0_file is None:
            in_path = os.path.dirname(self.mask_file)
            param0_file = "{}/model_sim_param0_s13.csv".format(in_path)
        df0 = pd.read_csv(param0_file)
        param0 = df0.iloc[0, 1:].values

        n_obs = z1.shape[0]
        nxn = z1.shape[1]
        m_t = z1.shape[2]
        n_1 = (np.sqrt(nxn) - 1) // 2
        in0 = gdal.Open(mask_ws1, gdal.GA_ReadOnly)
        W_mean = []
        for i in range(m_t):
            print("Scene {} of {}".format(i + 1, m_t))
            z = z1[:, :, i, :1]
            mean_hv = z[:, :, 0]
            param0w = (100 * mean_hv) ** 2  # dimension: (n_obs, nxn)
            param0c = np.tile(param0, (n_obs, 1))
            param0c = np.concatenate((param0c, param0w), axis=1)

            for k in range(kn):
                bound0c = [
                    [(w0 - 0.0001, w0 + 0.0001) for w0 in param0c[iw, : 7 + nxn // 2]]
                    + [
                        (w0 * 0.2, w0 * 1.8)
                        for w0 in param0c[iw, 7 + nxn // 2 : 7 + nxn // 2 + 1]
                    ]
                    + [
                        (w0 - 0.0001, w0 + 0.0001)
                        for w0 in param0c[iw, 7 + nxn // 2 + 1 :]
                    ]
                    for iw in range(param0c.shape[0])
                ]
                model_00 = Retrieval(1, 1, nxn)
                params, z_hat = model_00.sar_model_inverse_03(param0c, z, bound0c)
                w0 = params[:, -nxn // 2]

            W_mean.append(w0)

        W_mean = np.array(W_mean).T
        print("Dimension of W: {}".format(W_mean.shape))  # should be (n_obs, m_t)

        in0 = None

        return W_mean

    def inversion_recursive_ws_sm(self, z1, mask_ws1, sm0=0.2, param0_file=None):
        """

        Args:
            sm0: initial S0
            z1: input signal (n_obs, nxn, n_time, 2_pols)
            mask_ws1: raster mask file used to define valid pixels (# of valid: n_obs)
            param0_file: file storing the initial parameters

        Returns: W (agb), S (agb) - with the dimension (obs, n_time)

        """

        kn = 1
        if param0_file is None:
            in_path = os.path.dirname(self.mask_file)
            param0_file = "{}/model_sim_param0_s13.csv".format(in_path)
        df0 = pd.read_csv(param0_file)
        param0 = df0.iloc[0, 1:].values

        n_obs = z1.shape[0]
        nxn = z1.shape[1]
        m_t = z1.shape[2]
        n_1 = (np.sqrt(nxn) - 1) // 2
        in0 = gdal.Open(mask_ws1, gdal.GA_ReadOnly)
        # print(in0.GetGeoTransform())
        W_mean = []
        S_mean = []
        for i in range(m_t):
            print("Scene {} of {}".format(i + 1, m_t))
            z = z1[:, :, i, :]
            mean_hv = z[:, :, 0]
            param0w = (100 * mean_hv) ** 2  # dimension: (n_obs, nxn)
            param0s = np.append(param0, sm0)
            param0c = np.tile(param0s, (n_obs, 1))
            param0c = np.concatenate((param0c, param0w), axis=1)

            for k in range(kn):
                bound0c = [
                    [(w0 * 0.9999, w0 * 1.0001) for w0 in param0c[iw, :6]]
                    + [(w0 * 0.9999, w0 * 1.0001) for w0 in param0c[iw, 6:12]]
                    + [(w0 * 0.5, w0 * 2) for w0 in param0c[iw, 12:13]]
                    + [
                        (w0 * 0.9999, w0 * 1.0001)
                        for w0 in param0c[iw, 13 : 13 + nxn // 2]
                    ]
                    + [
                        (w0 * 0.2, w0 * 5)
                        for w0 in param0c[iw, 13 + nxn // 2 : 13 + nxn // 2 + 1]
                    ]
                    + [
                        (w0 * 0.9999, w0 * 1.0001)
                        for w0 in param0c[iw, 13 + nxn // 2 + 1 :]
                    ]
                    for iw in range(param0c.shape[0])
                ]
                model_00 = Retrieval(2, 1, nxn)
                param0c, _ = model_00.sar_model_inverse_03(param0c, z, bound0c)
                w0 = param0c[:, -nxn // 2]
                s0 = param0c[:, 12]

            W_mean.append(w0)
            S_mean.append(s0)

        W_mean = np.array(W_mean).T
        S_mean = np.array(S_mean).T
        print("Dimension of W: {}".format(W_mean.shape))  # should be (n_obs, m_t)

        in0 = None

        return W_mean, S_mean

    def inversion_recursive_ws_sm_wkagb(
        self, z1, z2, mask_ws1, sm0=0.2, param0_file=None, init_weights=[4, 1]
    ):
        """

        Args:
            sm0: initial S0
            z1: input signal SAR (n_obs, nxn, n_time, 2_pols)
            z2: input signal VWC (n_obs, nxn, n_time, 1)
            mask_ws1: raster mask file used to define valid pixels (# of valid: n_obs)
            param0_file: file storing the initial parameters

        Returns: W (agb), S (agb) - with the dimension (obs, n_time)

        """

        kn = 1
        if param0_file is None:
            in_path = os.path.dirname(self.mask_file)
            param0_file = "{}/model_sim_param0_s13.csv".format(in_path)
        df0 = pd.read_csv(param0_file)
        param0 = df0.iloc[0, 1:].values

        n_obs = z1.shape[0]
        nxn = z1.shape[1]
        m_t = z1.shape[2]
        n_1 = (np.sqrt(nxn) - 1) // 2
        in0 = gdal.Open(mask_ws1, gdal.GA_ReadOnly)
        # print(in0.GetGeoTransform())
        S_mean = []
        for i in range(m_t):
            print("Scene {} of {}".format(i + 1, m_t))
            z = z1[:, :, i, :]
            mean_hv = z[:, :, 0]
            W_0 = (100 * mean_hv) ** 2  # dimension: (n_obs, nxn)
            W_i = z2[:, :, i, 0]
            # print(np.mean(W_i))
            # print(np.sum(~np.isnan(W_i.ravel())))
            param0w = W_i  # AGB_i
            param0s = np.append(param0, sm0)
            param0c = np.tile(param0s, (n_obs, 1))
            param0c = np.concatenate((param0c, param0w), axis=1)

            for k in range(kn):
                bound0c = [
                    [(w0 * 0.9999, w0 * 1.0001) for w0 in param0c[iw, : 13 + nxn // 2]]
                    + [
                        (w0 * 0.9999, w0 * 1.0001)
                        for w0 in param0c[iw, 13 + nxn // 2 : 13 + nxn // 2 + 1]
                    ]
                    + [
                        (w0 * 0.9999, w0 * 1.0001)
                        for w0 in param0c[iw, 13 + nxn // 2 + 1 :]
                    ]
                    for iw in range(param0c.shape[0])
                ]
                model_00 = Retrieval(2, 1, nxn)
                params, z_hat = model_00.sar_model_inverse_03(
                    param0c, z, bound0c, init_weights=init_weights
                )
                w0 = params[:, -nxn // 2]

                array0 = in0.GetRasterBand(1).ReadAsArray()
                # print(array0[array0 > self.valid_min].shape)
                array0[array0 > self.valid_min] = w0
                if not os.path.exists("tmp"):
                    os.mkdir("tmp")
                try:
                    os.remove(
                        "tmp/tmp_agb_{}_{}_{}.tif".format(
                            os.path.basename(self.out_name), k, i
                        )
                    )
                except OSError:
                    pass
                driver = gdal.GetDriverByName("GTiff")
                ds = driver.CreateCopy(
                    "tmp/tmp_agb_{}_{}_{}.tif".format(
                        os.path.basename(self.out_name), k, i
                    ),
                    in0,
                    0,
                    ["COMPRESS=LZW", "PREDICTOR=2"],
                )
                ds.GetRasterBand(1).WriteArray(array0)
                ds.FlushCache()  # Write to disk.
                ds = None

                df_agb = pp.array_reshape_rolling(
                    [
                        "tmp/tmp_agb_{}_{}_{}.tif".format(
                            os.path.basename(self.out_name), k, i
                        )
                    ],
                    mask_ws1,
                    name="tmp/tmp_agb_nxn_{}_{}_{}".format(
                        os.path.basename(self.out_name), k, i
                    ),
                    m=n_1,
                    n=n_1,
                    valid_min=self.valid_min,
                )
                param0w = df_agb.iloc[:, 1:].values
                # param0w[np.isnan(param0w)] = W_i[np.isnan(param0w)]
                param0w[np.isnan(param0w)] = W_0[np.isnan(param0w)]

                param0c = np.concatenate((param0c[:, :13], param0w), axis=1)

                """
                Update Parameters...
                """
                bound0c = [
                    [(w0 * 0.9999, w0 * 1.0001) for w0 in param0c[iw, :6]]
                    # [(w0 * 0.8, w0 * 1.2) for w0 in param0c[iw, :6]]
                    + [(w0 * 0.9999, w0 * 1.0001) for w0 in param0c[iw, 6:12]]
                    + [(w0 * 0.2, w0 * 5) for w0 in param0c[iw, 12:13]]
                    + [(w0 * 0.9999, w0 * 1.0001) for w0 in param0c[iw, 13:]]
                    for iw in range(param0c.shape[0])
                ]
                model_00 = Retrieval(2, 1, nxn)
                param0c, y_hat = model_00.sar_model_inverse_03(
                    param0c, z, bound0c, init_weights=init_weights
                )
                s0 = param0c[:, 12]
                # density_scatter_plot(
                #     z.flatten(),
                #     y_hat.flatten(),
                #     x_label="Measured Backscatter",
                #     y_label="Predicted Backscatter",
                #     x_limit=(0, 0.6),
                #     y_limit=(0, 0.6),
                #     file_name=(self.out_name + "_ws1_y_s{}_k{}.png").format(i, k),
                # )

                s0sum = np.sqrt(np.mean((s0 - params[:, 12]) ** 2))
                print(f"k = {k}, w_res = {s0sum} ")
                if s0sum < 0.05:
                    print(f"early stop of w0 at k = {k}")
                    break

            S_mean.append(s0)

        S_mean = np.array(S_mean).T
        print("Dimension of W: {}".format(S_mean.shape))  # should be (n_obs, m_t)

        in0 = None

        return S_mean

    def inversion_nobs_03c(self, z1, param0_file=None, init_weights=[1, 1]):
        """

        Args:
            z1: input signal SAR (n_obs, 1, n_time, 2_pols)
            param0_file: file storing the initial parameters

        Returns: W (agb), S (agb) - with the dimension (obs, n_time)

        """

        kn = 10
        if param0_file is None:
            in_path = os.path.dirname(self.mask_file)
            param0_file = "{}/model_sim_param0_s13.csv".format(in_path)
        df0 = pd.read_csv(param0_file)
        param0 = df0.iloc[0, 1:].values

        n_obs = z1.shape[0]
        m_t = z1.shape[2]

        W_mean = []
        S_mean = []
        for i in range(m_t):
            print("Scene {} of {}".format(i + 1, m_t))
            z = z1[:, :, i, :]
            mean_hv = z[:, :, 0]
            param0w = (100 * mean_hv) ** 2  # dimension: (n_obs, 1)
            param0c = np.tile(param0, (n_obs, 1))
            param0c = np.concatenate((param0c, param0w), axis=1)

            for k in range(kn):
                bound0c = [
                    [(w0 * 0.9999, w0 * 1.0001) for w0 in param0c[iw, :13]]
                    + [(w0 * 0.2, w0 * 1.8) for w0 in param0c[iw, 13:14]]
                    for iw in range(param0c.shape[0])
                ]
                model_00 = Retrieval(2, 1, 1)
                params, z_hat = model_00.sar_model_inverse_03(param0c, z, bound0c)
                w0 = params[:, -1]
                w0sum = np.sqrt(np.mean((w0 - param0c[:, -1]) ** 2))

                param0w = w0[:, None]
                param0w[np.isnan(param0w)] = (100 * mean_hv[np.isnan(param0w)]) ** 2

                param0c = np.concatenate((param0c[:, :13], param0w), axis=1)

                """
                Update Parameters...
                """
                bound0c = [
                    [(w0 * 0.9999, w0 * 1.0001) for w0 in param0c[iw, :6]]
                    # [(w0 * 0.8, w0 * 1.2) for w0 in param0c[iw, :6]]
                    + [(w0 * 0.9999, w0 * 1.0001) for w0 in param0c[iw, 6:12]]
                    + [(w0 * 0.6, w0 * 1.4) for w0 in param0c[iw, 12:13]]
                    + [(w0 * 0.9999, w0 * 1.0001) for w0 in param0c[iw, 13:]]
                    for iw in range(param0c.shape[0])
                ]
                model_00 = Retrieval(2, 1, 1)
                param0c, y_hat = model_00.sar_model_inverse_03(param0c, z, bound0c)
                s0 = params[:, 12]
                if w0sum < 1:
                    print(f"early stop of w0 at k = {k}")
                    break
            W_mean.append(w0)
            S_mean.append(s0)
        W_mean = np.array(W_mean).T
        S_mean = np.array(S_mean).T

        return W_mean, S_mean

    def inversion_nobs_sm_03e_wkks(
        self, z1, z2, sm0, param0_file=None, init_weights=[1, 1]
    ):
        """

        Args:
            sm0: initial S0
            z1: input signal HH/VV (n_obs, nxn, n_time, n_pols)
            z2: input signal ks0 from bare interp (n_obs, nxn, 1, 1)
            sm0: input signal sm0 from bare interp (n_obs, nxn, n_time, 1)
            param0_file: file storing the initial parameters

        Returns: W (agb), S (agb) - with the dimension (obs, n_time)

        """

        n_obs = z1.shape[0]
        nxn = z1.shape[1]
        m_t = z1.shape[2]
        n_pols = z1.shape[3]

        if param0_file is None:
            in_path = os.path.dirname(self.mask_file)
            param0_file = "{}/model_sim_param0_s13.csv".format(in_path)
        df0 = pd.read_csv(param0_file)
        param0 = df0.iloc[0, 1 : n_pols * 6 + 1].values
        param_L = z2[:, 0, :, 0]  # ks (n_obs, 1)

        param0n = np.tile(param0, (n_obs, 1))
        param0LE = np.concatenate([param0n, param_L], axis=1)
        # param0s = np.tile(np.full((m_t,), sm0), (n_obs, 1))
        param0s = sm0[:, 0, :, 0]  # initial SM
        # param0s = z1[:, 0, :, 1] * 4   # HH as initial SM
        param0Ls = np.concatenate([param0LE, param0s], axis=1)

        mean_hv = z1[:, 0, :, 0]
        param0w = (100 * mean_hv) ** 2  # dimension: (n_obs, nxn)
        # param0w = z2[:, 0, :, 0]  # AGB (n_obs, m_t)
        # param0w = np.tile(np.nanmean(param0w, axis=1), [1, m_t])
        param0c = np.concatenate((param0Ls, param0w), axis=1)

        bound0c = [
            [
                (w0 * 0.9999, w0 * 1.0001) for w0 in param0c[iw, : n_pols * 6]
            ]  # A,B,C,a,c,D
            + [
                (w0 * 0.9, w0 * 1.1)
                for w0 in param0c[iw, n_pols * 6 : n_pols * 6 + nxn]
            ]  # L
            + [
                (w0 * 0.6, w0 * 1.6)
                for w0 in param0c[iw, n_pols * 6 + nxn : n_pols * 6 + nxn + m_t]
            ]  # SM
            + [
                (w0 * 0.2, w0 * 1.8) for w0 in param0c[iw, n_pols * 6 + nxn + m_t :]
            ]  # W
            for iw in range(param0c.shape[0])
        ]
        model_00 = Retrieval(n_pols, m_t, 1)
        param0c, y_hat = model_00.sar_model_inverse_03_sme(
            param0c, z1, bound0c, init_weights=init_weights
        )
        S_mean = param0c[:, n_pols * 6 + nxn : n_pols * 6 + nxn + m_t]
        W_mean = param0c[:, n_pols * 6 + nxn + m_t :]

        # S_lst = []
        # W_lst = []
        # for i in range(m_t):
        #     model_00 = Retrieval(n_pols, 1, 1)
        #     z1t = z1[:, :, i:i + 1, :]
        #     param0ct = np.concatenate([param0c[:, :n_pols*6 + nxn],
        #                                param0c[:, n_pols*6 + nxn + i:n_pols*6 + nxn + i + 1],
        #                                param0c[:, n_pols*6 + nxn + m_t + i:n_pols*6 + nxn + m_t + i + 1]
        #                                ], axis=1)
        #     bound0ct = [
        #         [(w0 * 0.9999, w0 * 1.0001) for w0 in param0ct[iw, :n_pols*6]]  # A,B,C,a,c,D
        #         + [(w0 * 0.9999, w0 * 1.0001) for w0 in param0ct[iw, n_pols*6:n_pols*6 + nxn]]  # L
        #         + [(w0 * 0.6, w0 * 1.6) for w0 in param0ct[iw, n_pols*6 + nxn:n_pols*6 + nxn + 1]]  # SM
        #         + [(w0 * 0.2, w0 * 1.8) for w0 in param0ct[iw, n_pols*6 + nxn + 1:]]  # W
        #         for iw in range(param0ct.shape[0])
        #     ]
        #     param0ct, y_hat = model_00.sar_model_inverse_03_sme(param0ct, z1t, bound0ct, init_weights=init_weights)
        #     S_t = param0ct[:, n_pols*6+nxn:n_pols*6+nxn+1]
        #     S_lst.append(S_t)
        #     W_lst.append(param0ct[:, n_pols*6+nxn+1:])
        # S_mean = np.concatenate(S_lst, axis=1)
        # W_mean = np.concatenate(W_lst, axis=1)

        print(f"Dimension of S: {S_mean.shape}")

        return S_mean, W_mean

    def inversion_nobs_sm_03h2_wkks(
        self, z1, z2, sm0, param0_file=None, init_weights=[1, 1]
    ):
        """
        W is from HV, and has no change in this retrieval
        Args:
            sm0: initial S0
            z1: input signal HV/HH (n_obs, nxn, n_time, n_pols=2)
            z2: input signal ks0 from bare interp (n_obs, nxn, 1, 1)
            sm0: input signal sm0 from bare interp (n_obs, nxn, n_time, 1)
            param0_file: file storing the initial parameters

        Returns: W (agb), S (agb) - with the dimension (obs, n_time)

        """

        n_obs = z1.shape[0]
        nxn = z1.shape[1]
        m_t = z1.shape[2]
        n_pols = z1.shape[3]

        if param0_file is None:
            in_path = os.path.dirname(self.mask_file)
            param0_file = "{}/model_sim_param0_s13.csv".format(in_path)
        df0 = pd.read_csv(param0_file)
        param0 = df0.iloc[0, 1 : n_pols * 6 + 1].values
        param_L = z2[:, 0, :, 0]  # ks (n_obs, 1)

        param0n = np.tile(param0, (n_obs, 1))
        param0LE = np.concatenate([param0n, param_L], axis=1)
        # param0s = np.tile(np.full((m_t,), sm0), (n_obs, 1))
        param0s = sm0[:, 0, :, 0]  # initial SM
        # hv90 = np.nanquantile(z1[:, 0, :, 1], 0.9, axis=1)
        # param0s[hv90>0.05, :] = z1[hv90>0.05, 0, :, 1] + 0.05   # HH as initial SM
        # param0s[hv90<=0.05, :] = z1[hv90<=0.05, 0, :, 1] * 10   # HH as initial SM
        param0s[:, :] = z1[:, 0, :, 1] + 0.05  # HH as initial SM

        mean_hv = z1[:, 0, :, 0]
        param0w = (100 * mean_hv) ** 2  # dimension: (n_obs, m_t)
        # param0w = z2[:, 0, :, 0]  # AGB (n_obs, m_t)
        # param0w = np.tile(np.nanmean(param0w, axis=1), [1, m_t])
        param0Ls = np.concatenate([param0LE, param0s], axis=1)
        param0c = np.concatenate((param0Ls, param0w), axis=1)

        bound0c = [
            [
                (w0 * 0.9999, w0 * 1.0001) for w0 in param0c[iw, : n_pols * 6]
            ]  # A,B,C,a,c,D
            + [
                (w0 * 0.9999, w0 * 1.0001)
                for w0 in param0c[iw, n_pols * 6 : n_pols * 6 + nxn]
            ]  # L
            + [
                (w0 * 0.6, w0 * 1.6)
                for w0 in param0c[iw, n_pols * 6 + nxn : n_pols * 6 + nxn + m_t]
            ]  # SM
            # + [(w0 * 0.2, 0.6) for w0 in param0c[iw, n_pols*6 + nxn:n_pols*6 + nxn + m_t]]    # SM
            + [
                (w0 * 0.9999, w0 * 1.0001)
                for w0 in param0c[iw, n_pols * 6 + nxn + m_t :]
            ]  # W
            for iw in range(param0c.shape[0])
        ]
        model_00 = Retrieval(n_pols, m_t, 1)
        param0c, y_hat = model_00.sar_model_inverse_03_sme(
            param0c, z1, bound0c, init_weights=init_weights
        )
        S_mean = param0c[:, n_pols * 6 + nxn : n_pols * 6 + nxn + m_t]
        W_mean = param0c[:, n_pols * 6 + nxn + m_t :]

        # S_lst = []
        # W_lst = []
        # for i in range(m_t):
        #     model_00 = Retrieval(n_pols, 1, 1)
        #     z1t = z1[:, :, i:i + 1, :]
        #     param0ct = np.concatenate([param0c[:, :n_pols*6 + nxn],
        #                                param0c[:, n_pols*6 + nxn + i:n_pols*6 + nxn + i + 1],
        #                                param0c[:, n_pols*6 + nxn + m_t + i:n_pols*6 + nxn + m_t + i + 1]
        #                                ], axis=1)
        #     bound0ct = [
        #         [(w0 * 0.9999, w0 * 1.0001) for w0 in param0ct[iw, :n_pols*6]]  # A,B,C,a,c,D
        #         + [(w0 * 0.9999, w0 * 1.0001) for w0 in param0ct[iw, n_pols*6:n_pols*6 + nxn]]  # L
        #         + [(w0 * 0.6, w0 * 1.6) for w0 in param0ct[iw, n_pols*6 + nxn:n_pols*6 + nxn + 1]]  # SM
        #         + [(w0 * 0.2, w0 * 1.8) for w0 in param0ct[iw, n_pols*6 + nxn + 1:]]  # W
        #         for iw in range(param0ct.shape[0])
        #     ]
        #     param0ct, y_hat = model_00.sar_model_inverse_03_sme(param0ct, z1t, bound0ct, init_weights=init_weights)
        #     S_t = param0ct[:, n_pols*6+nxn:n_pols*6+nxn+1]
        #     S_lst.append(S_t)
        #     W_lst.append(param0ct[:, n_pols*6+nxn+1:])
        # S_mean = np.concatenate(S_lst, axis=1)
        # W_mean = np.concatenate(W_lst, axis=1)

        print(f"Dimension of S: {S_mean.shape}")

        return S_mean, W_mean

    def inversion_nobs_sm_03h1_wkks(
        self, z1, z2, sm0, param0_file=None, init_weights=[1, 1]
    ):
        """
        W is from mean HV, and has no change in this retrieval, ks no change
        Args:
            sm0: initial S0
            z1: input signal HV/HH (n_obs, nxn, n_time, n_pols=2)
            z2: input signal ks0 from bare interp (n_obs, nxn, 1, 1)
            sm0: input signal sm0 from bare interp (n_obs, nxn, n_time, 1)
            param0_file: file storing the initial parameters

        Returns: W (agb), S (agb) - with the dimension (obs, n_time)

        """

        n_obs = z1.shape[0]
        nxn = z1.shape[1]
        m_t = z1.shape[2]
        n_pols = z1.shape[3]

        if param0_file is None:
            in_path = os.path.dirname(self.mask_file)
            param0_file = "{}/model_sim_param0_s13.csv".format(in_path)
        df0 = pd.read_csv(param0_file)
        param0 = df0.iloc[0, 1 : n_pols * 6 + 1].values
        param_L = z2[:, 0, :, 0]  # ks (n_obs, 1)

        param0n = np.tile(param0, (n_obs, 1))
        param0LE = np.concatenate([param0n, param_L], axis=1)
        # param0s = np.tile(np.full((m_t,), sm0), (n_obs, 1))
        param0s = sm0[:, 0, :, 0]  # initial SM
        hv90 = np.nanquantile(z1[:, 0, :, 1], 0.9, axis=1)
        param0s[hv90 > 0.05, :] = z1[hv90 > 0.05, 0, :, 1] + 0.05  # HH as initial SM
        param0s[hv90 <= 0.05, :] = z1[hv90 <= 0.05, 0, :, 1] * 10  # HH as initial SM

        param0Ls = np.concatenate([param0LE, param0s], axis=1)

        mean_hv = np.tile(np.nanmean(z1[:, 0, :, 0], axis=-1)[:, None], (1, m_t))
        param0w = (100 * mean_hv) ** 2  # dimension: (n_obs, 1)
        # param0w = z2[:, 0, :, 0]  # AGB (n_obs, m_t)
        # param0w = np.tile(np.nanmean(param0w, axis=1), [1, m_t])
        param0c = np.concatenate((param0Ls, param0w), axis=1)

        bound0c = [
            [
                (w0 * 0.9999, w0 * 1.0001) for w0 in param0c[iw, : n_pols * 6]
            ]  # A,B,C,a,c,D
            + [
                (w0 * 0.9999, w0 * 1.0001)
                for w0 in param0c[iw, n_pols * 6 : n_pols * 6 + nxn]
            ]  # L
            + [
                (w0 * 0.6, w0 * 1.6)
                for w0 in param0c[iw, n_pols * 6 + nxn : n_pols * 6 + nxn + m_t]
            ]  # SM
            # + [(w0 * 0.2, w0 * 2) for w0 in param0c[iw, n_pols*6 + nxn:n_pols*6 + nxn + m_t]]    # SM
            + [
                (w0 * 0.9999, w0 * 1.0001)
                for w0 in param0c[iw, n_pols * 6 + nxn + m_t :]
            ]  # W
            for iw in range(param0c.shape[0])
        ]
        model_00 = Retrieval(n_pols, m_t, 1)
        param0c, y_hat = model_00.sar_model_inverse_03_sme(
            param0c, z1, bound0c, init_weights=init_weights
        )
        S_mean = param0c[:, n_pols * 6 + nxn : n_pols * 6 + nxn + m_t]
        W_mean = param0c[:, n_pols * 6 + nxn + m_t :]

        # S_lst = []
        # W_lst = []
        # for i in range(m_t):
        #     model_00 = Retrieval(n_pols, 1, 1)
        #     z1t = z1[:, :, i:i + 1, :]
        #     param0ct = np.concatenate([param0c[:, :n_pols*6 + nxn],
        #                                param0c[:, n_pols*6 + nxn + i:n_pols*6 + nxn + i + 1],
        #                                param0c[:, n_pols*6 + nxn + m_t + i:n_pols*6 + nxn + m_t + i + 1]
        #                                ], axis=1)
        #     bound0ct = [
        #         [(w0 * 0.9999, w0 * 1.0001) for w0 in param0ct[iw, :n_pols*6]]  # A,B,C,a,c,D
        #         + [(w0 * 0.9999, w0 * 1.0001) for w0 in param0ct[iw, n_pols*6:n_pols*6 + nxn]]  # L
        #         + [(w0 * 0.6, w0 * 1.6) for w0 in param0ct[iw, n_pols*6 + nxn:n_pols*6 + nxn + 1]]  # SM
        #         + [(w0 * 0.2, w0 * 1.8) for w0 in param0ct[iw, n_pols*6 + nxn + 1:]]  # W
        #         for iw in range(param0ct.shape[0])
        #     ]
        #     param0ct, y_hat = model_00.sar_model_inverse_03_sme(param0ct, z1t, bound0ct, init_weights=init_weights)
        #     S_t = param0ct[:, n_pols*6+nxn:n_pols*6+nxn+1]
        #     S_lst.append(S_t)
        #     W_lst.append(param0ct[:, n_pols*6+nxn+1:])
        # S_mean = np.concatenate(S_lst, axis=1)
        # W_mean = np.concatenate(W_lst, axis=1)

        print(f"Dimension of S: {S_mean.shape}")

        return S_mean, W_mean

    def inversion_nobs_sm_03c_wkagb(
        self, z1, z2, sm0=0.2, param0_file=None, init_weights=[1, 1]
    ):
        """

        Args:
            sm0: initial S0
            z1: input signal SAR (n_obs, nxn, n_time, 2_pols)
            z2: input signal VWC (n_obs, nxn, n_time, 1)
            param0_file: file storing the initial parameters

        Returns: W (agb), S (agb) - with the dimension (obs, n_time)

        """

        n_obs = z1.shape[0]
        nxn = z1.shape[1]
        m_t = z1.shape[2]

        if param0_file is None:
            in_path = os.path.dirname(self.mask_file)
            param0_file = "{}/model_sim_param0_s13.csv".format(in_path)
        df0 = pd.read_csv(param0_file)
        param0 = df0.iloc[0, 1:13].values
        param_L = df0.iloc[0, 13 : 13 + n_obs].values[:, None]
        param_E = df0.iloc[0, 13 + n_obs :].values[:, None]

        assert param_E.shape[0] == n_obs, "Parameter E shape not correct"

        param0n = np.tile(param0, (n_obs, 1))
        param0LE = np.concatenate([param0n, param_L, param_E], axis=1)
        # param0s = np.tile(np.full((m_t,), sm0), (n_obs, 1))
        param0s = z1[:, 0, :, 1] * 4  # HH as initial SM
        param0Ls = np.concatenate([param0LE, param0s], axis=1)
        param0w = z2[:, 0, :, 0]  # AGB (n_obs, m_t)
        param0c = np.concatenate((param0Ls, param0w), axis=1)

        bound0c = [
            [(w0 * 0.9999, w0 * 1.0001) for w0 in param0c[iw, :12]]  # A,B,C,a,c,D
            + [(w0 * 0.9999, w0 * 1.0001) for w0 in param0c[iw, 12 : 12 + nxn]]  # L
            + [
                (w0 * 0.9999, w0 * 1.0001)
                for w0 in param0c[iw, 12 + nxn : 12 + nxn * 2]
            ]  # E
            + [
                (w0 * 0.2, w0 * 4.0)
                for w0 in param0c[iw, 12 + nxn * 2 : 12 + nxn * 2 + m_t]
            ]  # SM
            # + [(0.04, 0.5) for w0 in param0c[iw, 12 + nxn:12 + nxn + m_t]]    # SM
            + [
                (w0 * 0.9999, w0 * 1.0001) for w0 in param0c[iw, 12 + nxn * 2 + m_t :]
            ]  # W
            for iw in range(param0c.shape[0])
        ]
        model_00 = Retrieval(2, m_t, 1)
        param0c, y_hat = model_00.sar_model_inverse_03_smc(
            param0c, z1, bound0c, init_weights=init_weights
        )
        S_mean = param0c[:, 14 : 14 + m_t]
        print(f"Dimension of S: {S_mean.shape}")

        return S_mean

    def inversion_nobs_sm_03d_wkagb(
        self, z1, z2, sm0=0.2, param0_file=None, init_weights=[1, 1]
    ):
        """

        Args:
            sm0: initial S0
            z1: input signal SAR (n_obs, nxn, n_time, 2_pols)
            z2: input signal VWC (n_obs, nxn, n_time, 1)
            param0_file: file storing the initial parameters

        Returns: W (agb), S (agb) - with the dimension (obs, n_time)

        """

        n_obs = z1.shape[0]
        nxn = z1.shape[1]
        m_t = z1.shape[2]

        if param0_file is None:
            in_path = os.path.dirname(self.mask_file)
            param0_file = "{}/model_sim_param0_s13.csv".format(in_path)
        df0 = pd.read_csv(param0_file)
        param0 = df0.iloc[0, 1:13].values
        param_L = df0.iloc[0, 13 : 13 + n_obs].values[:, None]
        param_E = df0.iloc[0, 13 + n_obs : 13 + n_obs * 2].values[:, None]
        param_G = df0.iloc[0, 13 + n_obs * 2 :].values[:, None]

        assert param_G.shape[0] == n_obs, "Parameter G shape not correct"

        param0n = np.tile(param0, (n_obs, 1))
        param0LE = np.concatenate([param0n, param_L, param_E, param_G], axis=1)
        # param0s = np.tile(np.full((m_t,), sm0), (n_obs, 1))
        param0s = z1[:, 0, :, 1] * 4  # HH as initial SM
        param0Ls = np.concatenate([param0LE, param0s], axis=1)
        param0w = z2[:, 0, :, 0]  # AGB (n_obs, m_t)
        param0c = np.concatenate((param0Ls, param0w), axis=1)

        bound0c = [
            [(w0 * 0.9999, w0 * 1.0001) for w0 in param0c[iw, :12]]  # A,B,C,a,c,D
            + [(w0 * 0.9999, w0 * 1.0001) for w0 in param0c[iw, 12 : 12 + nxn]]  # L
            + [
                (w0 * 0.9999, w0 * 1.0001)
                for w0 in param0c[iw, 12 + nxn : 12 + nxn * 2]
            ]  # E
            + [
                (w0 * 0.9999, w0 * 1.0001)
                for w0 in param0c[iw, 12 + nxn * 2 : 12 + nxn * 3]
            ]  # G
            + [
                (w0 * 0.2, w0 * 4.0)
                for w0 in param0c[iw, 12 + nxn * 3 : 12 + nxn * 3 + m_t]
            ]  # SM
            # + [(0.04, 0.5) for w0 in param0c[iw, 12 + nxn:12 + nxn + m_t]]    # SM
            + [
                (w0 * 0.9999, w0 * 1.0001) for w0 in param0c[iw, 12 + nxn * 3 + m_t :]
            ]  # W
            for iw in range(param0c.shape[0])
        ]
        model_00 = Retrieval(2, m_t, 1)
        param0c, y_hat = model_00.sar_model_inverse_03_smd(
            param0c, z1, bound0c, init_weights=init_weights
        )
        S_mean = param0c[:, 15 : 15 + m_t]
        print(f"Dimension of S: {S_mean.shape}")

        return S_mean

    def inversion_nobs_sm_03e_wkagb(
        self, z1, z2, sm0=0.2, param0_file=None, init_weights=[1, 1]
    ):
        """

        Args:
            sm0: initial S0
            z1: input signal SAR (n_obs, nxn, n_time, n_pols)
            z2: input signal VWC (n_obs, nxn, n_time, 1)
            param0_file: file storing the initial parameters

        Returns: W (agb), S (agb) - with the dimension (obs, n_time)

        """

        n_obs = z1.shape[0]
        nxn = z1.shape[1]
        m_t = z1.shape[2]
        n_pols = z1.shape[3]

        if param0_file is None:
            in_path = os.path.dirname(self.mask_file)
            param0_file = "{}/model_sim_param0_s13.csv".format(in_path)
        df0 = pd.read_csv(param0_file)
        param0 = df0.iloc[0, 1 : n_pols * 6 + 1].values
        param_L = df0.iloc[0, n_pols * 6 + 1 : n_pols * 6 + 1 + n_obs].values[:, None]

        param0n = np.tile(param0, (n_obs, 1))
        param0LE = np.concatenate([param0n, param_L], axis=1)
        # param0s = np.tile(np.full((m_t,), sm0), (n_obs, 1))
        param0s = z1[:, 0, :, 1] * 4  # HH as initial SM
        # param0s = z1[:, 0, :, 1] + 0.05   # HH as initial SM
        param0Ls = np.concatenate([param0LE, param0s], axis=1)
        param0w = z2[:, 0, :, 0]  # AGB (n_obs, m_t)
        param0c = np.concatenate((param0Ls, param0w), axis=1)

        # bound0c = [
        #     [(w0 * 0.9999, w0 * 1.0001) for w0 in param0c[iw, :12]]                  # A,B,C,a,c,D
        #     + [(w0 * 0.9999, w0 * 1.0001) for w0 in param0c[iw, 12:12 + nxn]]        # L
        #     + [(w0 * 0.2, w0 * 3.0) for w0 in param0c[iw, 12 + nxn:12 + nxn + m_t]]    # SM
        #     # + [(0.04, 0.5) for w0 in param0c[iw, 12 + nxn:12 + nxn + m_t]]    # SM
        #     + [(w0 * 0.9999, w0 * 1.0001) for w0 in param0c[iw, 12 + nxn + m_t:]]    # W
        #     for iw in range(param0c.shape[0])
        # ]
        # model_00 = Retrieval(2, m_t, 1)
        # param0c, y_hat = model_00.sar_model_inverse_03_sme(param0c, z1, bound0c, init_weights=init_weights)
        # S_mean = param0c[:, 13:13+m_t]

        S_lst = []
        for i in range(m_t):
            model_00 = Retrieval(n_pols, 1, 1)
            z1t = z1[:, :, i : i + 1, :]
            param0ct = np.concatenate(
                [
                    param0c[:, : n_pols * 6 + nxn],
                    param0c[:, n_pols * 6 + nxn + i : n_pols * 6 + nxn + i + 1],
                    param0c[
                        :, n_pols * 6 + nxn + m_t + i : n_pols * 6 + nxn + m_t + i + 1
                    ],
                ],
                axis=1,
            )
            bound0ct = [
                [
                    (w0 * 0.9999, w0 * 1.0001) for w0 in param0ct[iw, : n_pols * 6]
                ]  # A,B,C,a,c,D
                + [
                    (w0 * 0.9999, w0 * 1.0001)
                    for w0 in param0ct[iw, n_pols * 6 : n_pols * 6 + nxn]
                ]  # L
                + [
                    (w0 * 0.2, w0 * 2.6)
                    for w0 in param0ct[iw, n_pols * 6 + nxn : n_pols * 6 + nxn + 1]
                ]  # SM
                # + [(w0 * 0.6, w0 * 1.6) for w0 in param0ct[iw, n_pols*6 + nxn:n_pols*6 + nxn + 1]]  # SM
                + [
                    (w0 * 0.9999, w0 * 1.0001)
                    for w0 in param0ct[iw, n_pols * 6 + nxn + 1 :]
                ]  # W
                for iw in range(param0ct.shape[0])
            ]
            param0ct, y_hat = model_00.sar_model_inverse_03_sme(
                param0ct, z1t, bound0ct, init_weights=init_weights
            )
            S_t = param0ct[:, n_pols * 6 + 1 : n_pols * 6 + 2]
            S_lst.append(S_t)
        S_mean = np.concatenate(S_lst, axis=1)

        print(f"Dimension of S: {S_mean.shape}")

        return S_mean

    def inversion_nobs_sm_03e_wkhv(
        self, z1, z2, sm0=0.2, param0_file=None, init_weights=[1, 1]
    ):
        """

        Args:
            sm0: initial S0
            z1: input signal HH/VV (n_obs, nxn, n_time, 2_pols)
            z2: input signal W = (100HV)**2 (n_obs, nxn, n_time, 1)
            param0_file: file storing the initial parameters

        Returns: W (agb), S (agb) - with the dimension (obs, n_time)

        """

        n_obs = z1.shape[0]
        nxn = z1.shape[1]
        m_t = z1.shape[2]
        n_pols = z1.shape[3]

        if param0_file is None:
            in_path = os.path.dirname(self.mask_file)
            param0_file = "{}/model_sim_param0_s13.csv".format(in_path)
        df0 = pd.read_csv(param0_file)
        param0 = df0.iloc[0, 1 : n_pols * 6 + 1].values
        param_L = df0.iloc[0, n_pols * 6 + 1 : n_pols * 6 + 1 + n_obs].values[:, None]

        param0n = np.tile(param0, (n_obs, 1))
        param0LE = np.concatenate([param0n, param_L], axis=1)
        # param0s = np.tile(np.full((m_t,), sm0), (n_obs, 1))
        param0s = z1[:, 0, :, 1] * 4  # HH as initial SM
        param0Ls = np.concatenate([param0LE, param0s], axis=1)
        param0w = z2[:, 0, :, 0]  # AGB (n_obs, m_t)
        # param0w = np.tile(np.nanmean(param0w, axis=1), [1, m_t])
        param0c = np.concatenate((param0Ls, param0w), axis=1)

        # bound0c = [
        #     [(w0 * 0.9999, w0 * 1.0001) for w0 in param0c[iw, :12]]                  # A,B,C,a,c,D
        #     + [(w0 * 0.9999, w0 * 1.0001) for w0 in param0c[iw, 12:12 + nxn]]        # L
        #     + [(w0 * 0.2, w0 * 3.0) for w0 in param0c[iw, 12 + nxn:12 + nxn + m_t]]    # SM
        #     # + [(0.04, 0.5) for w0 in param0c[iw, 12 + nxn:12 + nxn + m_t]]    # SM
        #     + [(w0 * 0.8, w0 * 1.25) for w0 in param0c[iw, 12 + nxn + m_t:]]    # W
        #     for iw in range(param0c.shape[0])
        # ]
        # model_00 = Retrieval(2, m_t, 1)
        # param0c, y_hat = model_00.sar_model_inverse_03_sme(param0c, z1, bound0c, init_weights=init_weights)
        # S_mean = param0c[:, 13:13+m_t]

        S_lst = []
        for i in range(m_t):
            model_00 = Retrieval(n_pols, 1, 1)
            z1t = z1[:, :, i : i + 1, :]
            param0ct = np.concatenate(
                [
                    param0c[:, : n_pols * 6 + nxn],
                    param0c[:, n_pols * 6 + nxn + i : n_pols * 6 + nxn + i + 1],
                    param0c[
                        :, n_pols * 6 + nxn + m_t + i : n_pols * 6 + nxn + m_t + i + 1
                    ],
                ],
                axis=1,
            )
            bound0ct = [
                [
                    (w0 * 0.9999, w0 * 1.0001) for w0 in param0ct[iw, : n_pols * 6]
                ]  # A,B,C,a,c,D
                # + [(w0 * 0.8, w0 * 1.25) for w0 in param0ct[iw, 12:12 + nxn]]        # L
                + [
                    (0.0001, 0.9999)
                    for w0 in param0ct[iw, n_pols * 6 : n_pols * 6 + nxn]
                ]  # L
                + [
                    (w0 * 0.2, 0.5)
                    for w0 in param0ct[iw, n_pols * 6 + nxn : n_pols * 6 + nxn + 1]
                ]  # SM
                + [
                    (w0 * 0.9999, w0 * 1.0001)
                    for w0 in param0ct[iw, n_pols * 6 + nxn + 1 :]
                ]  # W
                for iw in range(param0ct.shape[0])
            ]
            param0ct, y_hat = model_00.sar_model_inverse_03_sme(
                param0ct, z1t, bound0ct, init_weights=init_weights
            )
            S_t = param0ct[:, n_pols * 6 + nxn : n_pols * 6 + nxn + 1]
            S_lst.append(S_t)
        S_mean = np.concatenate(S_lst, axis=1)

        print(f"Dimension of S: {S_mean.shape}")

        return S_mean

    def inversion_nobs_sm_03e_wkks(
        self, z1, z2, sm0, param0_file=None, init_weights=[1, 1]
    ):
        """

        Args:
            sm0: initial S0
            z1: input signal HH/VV (n_obs, nxn, n_time, n_pols)
            z2: input signal ks0 from bare interp (n_obs, nxn, 1, 1)
            sm0: input signal sm0 from bare interp (n_obs, nxn, n_time, 1)
            param0_file: file storing the initial parameters

        Returns: W (agb), S (agb) - with the dimension (obs, n_time)

        """

        n_obs = z1.shape[0]
        nxn = z1.shape[1]
        m_t = z1.shape[2]
        n_pols = z1.shape[3]

        if param0_file is None:
            in_path = os.path.dirname(self.mask_file)
            param0_file = "{}/model_sim_param0_s13.csv".format(in_path)
        df0 = pd.read_csv(param0_file)
        param0 = df0.iloc[0, 1 : n_pols * 6 + 1].values
        param_L = z2[:, 0, :, 0]  # ks (n_obs, 1)

        param0n = np.tile(param0, (n_obs, 1))
        param0LE = np.concatenate([param0n, param_L], axis=1)
        # param0s = np.tile(np.full((m_t,), sm0), (n_obs, 1))
        param0s = sm0[:, 0, :, 0]  # initial SM
        # param0s = z1[:, 0, :, 1] * 4   # HH as initial SM
        param0Ls = np.concatenate([param0LE, param0s], axis=1)

        mean_hv = z1[:, 0, :, 0]
        param0w = (100 * mean_hv) ** 2  # dimension: (n_obs, nxn)
        # param0w = z2[:, 0, :, 0]  # AGB (n_obs, m_t)
        # param0w = np.tile(np.nanmean(param0w, axis=1), [1, m_t])
        param0c = np.concatenate((param0Ls, param0w), axis=1)

        bound0c = [
            [
                (w0 * 0.9999, w0 * 1.0001) for w0 in param0c[iw, : n_pols * 6]
            ]  # A,B,C,a,c,D
            + [
                (w0 * 0.9, w0 * 1.1)
                for w0 in param0c[iw, n_pols * 6 : n_pols * 6 + nxn]
            ]  # L
            + [
                (w0 * 0.6, w0 * 1.4)
                for w0 in param0c[iw, n_pols * 6 + nxn : n_pols * 6 + nxn + m_t]
            ]  # SM
            + [
                (w0 * 0.2, w0 * 1.8) for w0 in param0c[iw, n_pols * 6 + nxn + m_t :]
            ]  # W
            for iw in range(param0c.shape[0])
        ]
        model_00 = Retrieval(n_pols, m_t, 1)
        param0c, y_hat = model_00.sar_model_inverse_03_sme(
            param0c, z1, bound0c, init_weights=init_weights
        )
        S_mean = param0c[:, n_pols * 6 + nxn : n_pols * 6 + nxn + m_t]
        W_mean = param0c[:, n_pols * 6 + nxn + m_t :]

        # S_lst = []
        # W_lst = []
        # for i in range(m_t):
        #     model_00 = Retrieval(n_pols, 1, 1)
        #     z1t = z1[:, :, i:i + 1, :]
        #     param0ct = np.concatenate([param0c[:, :n_pols*6 + nxn],
        #                                param0c[:, n_pols*6 + nxn + i:n_pols*6 + nxn + i + 1],
        #                                param0c[:, n_pols*6 + nxn + m_t + i:n_pols*6 + nxn + m_t + i + 1]
        #                                ], axis=1)
        #     bound0ct = [
        #         [(w0 * 0.9999, w0 * 1.0001) for w0 in param0ct[iw, :n_pols*6]]  # A,B,C,a,c,D
        #         + [(w0 * 0.9999, w0 * 1.0001) for w0 in param0ct[iw, n_pols*6:n_pols*6 + nxn]]  # L
        #         + [(w0 * 0.6, w0 * 1.6) for w0 in param0ct[iw, n_pols*6 + nxn:n_pols*6 + nxn + 1]]  # SM
        #         + [(w0 * 0.2, w0 * 1.8) for w0 in param0ct[iw, n_pols*6 + nxn + 1:]]  # W
        #         for iw in range(param0ct.shape[0])
        #     ]
        #     param0ct, y_hat = model_00.sar_model_inverse_03_sme(param0ct, z1t, bound0ct, init_weights=init_weights)
        #     S_t = param0ct[:, n_pols*6+nxn:n_pols*6+nxn+1]
        #     S_lst.append(S_t)
        #     W_lst.append(param0ct[:, n_pols*6+nxn+1:])
        # S_mean = np.concatenate(S_lst, axis=1)
        # W_mean = np.concatenate(W_lst, axis=1)

        print(f"Dimension of S: {S_mean.shape}")

        return S_mean, W_mean

    def inversion_nobs_sm_03e_wkagb_iks(
        self, z1, z2, L0=0.2, param0_file=None, init_weights=[1, 1]
    ):
        """

        Args:
            L0: initial L
            z1: input signal SAR (n_obs, nxn, n_time, n_pols)
            z2: input signal VWC (n_obs, nxn, n_time, 1)
            param0_file: file storing the initial parameters

        Returns: W (agb), S (agb) - with the dimension (obs, n_time)

        """

        n_obs = z1.shape[0]
        nxn = z1.shape[1]
        m_t = z1.shape[2]
        n_pols = z1.shape[3]

        if param0_file is None:
            in_path = os.path.dirname(self.mask_file)
            param0_file = "{}/model_sim_param0_s13.csv".format(in_path)
        df0 = pd.read_csv(param0_file)
        param0 = df0.iloc[0, 1 : n_pols * 6].values
        param_L = L0[:, None]

        param0n = np.tile(param0, (n_obs, 1))
        param0LE = np.concatenate([param0n, param_L], axis=1)
        # param0s = np.tile(np.full((m_t,), sm0), (n_obs, 1))
        param0s = z1[:, 0, :, 1] * 4  # HH as initial SM
        param0Ls = np.concatenate([param0LE, param0s], axis=1)
        param0w = z2[:, 0, :, 0]  # AGB (n_obs, m_t)
        param0c = np.concatenate((param0Ls, param0w), axis=1)

        bound0c = [
            [
                (w0 * 0.9999, w0 * 1.0001) for w0 in param0c[iw, : n_pols * 6]
            ]  # A,B,C,a,c,D
            + [
                (w0 * 0.1, w0 * 3.0)
                for w0 in param0c[iw, n_pols * 6 : n_pols * 6 + nxn]
            ]  # L
            + [
                (w0 * 0.2, 0.5)
                for w0 in param0c[iw, n_pols * 6 + nxn : n_pols * 6 + nxn + m_t]
            ]  # SM
            + [
                (w0 * 0.9999, w0 * 1.0001)
                for w0 in param0c[iw, n_pols * 6 + nxn + m_t :]
            ]  # W
            for iw in range(param0c.shape[0])
        ]
        model_00 = Retrieval(n_pols, m_t, 1)
        param0c, y_hat = model_00.sar_model_inverse_03_sme(
            param0c, z1, bound0c, init_weights=init_weights
        )
        S_mean = param0c[:, n_pols * 6 + nxn : n_pols * 6 + nxn + m_t]

        # S_lst = []
        # for i in range(m_t):
        #     model_00 = Retrieval(2, 1, 1)
        #     z1t = z1[:, :, i:i+1, :]
        #     param0ct = np.concatenate([param0c[:, :12+nxn],
        #                                param0c[:, 12+nxn+i:12+nxn+i+1],
        #                                param0c[:, 12+nxn+m_t+i:12+nxn+m_t+i+1]
        #                                ], axis=1)
        #     bound0ct = [
        #         [(w0 * 0.9999, w0 * 1.0001) for w0 in param0ct[iw, :12]]                  # A,B,C,a,c,D
        #         + [(w0 * 0.5, w0 * 2.0) for w0 in param0ct[iw, 12:12 + nxn]]        # L
        #         + [(w0 * 0.2, w0 * 4.0) for w0 in param0ct[iw, 12 + nxn:12 + nxn + 1]]    # SM
        #         + [(w0 * 0.9999, w0 * 1.0001) for w0 in param0ct[iw, 12 + nxn + 1:]]    # W
        #         for iw in range(param0ct.shape[0])
        #     ]
        #     param0ct, y_hat = model_00.sar_model_inverse_03_sme(param0ct, z1t, bound0ct, init_weights=init_weights)
        #     S_t = param0ct[:, 13:14]
        #     S_lst.append(S_t)
        # S_mean = np.concatenate(S_lst, axis=1)

        print(f"Dimension of S: {S_mean.shape}")

        return S_mean


def plotly_scatter_00(wd):
    os.chdir(wd)

    Ahh = widgets.FloatText(value=0.11, description="A_HH:", disabled=False)
    Ahv = widgets.FloatText(value=0.03, description="A_HV:", disabled=False)
    bhh = widgets.FloatText(value=0.009, description="B_HH:", disabled=False)
    bhv = widgets.FloatText(value=0.012, description="B_HV:", disabled=False)
    chh = widgets.FloatText(value=0.30, description="C_HH:", disabled=False)
    chv = widgets.FloatText(value=0.23, description="C_HV:", disabled=False)
    dhh = widgets.FloatText(value=10, description="D_HH:", disabled=False)
    st = widgets.FloatText(value=0.003, description="S_t:", disabled=False)
    ahh = widgets.FloatText(value=0.2, description="alpha_HH:", disabled=False)
    ahv = widgets.FloatText(value=0.18, description="alpha_HV:", disabled=False)
    ghh = widgets.FloatText(value=1.3, description="gamma_HH:", disabled=False)
    ghv = widgets.FloatText(value=1.1, description="gamma_HV:", disabled=False)

    def update_numbers(
        Ahv=0.03,
        Ahh=0.11,
        bhv=0.012,
        bhh=0.009,
        chv=0.23,
        chh=0.30,
        dhh=10,
        ahv=0.18,
        ahh=0.2,
        ghv=1.1,
        ghh=1.3,
        st=0.003,
    ):
        param0 = np.array(
            [Ahv, Ahh, bhv, bhh, chv, chh, ahv, ahh, ghv, ghh, 1, dhh, st]
        )
        n_obs = 31
        W0 = np.linspace(0, 300, n_obs)
        param0c = np.tile(param0, (n_obs, 1))
        param0c = np.concatenate((param0c, W0[:, None]), axis=1)
        model_00 = Retrieval(2, 1, 1)
        y_hat = []
        for i in range(n_obs):
            y_hat.append(model_00.sar_model_03_fun(param0c[i, :]))
        y_hat = np.array(y_hat)
        y_hv = y_hat[:, 0, 0]
        y_hh = y_hat[:, 0, 1]

        param_names = [
            "A_HV",
            "A_HH",
            "B_HV",
            "B_HH",
            "C_HV",
            "C_HH",
            "alpha_HV",
            "alpha_HH",
            "gamma_HV",
            "gamma_HH",
            "D_HV",
            "D_HH",
            "S",
        ]
        df = pd.DataFrame(param0[None, :], columns=param_names)
        df.to_csv("data/model_sim_param0_s13.csv")

        arr1 = np.array([W0, y_hh, y_hv]).transpose()
        df1 = pd.DataFrame(arr1, columns=["W0", "y_hh", "y_hv"])
        return df1

    df = update_numbers()
    trace1 = go.Scatter(
        x=df.W0.values,
        y=df.y_hv.values,
        mode="markers",
        name="HV",
        marker=dict(
            size=10,
        ),
    )
    trace2 = go.Scatter(
        x=df.W0.values,
        y=df.y_hh.values,
        mode="markers",
        name="HH",
        marker=dict(size=10, color="rgb(255, 127, 14)"),
    )
    layout1 = go.Layout(
        title="HV vs. AGB",
        xaxis={"title": "AGB (Mg/ha)"},
        yaxis={"title": "Backscatter HV"},
        width=500,
        height=400,
        margin={"l": 60, "b": 40, "t": 60, "r": 30},
        hovermode="closest",
    )
    layout2 = go.Layout(
        title="HH vs. AGB",
        xaxis={"title": "AGB (Mg/ha)"},
        yaxis={"title": "Backscatter HH"},
        width=500,
        height=400,
        margin={"l": 60, "b": 40, "t": 60, "r": 30},
        hovermode="closest",
    )
    g1 = go.FigureWidget(data=[trace1], layout=layout1)
    g2 = go.FigureWidget(data=[trace2], layout=layout2)

    def response(change):
        df = update_numbers(
            Ahv.value,
            Ahh.value,
            bhv.value,
            bhh.value,
            chv.value,
            chh.value,
            dhh.value,
            ahv.value,
            ahh.value,
            ghv.value,
            ghh.value,
            st.value,
        )
        with g1.batch_update():
            g1.data[0].x = df.W0.values
            g1.data[0].y = df.y_hv.values
        with g2.batch_update():
            g2.data[0].x = df.W0.values
            g2.data[0].y = df.y_hh.values

    Ahv.observe(response, names="value")
    Ahh.observe(response, names="value")
    bhv.observe(response, names="value")
    bhh.observe(response, names="value")
    chv.observe(response, names="value")
    chh.observe(response, names="value")
    dhh.observe(response, names="value")
    ahv.observe(response, names="value")
    ahh.observe(response, names="value")
    ghv.observe(response, names="value")
    ghh.observe(response, names="value")
    st.observe(response, names="value")

    container1 = widgets.HBox([Ahv, Ahh, bhv, bhh])
    container2 = widgets.HBox([chv, chh, dhh, st])
    container3 = widgets.HBox([ahv, ahh, ghv, ghh])
    container4 = widgets.HBox([g1, g2])
    app = widgets.VBox([container1, container2, container3, container4])
    return app


def plotly_scatter_01(wd):
    os.chdir(wd)

    w_slider = widgets.FloatSlider(
        value=30.0,
        min=10.0,
        max=50.0,
        step=0.1,
        description="AGB Noise (Mg/ha): ",
        disabled=False,
        orientation="horizontal",
        readout=True,
        readout_format=".1f",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="80%"),
    )
    h_slider = widgets.FloatSlider(
        value=0.04,
        min=0.01,
        max=0.06,
        step=0.01,
        description="HH Noise (Power): ",
        disabled=False,
        orientation="horizontal",
        readout=True,
        readout_format=".3f",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="80%"),
    )
    v_slider = widgets.FloatSlider(
        value=0.01,
        min=0.005,
        max=0.03,
        step=0.005,
        description="HV Noise (Power): ",
        disabled=False,
        orientation="horizontal",
        readout=True,
        readout_format=".3f",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="80%"),
    )

    def update_numbers(w_noise=30.0, h_noise=0.04, v_noise=0.01):
        basemap_file = "data/mask0.tif"
        mask_file = "data/mask.tif"
        out_name = "data/mld03a_v2"
        sim00 = ModelSimulation(basemap_file, mask_file, out_name=out_name)
        W1, y_hat0 = sim00.data_noisy_w_vs_alos(
            w_noise=w_noise,
            h_noise=h_noise,
            v_noise=v_noise,
            mask_file=mask_file,
            n_1=1,
        )
        y1v = y_hat0[:, :, 0]
        y1h = y_hat0[:, :, 1]
        # print(np.std(W1, axis=1).flatten().shape)

        w1m = np.mean(W1, axis=1).flatten()
        w1s = np.std(W1, axis=1).flatten()
        y1h_mean = np.mean(y1h, axis=1).flatten()
        y1v_mean = np.mean(y1v, axis=1).flatten()
        y1h_std = np.std(y1h, axis=1).flatten()
        y1v_std = np.std(y1v, axis=1).flatten()

        return w1m, w1s, y1h_mean, y1h_std, y1v_mean, y1v_std

    w1m, w1s, y1h_mean, y1h_std, y1v_mean, y1v_std = update_numbers()
    trace1 = go.Scatter(
        x=w1m,
        y=y1v_mean,
        mode="markers",
        name="HH",
        error_y=dict(
            type="data",
            array=y1v_std,
            thickness=0.5,
            width=2,
        ),
        error_x=dict(
            type="data",
            array=w1s,
            thickness=0.5,
            width=2,
        ),
        marker=dict(size=10),
    )
    trace2 = go.Scatter(
        x=w1m,
        y=y1h_mean,
        mode="markers",
        name="HV",
        error_y=dict(
            type="data",
            array=y1h_std,
            thickness=0.5,
            width=2,
            color="rgb(255, 127, 14)",
        ),
        error_x=dict(
            type="data",
            array=w1s,
            thickness=0.5,
            width=2,
            color="rgb(255, 127, 14)",
        ),
        marker=dict(
            size=10,
            color="rgb(255, 127, 14)",
        ),
    )
    layout1 = go.Layout(
        title="HV vs. AGB",
        xaxis={"title": "AGB (Mg/ha)"},
        yaxis={
            "title": "Backscatter HV",
        },
        width=500,
        height=400,
        margin={"l": 60, "b": 40, "t": 60, "r": 30},
        hovermode="closest",
    )
    layout2 = go.Layout(
        title="HH vs. AGB",
        xaxis={"title": "AGB (Mg/ha)"},
        yaxis={
            "title": "Backscatter HH",
        },
        width=500,
        height=400,
        margin={"l": 60, "b": 40, "t": 60, "r": 30},
        hovermode="closest",
    )
    g1 = go.FigureWidget(data=[trace1], layout=layout1)
    g2 = go.FigureWidget(data=[trace2], layout=layout2)

    def response(change):
        w1m, w1s, y1h_mean, y1h_std, y1v_mean, y1v_std = update_numbers(
            w_slider.value, h_slider.value, v_slider.value
        )
        with g1.batch_update():
            g1.data[0].x = w1m
            g1.data[0].y = y1v_mean
            g1.data[0].error_x = dict(
                type="data",
                array=w1s,
                thickness=0.5,
                width=2,
            )
            g1.data[0].error_y = dict(
                type="data",
                array=y1v_std,
                thickness=0.5,
                width=2,
            )

        with g2.batch_update():
            g2.data[0].x = w1m
            g2.data[0].y = y1h_mean
            g2.data[0].error_x = dict(
                type="data",
                array=w1s,
                thickness=0.5,
                width=2,
            )
            g2.data[0].error_y = dict(
                type="data",
                array=y1h_std,
                thickness=0.5,
                width=2,
            )

    w_slider.observe(response, names="value")
    h_slider.observe(response, names="value")
    v_slider.observe(response, names="value")

    container4 = widgets.HBox([g1, g2])
    app = widgets.VBox([w_slider, h_slider, v_slider, container4])
    return app


def plotly_scatter_02(wd, m_t=10):
    os.chdir(wd)
    basemap_file = "data/mask0.tif"
    mask_file = "data/mask.tif"
    out_name = "data/mld03a_v2"
    sim00 = ModelSimulation(basemap_file, mask_file, out_name=out_name)
    W_mean, y0 = sim00.inversion_recursive_ws_sim(m_t=m_t)

    r2 = np.zeros((m_t, 10))
    rmse = np.zeros((m_t, 10))
    W_mean2 = []
    for i in range(m_t):
        # print(i)
        # for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 9, 12, 10, 11]:
        x0 = np.zeros((W_mean.shape[0], 10))
        for k in range(10):
            x0r = np.roll(W_mean, k, axis=1)
            x0[:, k] = np.mean(x0r[:, : i + 1], axis=1)
            r2[i, k] = r2_score(x0[:, k], y0[:, None])
            rmse[i, k] = np.sqrt(mean_squared_error(x0[:, k], y0[:, None]))
        W_mean2.append(x0)

    trace1 = go.Scatter(
        x=np.arange(m_t) + 1,
        y=np.mean(r2, axis=1),
        mode="lines+markers",
        name="R<sup>2</sup>",
        marker=dict(size=10),
        error_y=dict(
            type="data",
            array=np.std(r2, axis=1),
            thickness=1,
            width=2,
        ),
    )
    layout1 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "R<sup>2</sup>",
        },
        width=350,
        height=250,
        margin={"l": 60, "b": 40, "t": 30, "r": 10},
        hovermode="closest",
    )
    trace2 = go.Scatter(
        x=np.arange(m_t) + 1,
        y=np.mean(rmse, axis=1),
        mode="lines+markers",
        name="RMSE",
        marker=dict(size=10),
        error_y=dict(
            type="data",
            array=np.std(rmse, axis=1),
            thickness=1,
            width=2,
        ),
    )
    layout2 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "RMSE",
        },
        width=350,
        height=250,
        margin={"l": 60, "b": 40, "t": 30, "r": 10},
        hovermode="closest",
    )
    g1 = go.FigureWidget(data=[trace1], layout=layout1)
    g2 = go.FigureWidget(data=[trace2], layout=layout2)
    mt_slider = widgets.IntSlider(
        value=1,
        min=1,
        max=m_t,
        step=1,
        disabled=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="80%"),
    )

    def update_numbers(m_ti=0):
        y = W_mean2[m_ti]
        ymean = np.mean(y, axis=1)
        x = y0
        # xy = np.vstack([x, ymean])
        # z0 = gaussian_kde(xy)(xy)
        xy = np.vstack([x, y[:, 0]])
        z0 = gaussian_kde(xy)(xy)
        return x, y, z0, ymean

    x, y, z0, ymean = update_numbers()
    trace3 = go.Scatter(
        x=x,
        y=ymean,
        mode="markers",
        name="points",
        error_y=dict(
            type="data",
            array=np.std(y, axis=1),
            thickness=0.5,
            width=2,
        ),
        marker=dict(
            color=z0,
            colorscale="Blues",
            reversescale=True,
            size=10,
            opacity=0.5,
            line=dict(width=1),
        ),
    )
    trace4 = go.Scatter(
        x=[0, 200],
        y=[0, 200],
        mode="lines",
        line=dict(width=2, color="rgb(0.8, 0.8, 0.8)"),
    )
    data = [trace3, trace4]
    layout4 = go.Layout(
        xaxis={"title": "Measured AGB", "range": [0, 200]},
        yaxis={"title": "Predicted AGB", "range": [0, 200]},
        width=600,
        height=500,
        margin={"l": 80, "b": 60, "t": 30, "r": 30},
        showlegend=False,
        hovermode="closest",
    )
    g3 = go.FigureWidget(data=data, layout=layout4)

    def response(change):
        x, y, z0, ymean = update_numbers(mt_slider.value - 1)
        with g3.batch_update():
            g3.data[0].x = x
            g3.data[0].y = ymean
            g3.data[0].marker = dict(
                color=z0,
                colorscale="Blues",
                reversescale=True,
                size=10,
                opacity=0.5,
                line=dict(width=1),
            )
            g3.data[0].error_y = dict(
                type="data",
                array=np.std(y, axis=1),
                thickness=0.5,
                width=2,
            )

    mt_slider.observe(response, names="value")

    container1 = widgets.VBox([g1, g2])
    container2 = widgets.HBox([container1, g3])
    app = widgets.VBox([widgets.Label("Number of Scenes: "), mt_slider, container2])

    return app


def plotly_scatter_03(wd):
    os.chdir(wd)

    w1_slider = widgets.FloatSlider(
        value=100.0,
        min=0.0,
        max=100.0,
        step=5,
        description="AGB Absolute Noise (Mg/ha): ",
        disabled=False,
        orientation="horizontal",
        readout=True,
        readout_format=".1f",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="80%"),
    )
    w2_slider = widgets.FloatSlider(
        value=100.0,
        min=0.0,
        max=100.0,
        step=5,
        description="AGB Relative Noise (% AGB): ",
        disabled=False,
        orientation="horizontal",
        readout=True,
        readout_format=".1f",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="80%"),
    )

    def update_numbers(w1_noise=100, w2_noise=100):
        mask_file = "lc-200-v2.tif"
        agb_file = "lidar_agb.tif"
        alos_file = pd.read_csv("alos_list.csv")["file_name"].tolist()
        out_radar_list = alos_file[14:]

        if not os.path.exists("../output"):
            os.mkdir("../output")
        out_name = "../output/field03a_v2"
        mdl00 = FieldRetrieval(mask_file, out_name=out_name)
        x0, y0, mask0 = mdl00.data_cleaner(
            out_radar_list,
            agb_file=agb_file,
            mask_file=mask_file,
            w1_noise=w1_noise,
            w2_noise=w2_noise,
        )
        # print(mask0)
        x0 = x0.reshape([y0.shape[0], -1, 2])
        x0v = x0[:, :, 0]
        x0h = x0[:, :, 1]

        # r2 = r2_score(
        #     scipy.stats.zscore(y0[:, None]),
        #     scipy.stats.zscore(np.mean(x0v, axis=1)[:, None]),
        # )
        # print("Variance score 1: {:.2f}".format(r2))
        # rmse = np.sqrt(
        #     mean_squared_error(
        #         scipy.stats.zscore(y0[:, None]),
        #         scipy.stats.zscore(np.mean(x0v, axis=1)[:, None]),
        #     )
        # )
        # print("RMSE: {:.5f}".format(rmse))
        # r2 = r2_score(
        #     scipy.stats.zscore(y0[:, None]),
        #     scipy.stats.zscore(np.mean(x0h, axis=1)[:, None]),
        # )
        # print("Variance score 1: {:.2f}".format(r2))
        # rmse = np.sqrt(
        #     mean_squared_error(
        #         scipy.stats.zscore(y0[:, None]),
        #         scipy.stats.zscore(np.mean(x0h, axis=1)[:, None]),
        #     )
        # )
        # print("RMSE: {:.5f}".format(rmse))

        return y0, x0v, x0h

    y0, x0v, x0h = update_numbers()
    trace1 = go.Scatter(
        x=y0,
        y=np.mean(x0v, axis=1),
        mode="markers",
        name="HV",
        error_y=dict(
            type="data",
            array=np.std(x0v, axis=1),
            thickness=0.5,
            width=2,
        ),
        marker=dict(size=10),
    )
    trace2 = go.Scatter(
        x=y0,
        y=np.mean(x0h, axis=1),
        mode="markers",
        name="HH",
        error_y=dict(
            type="data",
            array=np.std(x0h, axis=1),
            thickness=0.5,
            width=2,
            color="rgb(255, 127, 14)",
        ),
        marker=dict(
            size=10,
            color="rgb(255, 127, 14)",
        ),
    )
    layout1 = go.Layout(
        title="HV vs. AGB",
        xaxis={"title": "AGB (Mg/ha)"},
        yaxis={
            "title": "Backscatter HV",
        },
        width=500,
        height=400,
        margin={"l": 60, "b": 40, "t": 60, "r": 30},
        hovermode="closest",
    )
    layout2 = go.Layout(
        title="HH vs. AGB",
        xaxis={"title": "AGB (Mg/ha)"},
        yaxis={
            "title": "Backscatter HH",
        },
        width=500,
        height=400,
        margin={"l": 60, "b": 40, "t": 60, "r": 30},
        hovermode="closest",
    )

    g1 = go.FigureWidget(data=[trace1], layout=layout1)
    g2 = go.FigureWidget(data=[trace2], layout=layout2)
    label1 = widgets.Label(f"Number of Valid Obs. ({y0.shape[0]})")

    def response(change):
        y0, x0v, x0h = update_numbers(w1_slider.value, w2_slider.value)
        with g1.batch_update():
            g1.data[0].x = y0
            g1.data[0].y = np.mean(x0v, axis=1)
            g1.data[0].error_y = dict(
                type="data",
                array=np.std(x0v, axis=1),
                thickness=0.5,
                width=2,
            )
        with g2.batch_update():
            g2.data[0].x = y0
            g2.data[0].y = np.mean(x0h, axis=1)
            g2.data[0].error_y = dict(
                type="data",
                array=np.std(x0h, axis=1),
                thickness=0.5,
                width=2,
                color="rgb(255, 127, 14)",
            )
        label1.value = f"Number of Valid Obs. ({y0.shape[0]})"

    w1_slider.observe(response, names="value")
    w2_slider.observe(response, names="value")

    container4 = widgets.HBox([g1, g2])
    app = widgets.VBox([w1_slider, w2_slider, label1, container4])

    return app


def plotly_scatter_04(wd):
    os.chdir(wd)
    agb_file = "lidar_agb.tif"
    alos_file = pd.read_csv("alos_list.csv")["file_name"].tolist()
    out_radar_list = alos_file[14:]

    if not os.path.exists("../output"):
        os.mkdir("../output")
    mask_file = "../output/field03a_v2_mask_ws0.tif"
    out_name = "../output/field03a_v2"
    mdl00 = FieldRetrieval(mask_file, out_name=out_name)
    z1, W1, mask_ws1 = mdl00.inversion_setup(
        out_radar_list, agb_file=agb_file, mask_file=mask_file, n_1=0
    )

    z1_dim = z1.shape
    z1r = z1.reshape([-1, z1_dim[1] * z1_dim[2] * z1_dim[3]])
    z0 = mdl00.inversion_return_valid(z1r, mask_ws1, mask_file=mask_file)
    W0 = mdl00.inversion_return_valid(W1, mask_ws1, mask_file=mask_file)
    z0 = z0.reshape([-1, z1_dim[1], z1_dim[2], z1_dim[3]])
    param0_file = mdl00.params_calibration(z0, W0)

    W_mean, _ = mdl00.inversion_recursive_ws(z0, mask_file, param0_file=param0_file)
    y0 = W0

    m_t = W_mean.shape[1]
    y0 = y0.flatten()
    r2 = []
    rmse = []
    W_mean2 = []
    for i in range(m_t):
        # print(i)
        # for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 9, 12, 10, 11]:
        x0 = np.mean(W_mean[:, : i + 1], axis=1)
        r2_1 = r2_score(x0[:, None], y0[:, None])
        # print("Variance score 1: {:.2f}".format(r2_1))
        rmse_1 = np.sqrt(mean_squared_error(x0[:, None], y0[:, None]))
        # print("RMSE: {:.5f}".format(rmse_1))
        W_mean2.append(x0)
        r2.append(r2_1)
        rmse.append(rmse_1)

    r2 = np.array(r2)
    rmse = np.array(rmse)

    trace1 = go.Scatter(
        x=np.arange(m_t) + 1,
        y=r2,
        mode="lines+markers",
        name="R<sup>2</sup>",
        marker=dict(size=10),
    )
    trace2 = go.Scatter(
        x=np.arange(m_t) + 1,
        y=rmse,
        mode="lines+markers",
        name="RMSE",
        marker=dict(size=10),
    )
    layout1 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "R<sup>2</sup>",
        },
        width=400,
        height=250,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        hovermode="closest",
    )
    layout2 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "RMSE",
        },
        width=400,
        height=250,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        hovermode="closest",
    )
    g1 = go.FigureWidget(data=[trace1], layout=layout1)
    g2 = go.FigureWidget(data=[trace2], layout=layout2)
    mt_slider = widgets.IntSlider(
        value=1,
        min=1,
        max=m_t,
        step=1,
        disabled=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="80%"),
    )

    def update_numbers(m_ti=0):
        y = W_mean2[m_ti]
        x = y0
        xy = np.vstack([x, y])
        z0 = gaussian_kde(xy)(xy)
        p2 = np.sum((x - 20 < y) & (y < x + 20) & (x < 100)) / x[x < 100].size * 100
        p1 = np.sum((x - 10 < y) & (y < x + 10) & (x < 100)) / x[x < 100].size * 100
        return x, y, z0, p1, p2

    x, y, z0, p1, p2 = update_numbers()
    trace3 = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        name="points",
        marker=dict(
            color=z0,
            colorscale="Blues_r",
            reversescale=True,
            size=10,
            opacity=0.5,
            line=dict(width=1),
        ),
    )
    trace4 = go.Scatter(
        x=[0, 200],
        y=[0, 200],
        mode="lines",
        line=dict(width=2, color="rgb(0.8, 0.8, 0.8)"),
    )
    data = [trace3, trace4]
    layout3 = go.Layout(
        xaxis={
            "title": "Measured AGB",
            "range": [0, 200],
        },
        yaxis={
            "title": "Predicted AGB",
            "range": [0, 200],
        },
        width=500,
        height=500,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        showlegend=False,
        hovermode="closest",
        annotations=[
            dict(
                x=160,
                y=25,
                showarrow=False,
                text="{:.1f}% within 20 Mg/ha <br> {:.1f}% within 10 Mg/ha".format(
                    p2, p1
                ),
            )
        ],
    )
    g3 = go.FigureWidget(data=data, layout=layout3)

    def response(change):
        x, y, z0, p1, p2 = update_numbers(mt_slider.value - 1)
        with g3.batch_update():
            g3.data[0].x = x
            g3.data[0].y = y
            g3.data[0].marker = dict(
                color=z0,
                colorscale="Blues_r",
                reversescale=True,
                size=10,
                opacity=0.5,
                line=dict(width=1),
            )
            g3.layout.annotations[0][
                "text"
            ] = "{:.1f}% within 20 Mg/ha <br> {:.1f}% within 10 Mg/ha".format(p2, p1)

    mt_slider.observe(response, names="value")

    container1 = widgets.VBox([g1, g2])
    container2 = widgets.HBox([container1, g3])
    app = widgets.VBox([widgets.Label("Number of Scenes: "), mt_slider, container2])

    return app


def mapping_prediction_01(
    wd, input_csv, mask_file, param_file, out_name, agb_file=None
):
    os.chdir(wd)
    out_radar_list = pd.read_csv(input_csv)["file_name"].tolist()[:24]
    if not os.path.exists("../output"):
        os.mkdir("../output")
    mdl00 = FieldRetrieval(mask_file, out_name=out_name)
    z1, _, mask_ws1 = mdl00.inversion_setup(out_radar_list, mask_file=mask_file, n_1=0)
    z1_dim = z1.shape
    z1r = z1.reshape([-1, z1_dim[1] * z1_dim[2] * z1_dim[3]])
    z0 = mdl00.inversion_return_valid(z1r, mask_ws1, mask_file=mask_file)
    z0 = z0.reshape([-1, z1_dim[1], z1_dim[2], z1_dim[3]])

    W_mean, _ = mdl00.inversion_recursive_ws(z0, mask_file, param0_file=param_file)
    fig, ax = plt.subplots(
        nrows=np.ceil(W_mean.shape[1] / 4).astype(int),
        ncols=4,
        figsize=(12, 2.5 * np.ceil(W_mean.shape[1] / 4)),
        sharex=True,
        sharey=True,
    )
    ax1 = ax.flatten()
    in0 = gdal.Open(mask_file, gdal.GA_ReadOnly)
    for k in range(len(out_radar_list) // 2):
        basename = os.path.basename(out_radar_list[k * 2])
        agb_name = f"../output/{out_name}_predictions_{basename[:-6]}.tif"
        array0 = in0.GetRasterBand(1).ReadAsArray()
        array0[array0 > 0] = W_mean[:, k]
        try:
            os.remove(agb_name)
        except OSError:
            pass
        driver = gdal.GetDriverByName("GTiff")
        ds = driver.CreateCopy(
            agb_name,
            in0,
            0,
            ["COMPRESS=LZW", "PREDICTOR=2"],
        )
        ds.GetRasterBand(1).WriteArray(array0)
        ds.FlushCache()  # Write to disk.
        ds = None

        im = ax1[k].imshow(array0, cmap="gist_earth_r", vmin=0, vmax=150)
        ax1[k].set_title(basename[:-6])
        ax1[k].set_axis_off()

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.2, 0.03, 0.6])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel("AGB (Mg/ha)")

    in0 = None

    # return fig


# scatter 0
def scatter_plot_radar_agb(
    wd, mask_file, agb_file, sar_list, out_name, x_txt="AGB (Mg/ha)"
):
    os.chdir(wd)

    w1_slider = widgets.FloatSlider(
        value=5.0,
        min=0.0,
        max=100.0,
        step=5,
        description=f"Absolute Noise in {x_txt}: ",
        disabled=False,
        orientation="horizontal",
        readout=True,
        readout_format=".1f",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="80%"),
    )
    w2_slider = widgets.FloatSlider(
        value=5.0,
        min=0.0,
        max=100.0,
        step=5,
        description=f"Relative Noise (%) in {x_txt}: ",
        disabled=False,
        orientation="horizontal",
        readout=True,
        readout_format=".1f",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="80%"),
    )

    def update_numbers(w1_noise=10, w2_noise=10):
        if not os.path.exists("output"):
            os.mkdir("output")
        out_prefix = f"output/{out_name}"
        mdl00 = FieldRetrieval(mask_file, out_name=out_prefix)
        x0, y0, mask0 = mdl00.data_cleaner(
            sar_list,
            agb_file=agb_file,
            mask_file=mask_file,
            w1_noise=w1_noise,
            w2_noise=w2_noise,
        )
        # print(mask0)
        x0 = x0.reshape([y0.shape[0], -1, 2])
        x0v = x0[:, :, 0]
        x0h = x0[:, :, 1]

        # r2 = r2_score(
        #     scipy.stats.zscore(y0[:, None]),
        #     scipy.stats.zscore(np.mean(x0v, axis=1)[:, None]),
        # )
        # print("Variance score 1: {:.2f}".format(r2))
        # rmse = np.sqrt(
        #     mean_squared_error(
        #         scipy.stats.zscore(y0[:, None]),
        #         scipy.stats.zscore(np.mean(x0v, axis=1)[:, None]),
        #     )
        # )
        # print("RMSE: {:.5f}".format(rmse))
        # r2 = r2_score(
        #     scipy.stats.zscore(y0[:, None]),
        #     scipy.stats.zscore(np.mean(x0h, axis=1)[:, None]),
        # )
        # print("Variance score 1: {:.2f}".format(r2))
        # rmse = np.sqrt(
        #     mean_squared_error(
        #         scipy.stats.zscore(y0[:, None]),
        #         scipy.stats.zscore(np.mean(x0h, axis=1)[:, None]),
        #     )
        # )
        # print("RMSE: {:.5f}".format(rmse))

        df = pd.DataFrame(
            {
                "AGB_measured": y0.ravel(),
                "VH_mean": np.nanmean(x0v, axis=-1),
                "VV_mean": np.nanmean(x0h, axis=-1),
            }
        )
        df.to_csv(f"{out_prefix}_mean_measured.csv")
        return y0, x0v, x0h

    y0, x0v, x0h = update_numbers()

    trace1 = go.Scatter(
        x=y0,
        y=np.nanmean(x0v, axis=1),
        mode="markers",
        name="HV",
        error_y=dict(
            type="data",
            array=np.nanstd(x0v, axis=1),
            thickness=0.5,
            width=2,
        ),
        marker=dict(size=10),
    )
    trace2 = go.Scatter(
        x=y0,
        y=np.nanmean(x0h, axis=1),
        mode="markers",
        name="HH",
        error_y=dict(
            type="data",
            array=np.nanstd(x0h, axis=1),
            thickness=0.5,
            width=2,
            color="rgb(255, 127, 14)",
        ),
        marker=dict(
            size=10,
            color="rgb(255, 127, 14)",
        ),
    )
    layout1 = go.Layout(
        title=f"HV vs. {x_txt}",
        xaxis={"title": f"{x_txt}"},
        yaxis={
            "title": "Backscatter HV",
        },
        width=500,
        height=400,
        margin={"l": 60, "b": 40, "t": 60, "r": 30},
        hovermode="closest",
    )
    layout2 = go.Layout(
        title=f"HH vs. {x_txt}",
        xaxis={"title": f"{x_txt}"},
        yaxis={
            "title": "Backscatter HH",
        },
        width=500,
        height=400,
        margin={"l": 60, "b": 40, "t": 60, "r": 30},
        hovermode="closest",
    )

    g1 = go.FigureWidget(data=[trace1], layout=layout1)
    g2 = go.FigureWidget(data=[trace2], layout=layout2)
    label1 = widgets.Label(f"Number of Valid Obs. ({y0.shape[0]})")

    def response(change):
        y0, x0v, x0h = update_numbers(w1_slider.value, w2_slider.value)
        with g1.batch_update():
            g1.data[0].x = y0
            g1.data[0].y = np.nanmean(x0v, axis=1)
            g1.data[0].error_y = dict(
                type="data",
                array=np.nanstd(x0v, axis=1),
                thickness=0.5,
                width=2,
            )
        with g2.batch_update():
            g2.data[0].x = y0
            g2.data[0].y = np.nanmean(x0h, axis=1)
            g2.data[0].error_y = dict(
                type="data",
                array=np.nanstd(x0h, axis=1),
                thickness=0.5,
                width=2,
                color="rgb(255, 127, 14)",
            )
        label1.value = f"Number of Valid Obs. ({y0.shape[0]})"

    w1_slider.observe(response, names="value")
    w2_slider.observe(response, names="value")

    container4 = widgets.HBox([g1, g2])
    app = widgets.VBox([w1_slider, w2_slider, label1, container4])

    return app


# scatter clean v2
def scatter_plot_radar_agb_v2(
    wd, mask_file, agb_file, sar_list, out_name, x_txt="AGB (Mg/ha)"
):
    os.chdir(wd)

    w2_slider = widgets.FloatSlider(
        value=25.0,
        min=0.0,
        max=50.0,
        step=5,
        description=f"Relative Noise (%) in {x_txt}: ",
        disabled=False,
        orientation="horizontal",
        readout=True,
        readout_format=".1f",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="80%"),
    )

    def update_numbers(w2_noise=25):
        if not os.path.exists("output"):
            os.mkdir("output")
        out_prefix = f"output/{out_name}"
        mdl00 = FieldRetrieval(mask_file, out_name=out_prefix)
        x0, y0, mask0 = mdl00.data_cleaner_2(
            sar_list,
            agb_file=agb_file,
            mask_file=mask_file,
            w2_noise=w2_noise,
        )
        # print(mask0)
        x0 = x0.reshape([y0.shape[0], -1, 2])
        x0v = x0[:, :, 0]
        x0h = x0[:, :, 1]

        # r2 = r2_score(
        #     scipy.stats.zscore(y0[:, None]),
        #     scipy.stats.zscore(np.mean(x0v, axis=1)[:, None]),
        # )
        # print("Variance score 1: {:.2f}".format(r2))
        # rmse = np.sqrt(
        #     mean_squared_error(
        #         scipy.stats.zscore(y0[:, None]),
        #         scipy.stats.zscore(np.mean(x0v, axis=1)[:, None]),
        #     )
        # )
        # print("RMSE: {:.5f}".format(rmse))
        # r2 = r2_score(
        #     scipy.stats.zscore(y0[:, None]),
        #     scipy.stats.zscore(np.mean(x0h, axis=1)[:, None]),
        # )
        # print("Variance score 1: {:.2f}".format(r2))
        # rmse = np.sqrt(
        #     mean_squared_error(
        #         scipy.stats.zscore(y0[:, None]),
        #         scipy.stats.zscore(np.mean(x0h, axis=1)[:, None]),
        #     )
        # )
        # print("RMSE: {:.5f}".format(rmse))

        return y0, x0v, x0h

    y0, x0v, x0h = update_numbers()
    trace1 = go.Scatter(
        x=y0,
        y=np.nanmean(x0v, axis=1),
        mode="markers",
        name="VH",
        error_y=dict(
            type="data",
            array=np.nanstd(x0v, axis=1),
            thickness=0.5,
            width=2,
        ),
        marker=dict(size=10),
    )
    trace2 = go.Scatter(
        x=y0,
        y=np.nanmean(x0h, axis=1),
        mode="markers",
        name="VV",
        error_y=dict(
            type="data",
            array=np.nanstd(x0h, axis=1),
            thickness=0.5,
            width=2,
            color="rgb(255, 127, 14)",
        ),
        marker=dict(
            size=10,
            color="rgb(255, 127, 14)",
        ),
    )
    layout1 = go.Layout(
        title=f"VH vs. {x_txt}",
        xaxis={"title": f"{x_txt}"},
        yaxis={
            "title": "Backscatter VH",
        },
        width=500,
        height=400,
        margin={"l": 60, "b": 40, "t": 60, "r": 30},
        hovermode="closest",
    )
    layout2 = go.Layout(
        title=f"VV vs. {x_txt}",
        xaxis={"title": f"{x_txt}"},
        yaxis={
            "title": "Backscatter VV",
        },
        width=500,
        height=400,
        margin={"l": 60, "b": 40, "t": 60, "r": 30},
        hovermode="closest",
    )

    g1 = go.FigureWidget(data=[trace1], layout=layout1)
    g2 = go.FigureWidget(data=[trace2], layout=layout2)
    label1 = widgets.Label(f"Number of Valid Obs. ({y0.shape[0]})")

    def response(change):
        y0, x0v, x0h = update_numbers(w2_slider.value)
        with g1.batch_update():
            g1.data[0].x = y0
            g1.data[0].y = np.nanmean(x0v, axis=1)
            g1.data[0].error_y = dict(
                type="data",
                array=np.nanstd(x0v, axis=1),
                thickness=0.5,
                width=2,
            )
        with g2.batch_update():
            g2.data[0].x = y0
            g2.data[0].y = np.nanmean(x0h, axis=1)
            g2.data[0].error_y = dict(
                type="data",
                array=np.nanstd(x0h, axis=1),
                thickness=0.5,
                width=2,
                color="rgb(255, 127, 14)",
            )
        label1.value = f"Number of Valid Obs. ({y0.shape[0]})"

    w2_slider.observe(response, names="value")

    container4 = widgets.HBox([g1, g2])
    app = widgets.VBox([w2_slider, label1, container4])

    return app


def scatter_plot_radar_agb_3g(wd, W0, z0, x_txt="AGB (Mg/ha)"):
    os.chdir(wd)

    x0vh = z0[:, 0, :, 0]
    x0hh = z0[:, 0, :, 1]
    x0vv = z0[:, 0, :, 2]
    y0 = W0.ravel()
    trace1 = go.Scatter(
        x=y0,
        y=np.nanmean(x0vh, axis=1),
        mode="markers",
        name="HV",
        error_y=dict(
            type="data",
            array=np.nanstd(x0vh, axis=1),
            thickness=0.5,
            width=2,
        ),
        marker=dict(size=5),
    )
    trace2 = go.Scatter(
        x=y0,
        y=np.nanmean(x0hh, axis=1),
        mode="markers",
        name="HH",
        error_y=dict(
            type="data",
            array=np.nanstd(x0hh, axis=1),
            thickness=0.5,
            width=2,
            color="rgb(255, 127, 14)",
        ),
        marker=dict(
            size=5,
            color="rgb(255, 127, 14)",
        ),
    )
    trace3 = go.Scatter(
        x=y0,
        y=np.nanmean(x0vv, axis=1),
        mode="markers",
        name="VV",
        error_y=dict(
            type="data",
            array=np.nanstd(x0vv, axis=1),
            thickness=0.5,
            width=2,
            color="rgb(14, 168, 88)",
        ),
        marker=dict(
            size=5,
            color="rgb(14, 168, 88)",
        ),
    )
    layout1 = go.Layout(
        title=f"HV vs. {x_txt}",
        xaxis={"title": f"{x_txt}"},
        yaxis={
            "title": "Backscatter HV",
        },
        width=350,
        height=300,
        margin={"l": 60, "b": 40, "t": 60, "r": 30},
        hovermode="closest",
    )
    layout2 = go.Layout(
        title=f"HH vs. {x_txt}",
        xaxis={"title": f"{x_txt}"},
        yaxis={
            "title": "Backscatter HH",
        },
        width=350,
        height=300,
        margin={"l": 60, "b": 40, "t": 60, "r": 30},
        hovermode="closest",
    )
    layout3 = go.Layout(
        title=f"VV vs. {x_txt}",
        xaxis={"title": f"{x_txt}"},
        yaxis={
            "title": "Backscatter VV",
        },
        width=350,
        height=300,
        margin={"l": 60, "b": 40, "t": 60, "r": 30},
        hovermode="closest",
    )

    g1 = go.FigureWidget(data=[trace1], layout=layout1)
    g2 = go.FigureWidget(data=[trace2], layout=layout2)
    g3 = go.FigureWidget(data=[trace3], layout=layout3)
    label1 = widgets.Label(f"Number of Valid Obs. ({y0.shape[0]})")

    container4 = widgets.HBox([g1, g2, g3])
    app = widgets.VBox([label1, container4])

    return app


# hh/hv retrieval
def scatter_plot_agb_retrieval0(
    wd,
    mask_file,
    agb_file,
    sar_list,
    out_name,
    test_mask=None,
    x_range=[0, 200],
    x_txt="AGB (Mg/ha)",
):
    os.chdir(wd)

    if not os.path.exists("output"):
        os.mkdir("output")
    out_prefix = f"output/{out_name}"
    mdl00 = FieldRetrieval(mask_file, out_name=out_prefix)
    z1, W1, mask_ws1 = mdl00.inversion_setup(
        sar_list, agb_file=agb_file, mask_file=mask_file, n_1=0
    )
    z1_dim = z1.shape
    print(z1_dim)
    z1r = z1.reshape([-1, z1_dim[1] * z1_dim[2] * z1_dim[3]])
    z0 = mdl00.inversion_return_valid(z1r, mask_ws1, mask_file=mask_file)
    W0 = mdl00.inversion_return_valid(W1, mask_ws1, mask_file=mask_file)
    z0 = z0.reshape([-1, z1_dim[1], z1_dim[2], z1_dim[3]])
    param0_file = mdl00.params_calibration(z0, W0, out_prefix)
    W_mean, _ = mdl00.inversion_recursive_ws(z0, mask_file, param0_file=param0_file)

    y0 = W0
    m_t = W_mean.shape[1]
    y0 = y0.flatten()
    r2 = []
    rmse = []
    W_mean2 = []
    for i in range(m_t):
        # print(i)
        # for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 9, 12, 10, 11]:
        x0 = np.nanmean(W_mean[:, : i + 1], axis=1)
        r2_1 = r2_score(x0[~np.isnan(x0), None], y0[~np.isnan(x0), None])
        # print("Variance score 1: {:.2f}".format(r2_1))
        rmse_1 = np.sqrt(
            mean_squared_error(x0[~np.isnan(x0), None], y0[~np.isnan(x0), None])
        )
        # print("RMSE: {:.5f}".format(rmse_1))
        W_mean2.append(x0)
        r2.append(r2_1)
        rmse.append(rmse_1)
    r2 = np.array(r2)
    rmse = np.array(rmse)
    df = pd.DataFrame(
        {
            "AGB_measured": W0.ravel(),
            "AGB_predicted": W_mean2[-1],
            "HH_mean": np.nanmean(z0[:, 0, :, 1], axis=-1),
            "HV_mean": np.nanmean(z0[:, 0, :, 0], axis=-1),
        }
    )
    df.to_csv(f"{out_prefix}_mean_data.csv")

    trace1 = go.Scatter(
        x=np.arange(m_t) + 1,
        y=r2,
        mode="lines+markers",
        name="R<sup>2</sup>",
        marker=dict(size=10),
    )
    trace2 = go.Scatter(
        x=np.arange(m_t) + 1,
        y=rmse,
        mode="lines+markers",
        name="RMSE",
        marker=dict(size=10),
    )
    layout1 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "R<sup>2</sup>",
        },
        width=300,
        height=180,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        hovermode="closest",
    )
    layout2 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "RMSE",
        },
        width=300,
        height=180,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        hovermode="closest",
    )
    g1 = go.FigureWidget(data=[trace1], layout=layout1)
    g2 = go.FigureWidget(data=[trace2], layout=layout2)
    mt_slider = widgets.IntSlider(
        value=10,
        min=1,
        max=m_t,
        step=1,
        disabled=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="80%"),
    )

    def update_numbers(m_ti=-1):
        y = W_mean2[m_ti]
        x = y0
        xy = np.vstack([x, y])
        z0 = gaussian_kde(xy)(xy)
        p2 = np.sum((x - 20 < y) & (y < x + 20) & (x < 100)) / x[x < 100].size * 100
        p1 = np.sum((x - 10 < y) & (y < x + 10) & (x < 100)) / x[x < 100].size * 100
        return x, y, z0, p1, p2

    x, y, z0, p1, p2 = update_numbers()
    trace3 = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        name="points",
        marker=dict(
            color=z0,
            colorscale="Blues_r",
            reversescale=True,
            size=10,
            opacity=0.5,
            line=dict(width=1),
        ),
    )
    trace4 = go.Scatter(
        x=x_range,
        y=x_range,
        mode="lines",
        line=dict(width=2, color="rgb(0.8, 0.8, 0.8)"),
    )
    data = [trace3, trace4]
    layout3 = go.Layout(
        xaxis={
            "title": f"Measured {x_txt}",
            "range": x_range,
        },
        yaxis={
            "title": f"Predicted {x_txt}",
            "range": x_range,
        },
        width=400,
        height=360,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        showlegend=False,
        hovermode="closest",
        annotations=[
            dict(
                x=x_range[0] + 0.2 * (x_range[1] - x_range[0]),
                y=x_range[0] + 0.9 * (x_range[1] - x_range[0]),
                showarrow=False,
                text="{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1),
            )
        ],
    )
    g3 = go.FigureWidget(data=data, layout=layout3)

    if test_mask is None:

        def response(change):
            x, y, z0, p1, p2 = update_numbers(mt_slider.value - 1)
            with g3.batch_update():
                g3.data[0].x = x
                g3.data[0].y = y
                g3.data[0].marker = dict(
                    color=z0,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g3.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1)

        mt_slider.observe(response, names="value")
        container1 = widgets.VBox([g1, g2])
        container2 = widgets.HBox([container1, g3])
        app = widgets.VBox([widgets.Label("Number of Scenes: "), mt_slider, container2])
    else:
        mdl00 = FieldRetrieval(test_mask, out_name=f"tmp/tmp_test_{out_name}")
        z1_test, W1_test, mask_ws1_test = mdl00.inversion_setup(
            sar_list, agb_file=agb_file, mask_file=test_mask, n_1=0
        )
        z1r_test = z1_test.reshape([-1, z1_dim[1] * z1_dim[2] * z1_dim[3]])
        z0_test = mdl00.inversion_return_valid(
            z1r_test, mask_ws1_test, mask_file=test_mask
        )
        W0_test = mdl00.inversion_return_valid(
            W1_test, mask_ws1_test, mask_file=test_mask
        )
        z0_test = z0_test.reshape([-1, z1_dim[1], z1_dim[2], z1_dim[3]])
        W_mean_test, _ = mdl00.inversion_recursive_ws(
            z0_test, test_mask, param0_file=param0_file
        )
        y0_test = W0_test
        y0_test = y0_test.flatten()

        W_mean2_test = []
        for i in range(m_t):
            x0 = np.mean(W_mean_test[:, : i + 1], axis=1)
            W_mean2_test.append(x0)

        def update_numbers_test(m_ti=-1):
            y = W_mean2_test[m_ti]
            x = y0_test
            xy = np.vstack([x, y])
            z0 = gaussian_kde(xy)(xy)
            p2 = np.sum((x - 20 < y) & (y < x + 20) & (x < 100)) / x[x < 100].size * 100
            p1 = np.sum((x - 10 < y) & (y < x + 10) & (x < 100)) / x[x < 100].size * 100
            return x, y, z0, p1, p2

        x_test, y_test, z0_test, p1_test, p2_test = update_numbers_test()
        trace5 = go.Scatter(
            x=x_test,
            y=y_test,
            mode="markers",
            name="points",
            marker=dict(
                color=z0_test,
                colorscale="Blues_r",
                reversescale=True,
                size=10,
                opacity=0.5,
                line=dict(width=1),
            ),
        )
        trace6 = go.Scatter(
            x=x_range,
            y=x_range,
            mode="lines",
            line=dict(width=2, color="rgb(0.8, 0.8, 0.8)"),
        )
        data = [trace5, trace6]
        layout4 = go.Layout(
            xaxis={
                "title": f"Measured {x_txt} - Test",
                "range": x_range,
            },
            yaxis={
                "title": f"Predicted {x_txt}",
                "range": x_range,
            },
            width=400,
            height=360,
            margin={"l": 60, "b": 40, "t": 10, "r": 10},
            showlegend=False,
            hovermode="closest",
            annotations=[
                dict(
                    x=x_range[0] + 0.2 * (x_range[1] - x_range[0]),
                    y=x_range[0] + 0.9 * (x_range[1] - x_range[0]),
                    showarrow=False,
                    text="{:.1f}% within 20 <br> {:.1f}% within 10".format(
                        p2_test, p1_test
                    ),
                )
            ],
        )
        g4 = go.FigureWidget(data=data, layout=layout4)

        def response_test(change):
            x, y, z0, p1, p2 = update_numbers(mt_slider.value - 1)
            with g3.batch_update():
                g3.data[0].x = x
                g3.data[0].y = y
                g3.data[0].marker = dict(
                    color=z0,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g3.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1)
            x_test, y_test, z0_test, p1_test, p2_test = update_numbers_test(
                mt_slider.value - 1
            )
            with g4.batch_update():
                g4.data[0].x = x_test
                g4.data[0].y = y_test
                g4.data[0].marker = dict(
                    color=z0_test,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g4.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2_test, p1_test)

        mt_slider.observe(response_test, names="value")
        container1 = widgets.VBox([g1, g2])
        container2 = widgets.HBox([container1, g3, g4])
        app = widgets.VBox([widgets.Label("Number of Scenes: "), mt_slider, container2])

    return app


# sentinel retrieval (vh/vv)
def scatter_plot_agb_retrieval(
    wd,
    mask_file,
    agb_file,
    sar_list,
    out_name,
    test_mask=None,
    x_range=[0, 200],
    x_txt="AGB (Mg/ha)",
):
    os.chdir(wd)

    if not os.path.exists("output"):
        os.mkdir("output")
    out_prefix = f"output/{out_name}"
    mdl00 = FieldRetrieval(mask_file, out_name=out_prefix)
    z1, W1, mask_ws1 = mdl00.inversion_setup(
        sar_list, agb_file=agb_file, mask_file=mask_file, n_1=0
    )
    z1_dim = z1.shape
    z1r = z1.reshape([-1, z1_dim[1] * z1_dim[2] * z1_dim[3]])
    z0 = mdl00.inversion_return_valid(z1r, mask_ws1, mask_file=mask_file)
    W0 = mdl00.inversion_return_valid(W1, mask_ws1, mask_file=mask_file)
    z0 = z0.reshape([-1, z1_dim[1], z1_dim[2], z1_dim[3]])
    param0_file = mdl00.params_calibration_cband(z0, W0, out_prefix)
    W_mean, _ = mdl00.inversion_recursive_ws(z0, mask_file, param0_file=param0_file)

    y0 = W0
    m_t = W_mean.shape[1]
    y0 = y0.flatten()
    r2 = []
    rmse = []
    W_mean2 = []
    for i in range(m_t):
        # print(i)
        # for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 9, 12, 10, 11]:
        x0 = np.nanmean(W_mean[:, : i + 1], axis=1)
        r2_1 = r2_score(x0[~np.isnan(x0), None], y0[~np.isnan(x0), None])
        # print("Variance score 1: {:.2f}".format(r2_1))
        rmse_1 = np.sqrt(
            mean_squared_error(x0[~np.isnan(x0), None], y0[~np.isnan(x0), None])
        )
        # print("RMSE: {:.5f}".format(rmse_1))
        W_mean2.append(x0)
        r2.append(r2_1)
        rmse.append(rmse_1)
    r2 = np.array(r2)
    rmse = np.array(rmse)
    df = pd.DataFrame(
        {
            "AGB_measured": W0.ravel(),
            "AGB_predicted": W_mean2[-1],
            "HH_mean": np.nanmean(z0[:, 0, :, 1], axis=-1),
            "HV_mean": np.nanmean(z0[:, 0, :, 0], axis=-1),
        }
    )
    df.to_csv(f"{out_prefix}_mean_data.csv")

    trace1 = go.Scatter(
        x=np.arange(m_t) + 1,
        y=r2,
        mode="lines+markers",
        name="R<sup>2</sup>",
        marker=dict(size=10),
    )
    trace2 = go.Scatter(
        x=np.arange(m_t) + 1,
        y=rmse,
        mode="lines+markers",
        name="RMSE",
        marker=dict(size=10),
    )
    layout1 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "R<sup>2</sup>",
        },
        width=300,
        height=180,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        hovermode="closest",
    )
    layout2 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "RMSE",
        },
        width=300,
        height=180,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        hovermode="closest",
    )
    g1 = go.FigureWidget(data=[trace1], layout=layout1)
    g2 = go.FigureWidget(data=[trace2], layout=layout2)
    mt_slider = widgets.IntSlider(
        value=10,
        min=1,
        max=m_t,
        step=1,
        disabled=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="80%"),
    )

    def update_numbers(m_ti=9):
        y = W_mean2[m_ti]
        x = y0
        xy = np.vstack([x, y])
        z0 = gaussian_kde(xy)(xy)
        p2 = np.sum((x - 20 < y) & (y < x + 20) & (x < 100)) / x[x < 100].size * 100
        p1 = np.sum((x - 10 < y) & (y < x + 10) & (x < 100)) / x[x < 100].size * 100
        return x, y, z0, p1, p2

    x, y, z0, p1, p2 = update_numbers(0)
    
    trace3 = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        name="points",
        marker=dict(
            color=z0,
            colorscale="Blues_r",
            reversescale=True,
            size=10,
            opacity=0.5,
            line=dict(width=1),
        ),
    )
    trace4 = go.Scatter(
        x=x_range,
        y=x_range,
        mode="lines",
        line=dict(width=2, color="rgb(0.8, 0.8, 0.8)"),
    )
    data = [trace3, trace4]
    layout3 = go.Layout(
        xaxis={
            "title": f"Measured {x_txt}",
            "range": x_range,
        },
        yaxis={
            "title": f"Predicted {x_txt}",
            "range": x_range,
        },
        width=400,
        height=360,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        showlegend=False,
        hovermode="closest",
        annotations=[
            dict(
                x=x_range[0] + 0.2 * (x_range[1] - x_range[0]),
                y=x_range[0] + 0.9 * (x_range[1] - x_range[0]),
                showarrow=False,
                text="{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1),
            )
        ],
    )
    g3 = go.FigureWidget(data=data, layout=layout3)

    if test_mask is None:

        def response(change):
            x, y, z0, p1, p2 = update_numbers(mt_slider.value - 1)
            with g3.batch_update():
                g3.data[0].x = x
                g3.data[0].y = y
                g3.data[0].marker = dict(
                    color=z0,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g3.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1)

        mt_slider.observe(response, names="value")
        container1 = widgets.VBox([g1, g2])
        container2 = widgets.HBox([container1, g3])
        app = widgets.VBox([widgets.Label("Number of Scenes: "), mt_slider, container2])
    else:
        mdl00 = FieldRetrieval(test_mask, out_name=f"tmp/tmp_test_{out_name}")
        z1_test, W1_test, mask_ws1_test = mdl00.inversion_setup(
            sar_list, agb_file=agb_file, mask_file=test_mask, n_1=0
        )
        z1r_test = z1_test.reshape([-1, z1_dim[1] * z1_dim[2] * z1_dim[3]])
        z0_test = mdl00.inversion_return_valid(
            z1r_test, mask_ws1_test, mask_file=test_mask
        )
        W0_test = mdl00.inversion_return_valid(
            W1_test, mask_ws1_test, mask_file=test_mask
        )
        z0_test = z0_test.reshape([-1, z1_dim[1], z1_dim[2], z1_dim[3]])
        W_mean_test, _ = mdl00.inversion_recursive_ws(
            z0_test, test_mask, param0_file=param0_file
        )
        y0_test = W0_test
        y0_test = y0_test.flatten()

        W_mean2_test = []
        for i in range(m_t):
            x0 = np.mean(W_mean_test[:, : i + 1], axis=1)
            W_mean2_test.append(x0)

        def update_numbers_test(m_ti=9):
            y = W_mean2_test[m_ti]
            x = y0_test
            xy = np.vstack([x, y])
            z0 = gaussian_kde(xy)(xy)
            p2 = np.sum((x - 20 < y) & (y < x + 20) & (x < 100)) / x[x < 100].size * 100
            p1 = np.sum((x - 10 < y) & (y < x + 10) & (x < 100)) / x[x < 100].size * 100
            return x, y, z0, p1, p2

        x_test, y_test, z0_test, p1_test, p2_test = update_numbers_test()
        trace5 = go.Scatter(
            x=x_test,
            y=y_test,
            mode="markers",
            name="points",
            marker=dict(
                color=z0_test,
                colorscale="Blues_r",
                reversescale=True,
                size=10,
                opacity=0.5,
                line=dict(width=1),
            ),
        )
        trace6 = go.Scatter(
            x=x_range,
            y=x_range,
            mode="lines",
            line=dict(width=2, color="rgb(0.8, 0.8, 0.8)"),
        )
        data = [trace5, trace6]
        layout4 = go.Layout(
            xaxis={
                "title": f"Measured {x_txt} - Test",
                "range": x_range,
            },
            yaxis={
                "title": f"Predicted {x_txt}",
                "range": x_range,
            },
            width=400,
            height=360,
            margin={"l": 60, "b": 40, "t": 10, "r": 10},
            showlegend=False,
            hovermode="closest",
            annotations=[
                dict(
                    x=x_range[0] + 0.2 * (x_range[1] - x_range[0]),
                    y=x_range[0] + 0.9 * (x_range[1] - x_range[0]),
                    showarrow=False,
                    text="{:.1f}% within 20 <br> {:.1f}% within 10".format(
                        p2_test, p1_test
                    ),
                )
            ],
        )
        g4 = go.FigureWidget(data=data, layout=layout4)

        def response_test(change):
            x, y, z0, p1, p2 = update_numbers(mt_slider.value - 1)
            with g3.batch_update():
                g3.data[0].x = x
                g3.data[0].y = y
                g3.data[0].marker = dict(
                    color=z0,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g3.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1)
            x_test, y_test, z0_test, p1_test, p2_test = update_numbers_test(
                mt_slider.value - 1
            )
            with g4.batch_update():
                g4.data[0].x = x_test
                g4.data[0].y = y_test
                g4.data[0].marker = dict(
                    color=z0_test,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g4.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2_test, p1_test)

        mt_slider.observe(response_test, names="value")
        container1 = widgets.VBox([g1, g2])
        container2 = widgets.HBox([container1, g3, g4])
        app = widgets.VBox([widgets.Label("Number of Scenes: "), mt_slider, container2])

    return app


# sentinel retrieval (vh/vv)  +/-1 alpha
def scatter_plot_agb_retrieval1(
    wd,
    mask_file,
    agb_file,
    sar_list,
    out_name,
    test_mask=None,
    x_range=[0, 200],
    x_txt="AGB (Mg/ha)",
):
    os.chdir(wd)

    if not os.path.exists("output"):
        os.mkdir("output")
    out_prefix = f"output/{out_name}"
    mdl00 = FieldRetrieval(mask_file, out_name=out_prefix)
    z1, W1, mask_ws1 = mdl00.inversion_setup(
        sar_list, agb_file=agb_file, mask_file=mask_file, n_1=0
    )
    z1_dim = z1.shape
    z1r = z1.reshape([-1, z1_dim[1] * z1_dim[2] * z1_dim[3]])
    z0 = mdl00.inversion_return_valid(z1r, mask_ws1, mask_file=mask_file)
    W0 = mdl00.inversion_return_valid(W1, mask_ws1, mask_file=mask_file)
    z0 = z0.reshape([-1, z1_dim[1], z1_dim[2], z1_dim[3]])
    param0_file = mdl00.params_calibration_cband1(z0, W0, out_prefix)
    W_mean, _ = mdl00.inversion_recursive_ws(z0, mask_file, param0_file=param0_file)

    y0 = W0
    m_t = W_mean.shape[1]
    y0 = y0.flatten()
    r2 = []
    rmse = []
    W_mean2 = []
    for i in range(m_t):
        # print(i)
        # for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 9, 12, 10, 11]:
        x0 = np.nanmean(W_mean[:, : i + 1], axis=1)
        r2_1 = r2_score(x0[~np.isnan(x0), None], y0[~np.isnan(x0), None])
        # print("Variance score 1: {:.2f}".format(r2_1))
        rmse_1 = np.sqrt(
            mean_squared_error(x0[~np.isnan(x0), None], y0[~np.isnan(x0), None])
        )
        # print("RMSE: {:.5f}".format(rmse_1))
        W_mean2.append(x0)
        r2.append(r2_1)
        rmse.append(rmse_1)
    r2 = np.array(r2)
    rmse = np.array(rmse)
    df = pd.DataFrame(
        {
            "AGB_measured": W0.ravel(),
            "AGB_predicted": W_mean2[-1],
            "HH_mean": np.nanmean(z0[:, 0, :, 1], axis=-1),
            "HV_mean": np.nanmean(z0[:, 0, :, 0], axis=-1),
        }
    )
    df.to_csv(f"{out_prefix}_mean_data.csv")

    trace1 = go.Scatter(
        x=np.arange(m_t) + 1,
        y=r2,
        mode="lines+markers",
        name="R<sup>2</sup>",
        marker=dict(size=10),
    )
    trace2 = go.Scatter(
        x=np.arange(m_t) + 1,
        y=rmse,
        mode="lines+markers",
        name="RMSE",
        marker=dict(size=10),
    )
    layout1 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "R<sup>2</sup>",
        },
        width=300,
        height=180,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        hovermode="closest",
    )
    layout2 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "RMSE",
        },
        width=300,
        height=180,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        hovermode="closest",
    )
    g1 = go.FigureWidget(data=[trace1], layout=layout1)
    g2 = go.FigureWidget(data=[trace2], layout=layout2)
    mt_slider = widgets.IntSlider(
        value=10,
        min=1,
        max=m_t,
        step=1,
        disabled=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="80%"),
    )

    def update_numbers(m_ti=9):
        y = W_mean2[m_ti]
        x = y0
        xy = np.vstack([x, y])
        z0 = gaussian_kde(xy)(xy)
        p2 = np.sum((x - 20 < y) & (y < x + 20) & (x < 100)) / x[x < 100].size * 100
        p1 = np.sum((x - 10 < y) & (y < x + 10) & (x < 100)) / x[x < 100].size * 100
        return x, y, z0, p1, p2

    x, y, z0, p1, p2 = update_numbers()
    trace3 = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        name="points",
        marker=dict(
            color=z0,
            colorscale="Blues_r",
            reversescale=True,
            size=10,
            opacity=0.5,
            line=dict(width=1),
        ),
    )
    trace4 = go.Scatter(
        x=x_range,
        y=x_range,
        mode="lines",
        line=dict(width=2, color="rgb(0.8, 0.8, 0.8)"),
    )
    data = [trace3, trace4]
    layout3 = go.Layout(
        xaxis={
            "title": f"Measured {x_txt}",
            "range": x_range,
        },
        yaxis={
            "title": f"Predicted {x_txt}",
            "range": x_range,
        },
        width=400,
        height=360,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        showlegend=False,
        hovermode="closest",
        annotations=[
            dict(
                x=x_range[0] + 0.2 * (x_range[1] - x_range[0]),
                y=x_range[0] + 0.9 * (x_range[1] - x_range[0]),
                showarrow=False,
                text="{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1),
            )
        ],
    )
    g3 = go.FigureWidget(data=data, layout=layout3)

    if test_mask is None:

        def response(change):
            x, y, z0, p1, p2 = update_numbers(mt_slider.value - 1)
            with g3.batch_update():
                g3.data[0].x = x
                g3.data[0].y = y
                g3.data[0].marker = dict(
                    color=z0,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g3.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1)

        mt_slider.observe(response, names="value")
        container1 = widgets.VBox([g1, g2])
        container2 = widgets.HBox([container1, g3])
        app = widgets.VBox([widgets.Label("Number of Scenes: "), mt_slider, container2])
    else:
        mdl00 = FieldRetrieval(test_mask, out_name=f"tmp/tmp_test_{out_name}")
        z1_test, W1_test, mask_ws1_test = mdl00.inversion_setup(
            sar_list, agb_file=agb_file, mask_file=test_mask, n_1=0
        )
        z1r_test = z1_test.reshape([-1, z1_dim[1] * z1_dim[2] * z1_dim[3]])
        z0_test = mdl00.inversion_return_valid(
            z1r_test, mask_ws1_test, mask_file=test_mask
        )
        W0_test = mdl00.inversion_return_valid(
            W1_test, mask_ws1_test, mask_file=test_mask
        )
        z0_test = z0_test.reshape([-1, z1_dim[1], z1_dim[2], z1_dim[3]])
        W_mean_test, _ = mdl00.inversion_recursive_ws(
            z0_test, test_mask, param0_file=param0_file
        )
        y0_test = W0_test
        y0_test = y0_test.flatten()

        W_mean2_test = []
        for i in range(m_t):
            x0 = np.mean(W_mean_test[:, : i + 1], axis=1)
            W_mean2_test.append(x0)

        def update_numbers_test(m_ti=9):
            y = W_mean2_test[m_ti]
            x = y0_test
            xy = np.vstack([x, y])
            z0 = gaussian_kde(xy)(xy)
            p2 = np.sum((x - 20 < y) & (y < x + 20) & (x < 100)) / x[x < 100].size * 100
            p1 = np.sum((x - 10 < y) & (y < x + 10) & (x < 100)) / x[x < 100].size * 100
            return x, y, z0, p1, p2

        x_test, y_test, z0_test, p1_test, p2_test = update_numbers_test()
        trace5 = go.Scatter(
            x=x_test,
            y=y_test,
            mode="markers",
            name="points",
            marker=dict(
                color=z0_test,
                colorscale="Blues_r",
                reversescale=True,
                size=10,
                opacity=0.5,
                line=dict(width=1),
            ),
        )
        trace6 = go.Scatter(
            x=x_range,
            y=x_range,
            mode="lines",
            line=dict(width=2, color="rgb(0.8, 0.8, 0.8)"),
        )
        data = [trace5, trace6]
        layout4 = go.Layout(
            xaxis={
                "title": f"Measured {x_txt} - Test",
                "range": x_range,
            },
            yaxis={
                "title": f"Predicted {x_txt}",
                "range": x_range,
            },
            width=400,
            height=360,
            margin={"l": 60, "b": 40, "t": 10, "r": 10},
            showlegend=False,
            hovermode="closest",
            annotations=[
                dict(
                    x=x_range[0] + 0.2 * (x_range[1] - x_range[0]),
                    y=x_range[0] + 0.9 * (x_range[1] - x_range[0]),
                    showarrow=False,
                    text="{:.1f}% within 20 <br> {:.1f}% within 10".format(
                        p2_test, p1_test
                    ),
                )
            ],
        )
        g4 = go.FigureWidget(data=data, layout=layout4)

        def response_test(change):
            x, y, z0, p1, p2 = update_numbers(mt_slider.value - 1)
            with g3.batch_update():
                g3.data[0].x = x
                g3.data[0].y = y
                g3.data[0].marker = dict(
                    color=z0,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g3.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1)
            x_test, y_test, z0_test, p1_test, p2_test = update_numbers_test(
                mt_slider.value - 1
            )
            with g4.batch_update():
                g4.data[0].x = x_test
                g4.data[0].y = y_test
                g4.data[0].marker = dict(
                    color=z0_test,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g4.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2_test, p1_test)

        mt_slider.observe(response_test, names="value")
        container1 = widgets.VBox([g1, g2])
        container2 = widgets.HBox([container1, g3, g4])
        app = widgets.VBox([widgets.Label("Number of Scenes: "), mt_slider, container2])

    return app


# sentinel retrieval (vh)  +/-1 alpha
def scatter_plot_agb_retrieval1_single(
    wd,
    mask_file,
    agb_file,
    sar_list,
    out_name,
    test_mask=None,
    x_range=[0, 200],
    x_txt="AGB (Mg/ha)",
):
    os.chdir(wd)

    if not os.path.exists("output"):
        os.mkdir("output")
    out_prefix = f"output/{out_name}"
    mdl00 = FieldRetrieval(mask_file, out_name=out_prefix)
    z1, W1, mask_ws1 = mdl00.inversion_setup(
        sar_list, agb_file=agb_file, mask_file=mask_file, n_1=0
    )
    z1_dim = z1.shape
    z1r = z1.reshape([-1, z1_dim[1] * z1_dim[2] * z1_dim[3]])
    z0 = mdl00.inversion_return_valid(z1r, mask_ws1, mask_file=mask_file)
    W0 = mdl00.inversion_return_valid(W1, mask_ws1, mask_file=mask_file)
    z0 = z0.reshape([-1, z1_dim[1], z1_dim[2], z1_dim[3]])
    param0_file = mdl00.params_calibration_cband_single(z0, W0, out_prefix)
    W_mean = mdl00.inversion_recursive_ws_single(z0, mask_file, param0_file=param0_file)

    y0 = W0
    m_t = W_mean.shape[1]
    y0 = y0.flatten()
    r2 = []
    rmse = []
    W_mean2 = []
    for i in range(m_t):
        # print(i)
        # for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 9, 12, 10, 11]:
        x0 = np.nanmean(W_mean[:, : i + 1], axis=1)
        r2_1 = r2_score(x0[~np.isnan(x0), None], y0[~np.isnan(x0), None])
        # print("Variance score 1: {:.2f}".format(r2_1))
        rmse_1 = np.sqrt(
            mean_squared_error(x0[~np.isnan(x0), None], y0[~np.isnan(x0), None])
        )
        # print("RMSE: {:.5f}".format(rmse_1))
        W_mean2.append(x0)
        r2.append(r2_1)
        rmse.append(rmse_1)
    r2 = np.array(r2)
    rmse = np.array(rmse)
    df = pd.DataFrame(
        {
            "AGB_measured": W0.ravel(),
            "AGB_predicted": W_mean2[-1],
            "HH_mean": np.nanmean(z0[:, 0, :, 1], axis=-1),
            "HV_mean": np.nanmean(z0[:, 0, :, 0], axis=-1),
        }
    )
    df.to_csv(f"{out_prefix}_mean_data.csv")

    trace1 = go.Scatter(
        x=np.arange(m_t) + 1,
        y=r2,
        mode="lines+markers",
        name="R<sup>2</sup>",
        marker=dict(size=10),
    )
    trace2 = go.Scatter(
        x=np.arange(m_t) + 1,
        y=rmse,
        mode="lines+markers",
        name="RMSE",
        marker=dict(size=10),
    )
    layout1 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "R<sup>2</sup>",
        },
        width=300,
        height=180,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        hovermode="closest",
    )
    layout2 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "RMSE",
        },
        width=300,
        height=180,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        hovermode="closest",
    )
    g1 = go.FigureWidget(data=[trace1], layout=layout1)
    g2 = go.FigureWidget(data=[trace2], layout=layout2)
    mt_slider = widgets.IntSlider(
        value=10,
        min=1,
        max=m_t,
        step=1,
        disabled=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="80%"),
    )

    def update_numbers(m_ti=9):
        y = W_mean2[m_ti]
        x = y0
        xy = np.vstack([x, y])
        z0 = gaussian_kde(xy)(xy)
        p2 = np.sum((x - 20 < y) & (y < x + 20) & (x < 100)) / x[x < 100].size * 100
        p1 = np.sum((x - 10 < y) & (y < x + 10) & (x < 100)) / x[x < 100].size * 100
        return x, y, z0, p1, p2

    x, y, z0, p1, p2 = update_numbers()
    trace3 = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        name="points",
        marker=dict(
            color=z0,
            colorscale="Blues_r",
            reversescale=True,
            size=10,
            opacity=0.5,
            line=dict(width=1),
        ),
    )
    trace4 = go.Scatter(
        x=x_range,
        y=x_range,
        mode="lines",
        line=dict(width=2, color="rgb(0.8, 0.8, 0.8)"),
    )
    data = [trace3, trace4]
    layout3 = go.Layout(
        xaxis={
            "title": f"Measured {x_txt}",
            "range": x_range,
        },
        yaxis={
            "title": f"Predicted {x_txt}",
            "range": x_range,
        },
        width=400,
        height=360,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        showlegend=False,
        hovermode="closest",
        annotations=[
            dict(
                x=x_range[0] + 0.2 * (x_range[1] - x_range[0]),
                y=x_range[0] + 0.9 * (x_range[1] - x_range[0]),
                showarrow=False,
                text="{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1),
            )
        ],
    )
    g3 = go.FigureWidget(data=data, layout=layout3)

    if test_mask is None:

        def response(change):
            x, y, z0, p1, p2 = update_numbers(mt_slider.value - 1)
            with g3.batch_update():
                g3.data[0].x = x
                g3.data[0].y = y
                g3.data[0].marker = dict(
                    color=z0,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g3.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1)

        mt_slider.observe(response, names="value")
        container1 = widgets.VBox([g1, g2])
        container2 = widgets.HBox([container1, g3])
        app = widgets.VBox([widgets.Label("Number of Scenes: "), mt_slider, container2])
    else:
        mdl00 = FieldRetrieval(test_mask, out_name=f"tmp/tmp_test_{out_name}")
        z1_test, W1_test, mask_ws1_test = mdl00.inversion_setup(
            sar_list, agb_file=agb_file, mask_file=test_mask, n_1=0
        )
        z1r_test = z1_test.reshape([-1, z1_dim[1] * z1_dim[2] * z1_dim[3]])
        z0_test = mdl00.inversion_return_valid(
            z1r_test, mask_ws1_test, mask_file=test_mask
        )
        W0_test = mdl00.inversion_return_valid(
            W1_test, mask_ws1_test, mask_file=test_mask
        )
        z0_test = z0_test.reshape([-1, z1_dim[1], z1_dim[2], z1_dim[3]])
        W_mean_test, _ = mdl00.inversion_recursive_ws_single(
            z0_test, test_mask, param0_file=param0_file
        )
        y0_test = W0_test
        y0_test = y0_test.flatten()

        W_mean2_test = []
        for i in range(m_t):
            x0 = np.mean(W_mean_test[:, : i + 1], axis=1)
            W_mean2_test.append(x0)

        def update_numbers_test(m_ti=9):
            y = W_mean2_test[m_ti]
            x = y0_test
            xy = np.vstack([x, y])
            z0 = gaussian_kde(xy)(xy)
            p2 = np.sum((x - 20 < y) & (y < x + 20) & (x < 100)) / x[x < 100].size * 100
            p1 = np.sum((x - 10 < y) & (y < x + 10) & (x < 100)) / x[x < 100].size * 100
            return x, y, z0, p1, p2

        x_test, y_test, z0_test, p1_test, p2_test = update_numbers_test()
        trace5 = go.Scatter(
            x=x_test,
            y=y_test,
            mode="markers",
            name="points",
            marker=dict(
                color=z0_test,
                colorscale="Blues_r",
                reversescale=True,
                size=10,
                opacity=0.5,
                line=dict(width=1),
            ),
        )
        trace6 = go.Scatter(
            x=x_range,
            y=x_range,
            mode="lines",
            line=dict(width=2, color="rgb(0.8, 0.8, 0.8)"),
        )
        data = [trace5, trace6]
        layout4 = go.Layout(
            xaxis={
                "title": f"Measured {x_txt} - Test",
                "range": x_range,
            },
            yaxis={
                "title": f"Predicted {x_txt}",
                "range": x_range,
            },
            width=400,
            height=360,
            margin={"l": 60, "b": 40, "t": 10, "r": 10},
            showlegend=False,
            hovermode="closest",
            annotations=[
                dict(
                    x=x_range[0] + 0.2 * (x_range[1] - x_range[0]),
                    y=x_range[0] + 0.9 * (x_range[1] - x_range[0]),
                    showarrow=False,
                    text="{:.1f}% within 20 <br> {:.1f}% within 10".format(
                        p2_test, p1_test
                    ),
                )
            ],
        )
        g4 = go.FigureWidget(data=data, layout=layout4)

        def response_test(change):
            x, y, z0, p1, p2 = update_numbers(mt_slider.value - 1)
            with g3.batch_update():
                g3.data[0].x = x
                g3.data[0].y = y
                g3.data[0].marker = dict(
                    color=z0,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g3.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1)
            x_test, y_test, z0_test, p1_test, p2_test = update_numbers_test(
                mt_slider.value - 1
            )
            with g4.batch_update():
                g4.data[0].x = x_test
                g4.data[0].y = y_test
                g4.data[0].marker = dict(
                    color=z0_test,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g4.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2_test, p1_test)

        mt_slider.observe(response_test, names="value")
        container1 = widgets.VBox([g1, g2])
        container2 = widgets.HBox([container1, g3, g4])
        app = widgets.VBox([widgets.Label("Number of Scenes: "), mt_slider, container2])

    return app


# sentinel retrieval (vh)  no interactive term
def scatter_plot_agb_retrieval_04a_noint(
    wd,
    mask_file,
    agb_file,
    sar_list,
    out_name,
    test_mask=None,
    x_range=[0, 200],
    x_txt="AGB (Mg/ha)",
):
    os.chdir(wd)

    if not os.path.exists("output"):
        os.mkdir("output")
    out_prefix = f"output/{out_name}"
    mdl00 = FieldRetrieval(mask_file, out_name=out_prefix)
    z1, W1, mask_ws1 = mdl00.inversion_setup(
        sar_list, agb_file=agb_file, mask_file=mask_file, n_1=0
    )
    z1_dim = z1.shape
    z1r = z1.reshape([-1, z1_dim[1] * z1_dim[2] * z1_dim[3]])
    z0 = mdl00.inversion_return_valid(z1r, mask_ws1, mask_file=mask_file)
    W0 = mdl00.inversion_return_valid(W1, mask_ws1, mask_file=mask_file)
    z0 = z0.reshape([-1, z1_dim[1], z1_dim[2], z1_dim[3]])
    param0_file = mdl00.params_calibration_cband_04a_noint(z0, W0, out_prefix)
    W_mean, _ = mdl00.inversion_recursive_ws_04a_noint(
        z0, mask_file, param0_file=param0_file
    )

    y0 = W0
    m_t = W_mean.shape[1]
    y0 = y0.flatten()
    r2 = []
    rmse = []
    W_mean2 = []
    for i in range(m_t):
        # print(i)
        # for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 9, 12, 10, 11]:
        x0 = np.nanmean(W_mean[:, : i + 1], axis=1)
        r2_1 = r2_score(x0[~np.isnan(x0), None], y0[~np.isnan(x0), None])
        # print("Variance score 1: {:.2f}".format(r2_1))
        rmse_1 = np.sqrt(
            mean_squared_error(x0[~np.isnan(x0), None], y0[~np.isnan(x0), None])
        )
        # print("RMSE: {:.5f}".format(rmse_1))
        W_mean2.append(x0)
        r2.append(r2_1)
        rmse.append(rmse_1)
    r2 = np.array(r2)
    rmse = np.array(rmse)
    df = pd.DataFrame(
        {
            "AGB_measured": W0.ravel(),
            "AGB_predicted": W_mean2[-1],
            "HH_mean": np.nanmean(z0[:, 0, :, 1], axis=-1),
            "HV_mean": np.nanmean(z0[:, 0, :, 0], axis=-1),
        }
    )
    df.to_csv(f"{out_prefix}_mean_data.csv")

    trace1 = go.Scatter(
        x=np.arange(m_t) + 1,
        y=r2,
        mode="lines+markers",
        name="R<sup>2</sup>",
        marker=dict(size=10),
    )
    trace2 = go.Scatter(
        x=np.arange(m_t) + 1,
        y=rmse,
        mode="lines+markers",
        name="RMSE",
        marker=dict(size=10),
    )
    layout1 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "R<sup>2</sup>",
        },
        width=300,
        height=180,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        hovermode="closest",
    )
    layout2 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "RMSE",
        },
        width=300,
        height=180,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        hovermode="closest",
    )
    g1 = go.FigureWidget(data=[trace1], layout=layout1)
    g2 = go.FigureWidget(data=[trace2], layout=layout2)
    mt_slider = widgets.IntSlider(
        value=10,
        min=1,
        max=m_t,
        step=1,
        disabled=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="80%"),
    )

    def update_numbers(m_ti=9):
        y = W_mean2[m_ti]
        x = y0
        xy = np.vstack([x, y])
        z0 = gaussian_kde(xy)(xy)
        p2 = np.sum((x - 20 < y) & (y < x + 20) & (x < 100)) / x[x < 100].size * 100
        p1 = np.sum((x - 10 < y) & (y < x + 10) & (x < 100)) / x[x < 100].size * 100
        return x, y, z0, p1, p2

    x, y, z0, p1, p2 = update_numbers()
    trace3 = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        name="points",
        marker=dict(
            color=z0,
            colorscale="Blues_r",
            reversescale=True,
            size=10,
            opacity=0.5,
            line=dict(width=1),
        ),
    )
    trace4 = go.Scatter(
        x=x_range,
        y=x_range,
        mode="lines",
        line=dict(width=2, color="rgb(0.8, 0.8, 0.8)"),
    )
    data = [trace3, trace4]
    layout3 = go.Layout(
        xaxis={
            "title": f"Measured {x_txt}",
            "range": x_range,
        },
        yaxis={
            "title": f"Predicted {x_txt}",
            "range": x_range,
        },
        width=400,
        height=360,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        showlegend=False,
        hovermode="closest",
        annotations=[
            dict(
                x=x_range[0] + 0.2 * (x_range[1] - x_range[0]),
                y=x_range[0] + 0.9 * (x_range[1] - x_range[0]),
                showarrow=False,
                text="{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1),
            )
        ],
    )
    g3 = go.FigureWidget(data=data, layout=layout3)

    if test_mask is None:

        def response(change):
            x, y, z0, p1, p2 = update_numbers(mt_slider.value - 1)
            with g3.batch_update():
                g3.data[0].x = x
                g3.data[0].y = y
                g3.data[0].marker = dict(
                    color=z0,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g3.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1)

        mt_slider.observe(response, names="value")
        container1 = widgets.VBox([g1, g2])
        container2 = widgets.HBox([container1, g3])
        app = widgets.VBox([widgets.Label("Number of Scenes: "), mt_slider, container2])
    else:
        mdl00 = FieldRetrieval(test_mask, out_name=f"tmp/tmp_test_{out_name}")
        z1_test, W1_test, mask_ws1_test = mdl00.inversion_setup(
            sar_list, agb_file=agb_file, mask_file=test_mask, n_1=0
        )
        z1r_test = z1_test.reshape([-1, z1_dim[1] * z1_dim[2] * z1_dim[3]])
        z0_test = mdl00.inversion_return_valid(
            z1r_test, mask_ws1_test, mask_file=test_mask
        )
        W0_test = mdl00.inversion_return_valid(
            W1_test, mask_ws1_test, mask_file=test_mask
        )
        z0_test = z0_test.reshape([-1, z1_dim[1], z1_dim[2], z1_dim[3]])
        W_mean_test, _ = mdl00.inversion_recursive_ws_04a_noint(
            z0_test, test_mask, param0_file=param0_file
        )
        y0_test = W0_test
        y0_test = y0_test.flatten()

        W_mean2_test = []
        for i in range(m_t):
            x0 = np.mean(W_mean_test[:, : i + 1], axis=1)
            W_mean2_test.append(x0)

        def update_numbers_test(m_ti=9):
            y = W_mean2_test[m_ti]
            x = y0_test
            xy = np.vstack([x, y])
            z0 = gaussian_kde(xy)(xy)
            p2 = np.sum((x - 20 < y) & (y < x + 20) & (x < 100)) / x[x < 100].size * 100
            p1 = np.sum((x - 10 < y) & (y < x + 10) & (x < 100)) / x[x < 100].size * 100
            return x, y, z0, p1, p2

        x_test, y_test, z0_test, p1_test, p2_test = update_numbers_test()
        trace5 = go.Scatter(
            x=x_test,
            y=y_test,
            mode="markers",
            name="points",
            marker=dict(
                color=z0_test,
                colorscale="Blues_r",
                reversescale=True,
                size=10,
                opacity=0.5,
                line=dict(width=1),
            ),
        )
        trace6 = go.Scatter(
            x=x_range,
            y=x_range,
            mode="lines",
            line=dict(width=2, color="rgb(0.8, 0.8, 0.8)"),
        )
        data = [trace5, trace6]
        layout4 = go.Layout(
            xaxis={
                "title": f"Measured {x_txt} - Test",
                "range": x_range,
            },
            yaxis={
                "title": f"Predicted {x_txt}",
                "range": x_range,
            },
            width=400,
            height=360,
            margin={"l": 60, "b": 40, "t": 10, "r": 10},
            showlegend=False,
            hovermode="closest",
            annotations=[
                dict(
                    x=x_range[0] + 0.2 * (x_range[1] - x_range[0]),
                    y=x_range[0] + 0.9 * (x_range[1] - x_range[0]),
                    showarrow=False,
                    text="{:.1f}% within 20 <br> {:.1f}% within 10".format(
                        p2_test, p1_test
                    ),
                )
            ],
        )
        g4 = go.FigureWidget(data=data, layout=layout4)

        def response_test(change):
            x, y, z0, p1, p2 = update_numbers(mt_slider.value - 1)
            with g3.batch_update():
                g3.data[0].x = x
                g3.data[0].y = y
                g3.data[0].marker = dict(
                    color=z0,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g3.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1)
            x_test, y_test, z0_test, p1_test, p2_test = update_numbers_test(
                mt_slider.value - 1
            )
            with g4.batch_update():
                g4.data[0].x = x_test
                g4.data[0].y = y_test
                g4.data[0].marker = dict(
                    color=z0_test,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g4.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2_test, p1_test)

        mt_slider.observe(response_test, names="value")
        container1 = widgets.VBox([g1, g2])
        container2 = widgets.HBox([container1, g3, g4])
        app = widgets.VBox([widgets.Label("Number of Scenes: "), mt_slider, container2])

    return app


def scatter_plot_agb_retrieval_04a_noint_nos(
    wd,
    mask_file,
    agb_file,
    sar_list,
    out_name,
    test_mask=None,
    x_range=[0, 200],
    x_txt="AGB (Mg/ha)",
):
    os.chdir(wd)

    if not os.path.exists("output"):
        os.mkdir("output")
    out_prefix = f"output/{out_name}"
    mdl00 = FieldRetrieval(mask_file, out_name=out_prefix)
    z1, W1, mask_ws1 = mdl00.inversion_setup(
        sar_list, agb_file=agb_file, mask_file=mask_file, n_1=0
    )
    z1_dim = z1.shape
    z1r = z1.reshape([-1, z1_dim[1] * z1_dim[2] * z1_dim[3]])
    z0 = mdl00.inversion_return_valid(z1r, mask_ws1, mask_file=mask_file)
    W0 = mdl00.inversion_return_valid(W1, mask_ws1, mask_file=mask_file)
    z0 = z0.reshape([-1, z1_dim[1], z1_dim[2], z1_dim[3]])
    param0_file = mdl00.params_calibration_cband_04a_noint_nos(z0, W0, out_prefix)
    W_mean, _ = mdl00.inversion_recursive_ws_04a_noint(
        z0, mask_file, param0_file=param0_file
    )

    y0 = W0
    m_t = W_mean.shape[1]
    y0 = y0.flatten()
    r2 = []
    rmse = []
    W_mean2 = []
    for i in range(m_t):
        # print(i)
        # for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 9, 12, 10, 11]:
        x0 = np.nanmean(W_mean[:, : i + 1], axis=1)
        r2_1 = r2_score(x0[~np.isnan(x0), None], y0[~np.isnan(x0), None])
        # print("Variance score 1: {:.2f}".format(r2_1))
        rmse_1 = np.sqrt(
            mean_squared_error(x0[~np.isnan(x0), None], y0[~np.isnan(x0), None])
        )
        # print("RMSE: {:.5f}".format(rmse_1))
        W_mean2.append(x0)
        r2.append(r2_1)
        rmse.append(rmse_1)
    r2 = np.array(r2)
    rmse = np.array(rmse)
    df = pd.DataFrame(
        {
            "AGB_measured": W0.ravel(),
            "AGB_predicted": W_mean2[-1],
            "HH_mean": np.nanmean(z0[:, 0, :, 1], axis=-1),
            "HV_mean": np.nanmean(z0[:, 0, :, 0], axis=-1),
        }
    )
    df.to_csv(f"{out_prefix}_mean_data.csv")

    trace1 = go.Scatter(
        x=np.arange(m_t) + 1,
        y=r2,
        mode="lines+markers",
        name="R<sup>2</sup>",
        marker=dict(size=10),
    )
    trace2 = go.Scatter(
        x=np.arange(m_t) + 1,
        y=rmse,
        mode="lines+markers",
        name="RMSE",
        marker=dict(size=10),
    )
    layout1 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "R<sup>2</sup>",
        },
        width=300,
        height=180,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        hovermode="closest",
    )
    layout2 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "RMSE",
        },
        width=300,
        height=180,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        hovermode="closest",
    )
    g1 = go.FigureWidget(data=[trace1], layout=layout1)
    g2 = go.FigureWidget(data=[trace2], layout=layout2)
    mt_slider = widgets.IntSlider(
        value=10,
        min=1,
        max=m_t,
        step=1,
        disabled=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="80%"),
    )

    def update_numbers(m_ti=9):
        y = W_mean2[m_ti]
        x = y0
        xy = np.vstack([x, y])
        z0 = gaussian_kde(xy)(xy)
        p2 = np.sum((x - 20 < y) & (y < x + 20) & (x < 100)) / x[x < 100].size * 100
        p1 = np.sum((x - 10 < y) & (y < x + 10) & (x < 100)) / x[x < 100].size * 100
        return x, y, z0, p1, p2

    x, y, z0, p1, p2 = update_numbers()
    trace3 = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        name="points",
        marker=dict(
            color=z0,
            colorscale="Blues_r",
            reversescale=True,
            size=10,
            opacity=0.5,
            line=dict(width=1),
        ),
    )
    trace4 = go.Scatter(
        x=x_range,
        y=x_range,
        mode="lines",
        line=dict(width=2, color="rgb(0.8, 0.8, 0.8)"),
    )
    data = [trace3, trace4]
    layout3 = go.Layout(
        xaxis={
            "title": f"Measured {x_txt}",
            "range": x_range,
        },
        yaxis={
            "title": f"Predicted {x_txt}",
            "range": x_range,
        },
        width=400,
        height=360,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        showlegend=False,
        hovermode="closest",
        annotations=[
            dict(
                x=x_range[0] + 0.2 * (x_range[1] - x_range[0]),
                y=x_range[0] + 0.9 * (x_range[1] - x_range[0]),
                showarrow=False,
                text="{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1),
            )
        ],
    )
    g3 = go.FigureWidget(data=data, layout=layout3)

    if test_mask is None:

        def response(change):
            x, y, z0, p1, p2 = update_numbers(mt_slider.value - 1)
            with g3.batch_update():
                g3.data[0].x = x
                g3.data[0].y = y
                g3.data[0].marker = dict(
                    color=z0,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g3.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1)

        mt_slider.observe(response, names="value")
        container1 = widgets.VBox([g1, g2])
        container2 = widgets.HBox([container1, g3])
        app = widgets.VBox([widgets.Label("Number of Scenes: "), mt_slider, container2])
    else:
        mdl00 = FieldRetrieval(test_mask, out_name=f"tmp/tmp_test_{out_name}")
        z1_test, W1_test, mask_ws1_test = mdl00.inversion_setup(
            sar_list, agb_file=agb_file, mask_file=test_mask, n_1=0
        )
        z1r_test = z1_test.reshape([-1, z1_dim[1] * z1_dim[2] * z1_dim[3]])
        z0_test = mdl00.inversion_return_valid(
            z1r_test, mask_ws1_test, mask_file=test_mask
        )
        W0_test = mdl00.inversion_return_valid(
            W1_test, mask_ws1_test, mask_file=test_mask
        )
        z0_test = z0_test.reshape([-1, z1_dim[1], z1_dim[2], z1_dim[3]])
        W_mean_test, _ = mdl00.inversion_recursive_ws_04a_noint(
            z0_test, test_mask, param0_file=param0_file
        )
        y0_test = W0_test
        y0_test = y0_test.flatten()

        W_mean2_test = []
        for i in range(m_t):
            x0 = np.mean(W_mean_test[:, : i + 1], axis=1)
            W_mean2_test.append(x0)

        def update_numbers_test(m_ti=9):
            y = W_mean2_test[m_ti]
            x = y0_test
            xy = np.vstack([x, y])
            z0 = gaussian_kde(xy)(xy)
            p2 = np.sum((x - 20 < y) & (y < x + 20) & (x < 100)) / x[x < 100].size * 100
            p1 = np.sum((x - 10 < y) & (y < x + 10) & (x < 100)) / x[x < 100].size * 100
            return x, y, z0, p1, p2

        x_test, y_test, z0_test, p1_test, p2_test = update_numbers_test()
        trace5 = go.Scatter(
            x=x_test,
            y=y_test,
            mode="markers",
            name="points",
            marker=dict(
                color=z0_test,
                colorscale="Blues_r",
                reversescale=True,
                size=10,
                opacity=0.5,
                line=dict(width=1),
            ),
        )
        trace6 = go.Scatter(
            x=x_range,
            y=x_range,
            mode="lines",
            line=dict(width=2, color="rgb(0.8, 0.8, 0.8)"),
        )
        data = [trace5, trace6]
        layout4 = go.Layout(
            xaxis={
                "title": f"Measured {x_txt} - Test",
                "range": x_range,
            },
            yaxis={
                "title": f"Predicted {x_txt}",
                "range": x_range,
            },
            width=400,
            height=360,
            margin={"l": 60, "b": 40, "t": 10, "r": 10},
            showlegend=False,
            hovermode="closest",
            annotations=[
                dict(
                    x=x_range[0] + 0.2 * (x_range[1] - x_range[0]),
                    y=x_range[0] + 0.9 * (x_range[1] - x_range[0]),
                    showarrow=False,
                    text="{:.1f}% within 20 <br> {:.1f}% within 10".format(
                        p2_test, p1_test
                    ),
                )
            ],
        )
        g4 = go.FigureWidget(data=data, layout=layout4)

        def response_test(change):
            x, y, z0, p1, p2 = update_numbers(mt_slider.value - 1)
            with g3.batch_update():
                g3.data[0].x = x
                g3.data[0].y = y
                g3.data[0].marker = dict(
                    color=z0,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g3.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1)
            x_test, y_test, z0_test, p1_test, p2_test = update_numbers_test(
                mt_slider.value - 1
            )
            with g4.batch_update():
                g4.data[0].x = x_test
                g4.data[0].y = y_test
                g4.data[0].marker = dict(
                    color=z0_test,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g4.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2_test, p1_test)

        mt_slider.observe(response_test, names="value")
        container1 = widgets.VBox([g1, g2])
        container2 = widgets.HBox([container1, g3, g4])
        app = widgets.VBox([widgets.Label("Number of Scenes: "), mt_slider, container2])

    return app


def scatter_plot_agb_retrieval_04a_noint_noc(
    wd,
    mask_file,
    agb_file,
    sar_list,
    out_name,
    test_mask=None,
    x_range=[0, 200],
    x_txt="AGB (Mg/ha)",
):
    os.chdir(wd)

    if not os.path.exists("output"):
        os.mkdir("output")
    out_prefix = f"output/{out_name}"
    mdl00 = FieldRetrieval(mask_file, out_name=out_prefix)
    z1, W1, mask_ws1 = mdl00.inversion_setup(
        sar_list, agb_file=agb_file, mask_file=mask_file, n_1=0
    )
    z1_dim = z1.shape
    z1r = z1.reshape([-1, z1_dim[1] * z1_dim[2] * z1_dim[3]])
    z0 = mdl00.inversion_return_valid(z1r, mask_ws1, mask_file=mask_file)
    W0 = mdl00.inversion_return_valid(W1, mask_ws1, mask_file=mask_file)
    z0 = z0.reshape([-1, z1_dim[1], z1_dim[2], z1_dim[3]])
    param0_file = mdl00.params_calibration_cband_04a_noint_noc(z0, W0, out_prefix)
    W_mean, _ = mdl00.inversion_recursive_ws_04a_noint(
        z0, mask_file, param0_file=param0_file
    )

    y0 = W0
    m_t = W_mean.shape[1]
    y0 = y0.flatten()
    r2 = []
    rmse = []
    W_mean2 = []
    for i in range(m_t):
        # print(i)
        # for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 9, 12, 10, 11]:
        x0 = np.nanmean(W_mean[:, : i + 1], axis=1)
        r2_1 = r2_score(x0[~np.isnan(x0), None], y0[~np.isnan(x0), None])
        # print("Variance score 1: {:.2f}".format(r2_1))
        rmse_1 = np.sqrt(
            mean_squared_error(x0[~np.isnan(x0), None], y0[~np.isnan(x0), None])
        )
        # print("RMSE: {:.5f}".format(rmse_1))
        W_mean2.append(x0)
        r2.append(r2_1)
        rmse.append(rmse_1)
    r2 = np.array(r2)
    rmse = np.array(rmse)
    df = pd.DataFrame(
        {
            "AGB_measured": W0.ravel(),
            "AGB_predicted": W_mean2[-1],
            "HH_mean": np.nanmean(z0[:, 0, :, 1], axis=-1),
            "HV_mean": np.nanmean(z0[:, 0, :, 0], axis=-1),
        }
    )
    df.to_csv(f"{out_prefix}_mean_data.csv")

    trace1 = go.Scatter(
        x=np.arange(m_t) + 1,
        y=r2,
        mode="lines+markers",
        name="R<sup>2</sup>",
        marker=dict(size=10),
    )
    trace2 = go.Scatter(
        x=np.arange(m_t) + 1,
        y=rmse,
        mode="lines+markers",
        name="RMSE",
        marker=dict(size=10),
    )
    layout1 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "R<sup>2</sup>",
        },
        width=300,
        height=180,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        hovermode="closest",
    )
    layout2 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "RMSE",
        },
        width=300,
        height=180,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        hovermode="closest",
    )
    g1 = go.FigureWidget(data=[trace1], layout=layout1)
    g2 = go.FigureWidget(data=[trace2], layout=layout2)
    mt_slider = widgets.IntSlider(
        value=10,
        min=1,
        max=m_t,
        step=1,
        disabled=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="80%"),
    )

    def update_numbers(m_ti=9):
        y = W_mean2[m_ti]
        x = y0
        xy = np.vstack([x, y])
        z0 = gaussian_kde(xy)(xy)
        p2 = np.sum((x - 20 < y) & (y < x + 20) & (x < 100)) / x[x < 100].size * 100
        p1 = np.sum((x - 10 < y) & (y < x + 10) & (x < 100)) / x[x < 100].size * 100
        return x, y, z0, p1, p2

    x, y, z0, p1, p2 = update_numbers()
    trace3 = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        name="points",
        marker=dict(
            color=z0,
            colorscale="Blues_r",
            reversescale=True,
            size=10,
            opacity=0.5,
            line=dict(width=1),
        ),
    )
    trace4 = go.Scatter(
        x=x_range,
        y=x_range,
        mode="lines",
        line=dict(width=2, color="rgb(0.8, 0.8, 0.8)"),
    )
    data = [trace3, trace4]
    layout3 = go.Layout(
        xaxis={
            "title": f"Measured {x_txt}",
            "range": x_range,
        },
        yaxis={
            "title": f"Predicted {x_txt}",
            "range": x_range,
        },
        width=400,
        height=360,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        showlegend=False,
        hovermode="closest",
        annotations=[
            dict(
                x=x_range[0] + 0.2 * (x_range[1] - x_range[0]),
                y=x_range[0] + 0.9 * (x_range[1] - x_range[0]),
                showarrow=False,
                text="{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1),
            )
        ],
    )
    g3 = go.FigureWidget(data=data, layout=layout3)

    if test_mask is None:

        def response(change):
            x, y, z0, p1, p2 = update_numbers(mt_slider.value - 1)
            with g3.batch_update():
                g3.data[0].x = x
                g3.data[0].y = y
                g3.data[0].marker = dict(
                    color=z0,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g3.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1)

        mt_slider.observe(response, names="value")
        container1 = widgets.VBox([g1, g2])
        container2 = widgets.HBox([container1, g3])
        app = widgets.VBox([widgets.Label("Number of Scenes: "), mt_slider, container2])
    else:
        mdl00 = FieldRetrieval(test_mask, out_name=f"tmp/tmp_test_{out_name}")
        z1_test, W1_test, mask_ws1_test = mdl00.inversion_setup(
            sar_list, agb_file=agb_file, mask_file=test_mask, n_1=0
        )
        z1r_test = z1_test.reshape([-1, z1_dim[1] * z1_dim[2] * z1_dim[3]])
        z0_test = mdl00.inversion_return_valid(
            z1r_test, mask_ws1_test, mask_file=test_mask
        )
        W0_test = mdl00.inversion_return_valid(
            W1_test, mask_ws1_test, mask_file=test_mask
        )
        z0_test = z0_test.reshape([-1, z1_dim[1], z1_dim[2], z1_dim[3]])
        W_mean_test, _ = mdl00.inversion_recursive_ws_04a_noint(
            z0_test, test_mask, param0_file=param0_file
        )
        y0_test = W0_test
        y0_test = y0_test.flatten()

        W_mean2_test = []
        for i in range(m_t):
            x0 = np.mean(W_mean_test[:, : i + 1], axis=1)
            W_mean2_test.append(x0)

        def update_numbers_test(m_ti=9):
            y = W_mean2_test[m_ti]
            x = y0_test
            xy = np.vstack([x, y])
            z0 = gaussian_kde(xy)(xy)
            p2 = np.sum((x - 20 < y) & (y < x + 20) & (x < 100)) / x[x < 100].size * 100
            p1 = np.sum((x - 10 < y) & (y < x + 10) & (x < 100)) / x[x < 100].size * 100
            return x, y, z0, p1, p2

        x_test, y_test, z0_test, p1_test, p2_test = update_numbers_test()
        trace5 = go.Scatter(
            x=x_test,
            y=y_test,
            mode="markers",
            name="points",
            marker=dict(
                color=z0_test,
                colorscale="Blues_r",
                reversescale=True,
                size=10,
                opacity=0.5,
                line=dict(width=1),
            ),
        )
        trace6 = go.Scatter(
            x=x_range,
            y=x_range,
            mode="lines",
            line=dict(width=2, color="rgb(0.8, 0.8, 0.8)"),
        )
        data = [trace5, trace6]
        layout4 = go.Layout(
            xaxis={
                "title": f"Measured {x_txt} - Test",
                "range": x_range,
            },
            yaxis={
                "title": f"Predicted {x_txt}",
                "range": x_range,
            },
            width=400,
            height=360,
            margin={"l": 60, "b": 40, "t": 10, "r": 10},
            showlegend=False,
            hovermode="closest",
            annotations=[
                dict(
                    x=x_range[0] + 0.2 * (x_range[1] - x_range[0]),
                    y=x_range[0] + 0.9 * (x_range[1] - x_range[0]),
                    showarrow=False,
                    text="{:.1f}% within 20 <br> {:.1f}% within 10".format(
                        p2_test, p1_test
                    ),
                )
            ],
        )
        g4 = go.FigureWidget(data=data, layout=layout4)

        def response_test(change):
            x, y, z0, p1, p2 = update_numbers(mt_slider.value - 1)
            with g3.batch_update():
                g3.data[0].x = x
                g3.data[0].y = y
                g3.data[0].marker = dict(
                    color=z0,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g3.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1)
            x_test, y_test, z0_test, p1_test, p2_test = update_numbers_test(
                mt_slider.value - 1
            )
            with g4.batch_update():
                g4.data[0].x = x_test
                g4.data[0].y = y_test
                g4.data[0].marker = dict(
                    color=z0_test,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g4.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2_test, p1_test)

        mt_slider.observe(response_test, names="value")
        container1 = widgets.VBox([g1, g2])
        container2 = widgets.HBox([container1, g3, g4])
        app = widgets.VBox([widgets.Label("Number of Scenes: "), mt_slider, container2])

    return app


def scatter_plot_agb_retrieval_04a_noint_noc_walpha(
    wd,
    mask_file,
    agb_file,
    sar_list,
    out_name,
    test_mask=None,
    x_range=[0, 200],
    x_txt="AGB (Mg/ha)",
):
    os.chdir(wd)

    if not os.path.exists("output"):
        os.mkdir("output")
    out_prefix = f"output/{out_name}"
    mdl00 = FieldRetrieval(mask_file, out_name=out_prefix)
    z1, W1, mask_ws1 = mdl00.inversion_setup(
        sar_list, agb_file=agb_file, mask_file=mask_file, n_1=0
    )
    z1_dim = z1.shape
    z1r = z1.reshape([-1, z1_dim[1] * z1_dim[2] * z1_dim[3]])
    z0 = mdl00.inversion_return_valid(z1r, mask_ws1, mask_file=mask_file)
    W0 = mdl00.inversion_return_valid(W1, mask_ws1, mask_file=mask_file)
    z0 = z0.reshape([-1, z1_dim[1], z1_dim[2], z1_dim[3]])
    param0_file = mdl00.params_calibration_cband_04a_noint_noc_walpha(
        z0, W0, out_prefix
    )
    W_mean, _ = mdl00.inversion_recursive_ws_04a_noint(
        z0, mask_file, param0_file=param0_file
    )

    y0 = W0
    m_t = W_mean.shape[1]
    y0 = y0.flatten()
    r2 = []
    rmse = []
    W_mean2 = []
    for i in range(m_t):
        # print(i)
        # for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 9, 12, 10, 11]:
        x0 = np.nanmean(W_mean[:, : i + 1], axis=1)
        r2_1 = r2_score(x0[~np.isnan(x0), None], y0[~np.isnan(x0), None])
        # print("Variance score 1: {:.2f}".format(r2_1))
        rmse_1 = np.sqrt(
            mean_squared_error(x0[~np.isnan(x0), None], y0[~np.isnan(x0), None])
        )
        # print("RMSE: {:.5f}".format(rmse_1))
        W_mean2.append(x0)
        r2.append(r2_1)
        rmse.append(rmse_1)
    r2 = np.array(r2)
    rmse = np.array(rmse)
    df = pd.DataFrame(
        {
            "AGB_measured": W0.ravel(),
            "AGB_predicted": W_mean2[-1],
            "HH_mean": np.nanmean(z0[:, 0, :, 1], axis=-1),
            "HV_mean": np.nanmean(z0[:, 0, :, 0], axis=-1),
        }
    )
    df.to_csv(f"{out_prefix}_mean_data.csv")

    trace1 = go.Scatter(
        x=np.arange(m_t) + 1,
        y=r2,
        mode="lines+markers",
        name="R<sup>2</sup>",
        marker=dict(size=10),
    )
    trace2 = go.Scatter(
        x=np.arange(m_t) + 1,
        y=rmse,
        mode="lines+markers",
        name="RMSE",
        marker=dict(size=10),
    )
    layout1 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "R<sup>2</sup>",
        },
        width=300,
        height=180,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        hovermode="closest",
    )
    layout2 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "RMSE",
        },
        width=300,
        height=180,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        hovermode="closest",
    )
    g1 = go.FigureWidget(data=[trace1], layout=layout1)
    g2 = go.FigureWidget(data=[trace2], layout=layout2)
    mt_slider = widgets.IntSlider(
        value=10,
        min=1,
        max=m_t,
        step=1,
        disabled=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="80%"),
    )

    def update_numbers(m_ti=9):
        y = W_mean2[m_ti]
        x = y0
        xy = np.vstack([x, y])
        z0 = gaussian_kde(xy)(xy)
        p2 = np.sum((x - 20 < y) & (y < x + 20) & (x < 100)) / x[x < 100].size * 100
        p1 = np.sum((x - 10 < y) & (y < x + 10) & (x < 100)) / x[x < 100].size * 100
        return x, y, z0, p1, p2

    x, y, z0, p1, p2 = update_numbers()
    trace3 = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        name="points",
        marker=dict(
            color=z0,
            colorscale="Blues_r",
            reversescale=True,
            size=10,
            opacity=0.5,
            line=dict(width=1),
        ),
    )
    trace4 = go.Scatter(
        x=x_range,
        y=x_range,
        mode="lines",
        line=dict(width=2, color="rgb(0.8, 0.8, 0.8)"),
    )
    data = [trace3, trace4]
    layout3 = go.Layout(
        xaxis={
            "title": f"Measured {x_txt}",
            "range": x_range,
        },
        yaxis={
            "title": f"Predicted {x_txt}",
            "range": x_range,
        },
        width=400,
        height=360,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        showlegend=False,
        hovermode="closest",
        annotations=[
            dict(
                x=x_range[0] + 0.2 * (x_range[1] - x_range[0]),
                y=x_range[0] + 0.9 * (x_range[1] - x_range[0]),
                showarrow=False,
                text="{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1),
            )
        ],
    )
    g3 = go.FigureWidget(data=data, layout=layout3)

    if test_mask is None:

        def response(change):
            x, y, z0, p1, p2 = update_numbers(mt_slider.value - 1)
            with g3.batch_update():
                g3.data[0].x = x
                g3.data[0].y = y
                g3.data[0].marker = dict(
                    color=z0,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g3.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1)

        mt_slider.observe(response, names="value")
        container1 = widgets.VBox([g1, g2])
        container2 = widgets.HBox([container1, g3])
        app = widgets.VBox([widgets.Label("Number of Scenes: "), mt_slider, container2])
    else:
        mdl00 = FieldRetrieval(test_mask, out_name=f"tmp/tmp_test_{out_name}")
        z1_test, W1_test, mask_ws1_test = mdl00.inversion_setup(
            sar_list, agb_file=agb_file, mask_file=test_mask, n_1=0
        )
        z1r_test = z1_test.reshape([-1, z1_dim[1] * z1_dim[2] * z1_dim[3]])
        z0_test = mdl00.inversion_return_valid(
            z1r_test, mask_ws1_test, mask_file=test_mask
        )
        W0_test = mdl00.inversion_return_valid(
            W1_test, mask_ws1_test, mask_file=test_mask
        )
        z0_test = z0_test.reshape([-1, z1_dim[1], z1_dim[2], z1_dim[3]])
        W_mean_test, _ = mdl00.inversion_recursive_ws_04a_noint(
            z0_test, test_mask, param0_file=param0_file
        )
        y0_test = W0_test
        y0_test = y0_test.flatten()

        W_mean2_test = []
        for i in range(m_t):
            x0 = np.mean(W_mean_test[:, : i + 1], axis=1)
            W_mean2_test.append(x0)

        def update_numbers_test(m_ti=9):
            y = W_mean2_test[m_ti]
            x = y0_test
            xy = np.vstack([x, y])
            z0 = gaussian_kde(xy)(xy)
            p2 = np.sum((x - 20 < y) & (y < x + 20) & (x < 100)) / x[x < 100].size * 100
            p1 = np.sum((x - 10 < y) & (y < x + 10) & (x < 100)) / x[x < 100].size * 100
            return x, y, z0, p1, p2

        x_test, y_test, z0_test, p1_test, p2_test = update_numbers_test()
        trace5 = go.Scatter(
            x=x_test,
            y=y_test,
            mode="markers",
            name="points",
            marker=dict(
                color=z0_test,
                colorscale="Blues_r",
                reversescale=True,
                size=10,
                opacity=0.5,
                line=dict(width=1),
            ),
        )
        trace6 = go.Scatter(
            x=x_range,
            y=x_range,
            mode="lines",
            line=dict(width=2, color="rgb(0.8, 0.8, 0.8)"),
        )
        data = [trace5, trace6]
        layout4 = go.Layout(
            xaxis={
                "title": f"Measured {x_txt} - Test",
                "range": x_range,
            },
            yaxis={
                "title": f"Predicted {x_txt}",
                "range": x_range,
            },
            width=400,
            height=360,
            margin={"l": 60, "b": 40, "t": 10, "r": 10},
            showlegend=False,
            hovermode="closest",
            annotations=[
                dict(
                    x=x_range[0] + 0.2 * (x_range[1] - x_range[0]),
                    y=x_range[0] + 0.9 * (x_range[1] - x_range[0]),
                    showarrow=False,
                    text="{:.1f}% within 20 <br> {:.1f}% within 10".format(
                        p2_test, p1_test
                    ),
                )
            ],
        )
        g4 = go.FigureWidget(data=data, layout=layout4)

        def response_test(change):
            x, y, z0, p1, p2 = update_numbers(mt_slider.value - 1)
            with g3.batch_update():
                g3.data[0].x = x
                g3.data[0].y = y
                g3.data[0].marker = dict(
                    color=z0,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g3.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1)
            x_test, y_test, z0_test, p1_test, p2_test = update_numbers_test(
                mt_slider.value - 1
            )
            with g4.batch_update():
                g4.data[0].x = x_test
                g4.data[0].y = y_test
                g4.data[0].marker = dict(
                    color=z0_test,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g4.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2_test, p1_test)

        mt_slider.observe(response_test, names="value")
        container1 = widgets.VBox([g1, g2])
        container2 = widgets.HBox([container1, g3, g4])
        app = widgets.VBox([widgets.Label("Number of Scenes: "), mt_slider, container2])

    return app


def scatter_plot_agb_retrieval_04a_noint_nos_walpha(
    wd,
    mask_file,
    agb_file,
    sar_list,
    out_name,
    test_mask=None,
    x_range=[0, 200],
    x_txt="AGB (Mg/ha)",
):
    os.chdir(wd)

    if not os.path.exists("output"):
        os.mkdir("output")
    out_prefix = f"output/{out_name}"
    mdl00 = FieldRetrieval(mask_file, out_name=out_prefix)
    z1, W1, mask_ws1 = mdl00.inversion_setup(
        sar_list, agb_file=agb_file, mask_file=mask_file, n_1=0
    )
    z1_dim = z1.shape
    z1r = z1.reshape([-1, z1_dim[1] * z1_dim[2] * z1_dim[3]])
    z0 = mdl00.inversion_return_valid(z1r, mask_ws1, mask_file=mask_file)
    W0 = mdl00.inversion_return_valid(W1, mask_ws1, mask_file=mask_file)
    z0 = z0.reshape([-1, z1_dim[1], z1_dim[2], z1_dim[3]])
    param0_file = mdl00.params_calibration_cband_04a_noint_nos_walpha(
        z0, W0, out_prefix
    )
    W_mean, _ = mdl00.inversion_recursive_ws_04a_noint(
        z0, mask_file, param0_file=param0_file
    )

    y0 = W0
    m_t = W_mean.shape[1]
    y0 = y0.flatten()
    r2 = []
    rmse = []
    W_mean2 = []
    for i in range(m_t):
        # print(i)
        # for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 9, 12, 10, 11]:
        x0 = np.nanmean(W_mean[:, : i + 1], axis=1)
        r2_1 = r2_score(x0[~np.isnan(x0), None], y0[~np.isnan(x0), None])
        # print("Variance score 1: {:.2f}".format(r2_1))
        rmse_1 = np.sqrt(
            mean_squared_error(x0[~np.isnan(x0), None], y0[~np.isnan(x0), None])
        )
        # print("RMSE: {:.5f}".format(rmse_1))
        W_mean2.append(x0)
        r2.append(r2_1)
        rmse.append(rmse_1)
    r2 = np.array(r2)
    rmse = np.array(rmse)
    df = pd.DataFrame(
        {
            "AGB_measured": W0.ravel(),
            "AGB_predicted": W_mean2[-1],
            "HH_mean": np.nanmean(z0[:, 0, :, 1], axis=-1),
            "HV_mean": np.nanmean(z0[:, 0, :, 0], axis=-1),
        }
    )
    df.to_csv(f"{out_prefix}_mean_data.csv")

    trace1 = go.Scatter(
        x=np.arange(m_t) + 1,
        y=r2,
        mode="lines+markers",
        name="R<sup>2</sup>",
        marker=dict(size=10),
    )
    trace2 = go.Scatter(
        x=np.arange(m_t) + 1,
        y=rmse,
        mode="lines+markers",
        name="RMSE",
        marker=dict(size=10),
    )
    layout1 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "R<sup>2</sup>",
        },
        width=300,
        height=180,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        hovermode="closest",
    )
    layout2 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "RMSE",
        },
        width=300,
        height=180,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        hovermode="closest",
    )
    g1 = go.FigureWidget(data=[trace1], layout=layout1)
    g2 = go.FigureWidget(data=[trace2], layout=layout2)
    mt_slider = widgets.IntSlider(
        value=10,
        min=1,
        max=m_t,
        step=1,
        disabled=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="80%"),
    )

    def update_numbers(m_ti=9):
        y = W_mean2[m_ti]
        x = y0
        xy = np.vstack([x, y])
        z0 = gaussian_kde(xy)(xy)
        p2 = np.sum((x - 20 < y) & (y < x + 20) & (x < 100)) / x[x < 100].size * 100
        p1 = np.sum((x - 10 < y) & (y < x + 10) & (x < 100)) / x[x < 100].size * 100
        return x, y, z0, p1, p2

    x, y, z0, p1, p2 = update_numbers()
    trace3 = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        name="points",
        marker=dict(
            color=z0,
            colorscale="Blues_r",
            reversescale=True,
            size=10,
            opacity=0.5,
            line=dict(width=1),
        ),
    )
    trace4 = go.Scatter(
        x=x_range,
        y=x_range,
        mode="lines",
        line=dict(width=2, color="rgb(0.8, 0.8, 0.8)"),
    )
    data = [trace3, trace4]
    layout3 = go.Layout(
        xaxis={
            "title": f"Measured {x_txt}",
            "range": x_range,
        },
        yaxis={
            "title": f"Predicted {x_txt}",
            "range": x_range,
        },
        width=400,
        height=360,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        showlegend=False,
        hovermode="closest",
        annotations=[
            dict(
                x=x_range[0] + 0.2 * (x_range[1] - x_range[0]),
                y=x_range[0] + 0.9 * (x_range[1] - x_range[0]),
                showarrow=False,
                text="{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1),
            )
        ],
    )
    g3 = go.FigureWidget(data=data, layout=layout3)

    if test_mask is None:

        def response(change):
            x, y, z0, p1, p2 = update_numbers(mt_slider.value - 1)
            with g3.batch_update():
                g3.data[0].x = x
                g3.data[0].y = y
                g3.data[0].marker = dict(
                    color=z0,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g3.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1)

        mt_slider.observe(response, names="value")
        container1 = widgets.VBox([g1, g2])
        container2 = widgets.HBox([container1, g3])
        app = widgets.VBox([widgets.Label("Number of Scenes: "), mt_slider, container2])
    else:
        mdl00 = FieldRetrieval(test_mask, out_name=f"tmp/tmp_test_{out_name}")
        z1_test, W1_test, mask_ws1_test = mdl00.inversion_setup(
            sar_list, agb_file=agb_file, mask_file=test_mask, n_1=0
        )
        z1r_test = z1_test.reshape([-1, z1_dim[1] * z1_dim[2] * z1_dim[3]])
        z0_test = mdl00.inversion_return_valid(
            z1r_test, mask_ws1_test, mask_file=test_mask
        )
        W0_test = mdl00.inversion_return_valid(
            W1_test, mask_ws1_test, mask_file=test_mask
        )
        z0_test = z0_test.reshape([-1, z1_dim[1], z1_dim[2], z1_dim[3]])
        W_mean_test, _ = mdl00.inversion_recursive_ws_04a_noint(
            z0_test, test_mask, param0_file=param0_file
        )
        y0_test = W0_test
        y0_test = y0_test.flatten()

        W_mean2_test = []
        for i in range(m_t):
            x0 = np.mean(W_mean_test[:, : i + 1], axis=1)
            W_mean2_test.append(x0)

        def update_numbers_test(m_ti=9):
            y = W_mean2_test[m_ti]
            x = y0_test
            xy = np.vstack([x, y])
            z0 = gaussian_kde(xy)(xy)
            p2 = np.sum((x - 20 < y) & (y < x + 20) & (x < 100)) / x[x < 100].size * 100
            p1 = np.sum((x - 10 < y) & (y < x + 10) & (x < 100)) / x[x < 100].size * 100
            return x, y, z0, p1, p2

        x_test, y_test, z0_test, p1_test, p2_test = update_numbers_test()
        trace5 = go.Scatter(
            x=x_test,
            y=y_test,
            mode="markers",
            name="points",
            marker=dict(
                color=z0_test,
                colorscale="Blues_r",
                reversescale=True,
                size=10,
                opacity=0.5,
                line=dict(width=1),
            ),
        )
        trace6 = go.Scatter(
            x=x_range,
            y=x_range,
            mode="lines",
            line=dict(width=2, color="rgb(0.8, 0.8, 0.8)"),
        )
        data = [trace5, trace6]
        layout4 = go.Layout(
            xaxis={
                "title": f"Measured {x_txt} - Test",
                "range": x_range,
            },
            yaxis={
                "title": f"Predicted {x_txt}",
                "range": x_range,
            },
            width=400,
            height=360,
            margin={"l": 60, "b": 40, "t": 10, "r": 10},
            showlegend=False,
            hovermode="closest",
            annotations=[
                dict(
                    x=x_range[0] + 0.2 * (x_range[1] - x_range[0]),
                    y=x_range[0] + 0.9 * (x_range[1] - x_range[0]),
                    showarrow=False,
                    text="{:.1f}% within 20 <br> {:.1f}% within 10".format(
                        p2_test, p1_test
                    ),
                )
            ],
        )
        g4 = go.FigureWidget(data=data, layout=layout4)

        def response_test(change):
            x, y, z0, p1, p2 = update_numbers(mt_slider.value - 1)
            with g3.batch_update():
                g3.data[0].x = x
                g3.data[0].y = y
                g3.data[0].marker = dict(
                    color=z0,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g3.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1)
            x_test, y_test, z0_test, p1_test, p2_test = update_numbers_test(
                mt_slider.value - 1
            )
            with g4.batch_update():
                g4.data[0].x = x_test
                g4.data[0].y = y_test
                g4.data[0].marker = dict(
                    color=z0_test,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g4.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2_test, p1_test)

        mt_slider.observe(response_test, names="value")
        container1 = widgets.VBox([g1, g2])
        container2 = widgets.HBox([container1, g3, g4])
        app = widgets.VBox([widgets.Label("Number of Scenes: "), mt_slider, container2])

    return app


def scatter_plot_agb_retrieval_04a_AB(
    wd,
    mask_file,
    agb_file,
    sar_list,
    out_name,
    test_mask=None,
    x_range=[0, 200],
    x_txt="AGB (Mg/ha)",
):
    os.chdir(wd)

    if not os.path.exists("output"):
        os.mkdir("output")
    out_prefix = f"output/{out_name}"
    mdl00 = FieldRetrieval(mask_file, out_name=out_prefix)
    z1, W1, mask_ws1 = mdl00.inversion_setup(
        sar_list, agb_file=agb_file, mask_file=mask_file, n_1=0
    )
    z1_dim = z1.shape
    z1r = z1.reshape([-1, z1_dim[1] * z1_dim[2] * z1_dim[3]])
    z0 = mdl00.inversion_return_valid(z1r, mask_ws1, mask_file=mask_file)
    W0 = mdl00.inversion_return_valid(W1, mask_ws1, mask_file=mask_file)
    z0 = z0.reshape([-1, z1_dim[1], z1_dim[2], z1_dim[3]])
    param0_file = mdl00.params_calibration_cband_04a_AB(z0, W0, out_prefix)
    W_mean, _ = mdl00.inversion_recursive_ws_04a_noint(
        z0, mask_file, param0_file=param0_file
    )

    y0 = W0
    m_t = W_mean.shape[1]
    y0 = y0.flatten()
    r2 = []
    rmse = []
    W_mean2 = []
    for i in range(m_t):
        # print(i)
        # for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 9, 12, 10, 11]:
        x0 = np.nanmean(W_mean[:, : i + 1], axis=1)
        r2_1 = r2_score(x0[~np.isnan(x0), None], y0[~np.isnan(x0), None])
        # print("Variance score 1: {:.2f}".format(r2_1))
        rmse_1 = np.sqrt(
            mean_squared_error(x0[~np.isnan(x0), None], y0[~np.isnan(x0), None])
        )
        # print("RMSE: {:.5f}".format(rmse_1))
        W_mean2.append(x0)
        r2.append(r2_1)
        rmse.append(rmse_1)
    r2 = np.array(r2)
    rmse = np.array(rmse)
    df = pd.DataFrame(
        {
            "AGB_measured": W0.ravel(),
            "AGB_predicted": W_mean2[-1],
            "HH_mean": np.nanmean(z0[:, 0, :, 1], axis=-1),
            "HV_mean": np.nanmean(z0[:, 0, :, 0], axis=-1),
        }
    )
    df.to_csv(f"{out_prefix}_mean_data.csv")

    trace1 = go.Scatter(
        x=np.arange(m_t) + 1,
        y=r2,
        mode="lines+markers",
        name="R<sup>2</sup>",
        marker=dict(size=10),
    )
    trace2 = go.Scatter(
        x=np.arange(m_t) + 1,
        y=rmse,
        mode="lines+markers",
        name="RMSE",
        marker=dict(size=10),
    )
    layout1 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "R<sup>2</sup>",
        },
        width=300,
        height=180,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        hovermode="closest",
    )
    layout2 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "RMSE",
        },
        width=300,
        height=180,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        hovermode="closest",
    )
    g1 = go.FigureWidget(data=[trace1], layout=layout1)
    g2 = go.FigureWidget(data=[trace2], layout=layout2)
    mt_slider = widgets.IntSlider(
        value=10,
        min=1,
        max=m_t,
        step=1,
        disabled=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="80%"),
    )

    def update_numbers(m_ti=9):
        y = W_mean2[m_ti]
        x = y0
        xy = np.vstack([x, y])
        z0 = gaussian_kde(xy)(xy)
        p2 = np.sum((x - 20 < y) & (y < x + 20) & (x < 100)) / x[x < 100].size * 100
        p1 = np.sum((x - 10 < y) & (y < x + 10) & (x < 100)) / x[x < 100].size * 100
        return x, y, z0, p1, p2

    x, y, z0, p1, p2 = update_numbers()
    trace3 = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        name="points",
        marker=dict(
            color=z0,
            colorscale="Blues_r",
            reversescale=True,
            size=10,
            opacity=0.5,
            line=dict(width=1),
        ),
    )
    trace4 = go.Scatter(
        x=x_range,
        y=x_range,
        mode="lines",
        line=dict(width=2, color="rgb(0.8, 0.8, 0.8)"),
    )
    data = [trace3, trace4]
    layout3 = go.Layout(
        xaxis={
            "title": f"Measured {x_txt}",
            "range": x_range,
        },
        yaxis={
            "title": f"Predicted {x_txt}",
            "range": x_range,
        },
        width=400,
        height=360,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        showlegend=False,
        hovermode="closest",
        annotations=[
            dict(
                x=x_range[0] + 0.2 * (x_range[1] - x_range[0]),
                y=x_range[0] + 0.9 * (x_range[1] - x_range[0]),
                showarrow=False,
                text="{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1),
            )
        ],
    )
    g3 = go.FigureWidget(data=data, layout=layout3)

    if test_mask is None:

        def response(change):
            x, y, z0, p1, p2 = update_numbers(mt_slider.value - 1)
            with g3.batch_update():
                g3.data[0].x = x
                g3.data[0].y = y
                g3.data[0].marker = dict(
                    color=z0,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g3.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1)

        mt_slider.observe(response, names="value")
        container1 = widgets.VBox([g1, g2])
        container2 = widgets.HBox([container1, g3])
        app = widgets.VBox([widgets.Label("Number of Scenes: "), mt_slider, container2])
    else:
        mdl00 = FieldRetrieval(test_mask, out_name=f"tmp/tmp_test_{out_name}")
        z1_test, W1_test, mask_ws1_test = mdl00.inversion_setup(
            sar_list, agb_file=agb_file, mask_file=test_mask, n_1=0
        )
        z1r_test = z1_test.reshape([-1, z1_dim[1] * z1_dim[2] * z1_dim[3]])
        z0_test = mdl00.inversion_return_valid(
            z1r_test, mask_ws1_test, mask_file=test_mask
        )
        W0_test = mdl00.inversion_return_valid(
            W1_test, mask_ws1_test, mask_file=test_mask
        )
        z0_test = z0_test.reshape([-1, z1_dim[1], z1_dim[2], z1_dim[3]])
        W_mean_test, _ = mdl00.inversion_recursive_ws_04a_noint(
            z0_test, test_mask, param0_file=param0_file
        )
        y0_test = W0_test
        y0_test = y0_test.flatten()

        W_mean2_test = []
        for i in range(m_t):
            x0 = np.mean(W_mean_test[:, : i + 1], axis=1)
            W_mean2_test.append(x0)

        def update_numbers_test(m_ti=9):
            y = W_mean2_test[m_ti]
            x = y0_test
            xy = np.vstack([x, y])
            z0 = gaussian_kde(xy)(xy)
            p2 = np.sum((x - 20 < y) & (y < x + 20) & (x < 100)) / x[x < 100].size * 100
            p1 = np.sum((x - 10 < y) & (y < x + 10) & (x < 100)) / x[x < 100].size * 100
            return x, y, z0, p1, p2

        x_test, y_test, z0_test, p1_test, p2_test = update_numbers_test()
        trace5 = go.Scatter(
            x=x_test,
            y=y_test,
            mode="markers",
            name="points",
            marker=dict(
                color=z0_test,
                colorscale="Blues_r",
                reversescale=True,
                size=10,
                opacity=0.5,
                line=dict(width=1),
            ),
        )
        trace6 = go.Scatter(
            x=x_range,
            y=x_range,
            mode="lines",
            line=dict(width=2, color="rgb(0.8, 0.8, 0.8)"),
        )
        data = [trace5, trace6]
        layout4 = go.Layout(
            xaxis={
                "title": f"Measured {x_txt} - Test",
                "range": x_range,
            },
            yaxis={
                "title": f"Predicted {x_txt}",
                "range": x_range,
            },
            width=400,
            height=360,
            margin={"l": 60, "b": 40, "t": 10, "r": 10},
            showlegend=False,
            hovermode="closest",
            annotations=[
                dict(
                    x=x_range[0] + 0.2 * (x_range[1] - x_range[0]),
                    y=x_range[0] + 0.9 * (x_range[1] - x_range[0]),
                    showarrow=False,
                    text="{:.1f}% within 20 <br> {:.1f}% within 10".format(
                        p2_test, p1_test
                    ),
                )
            ],
        )
        g4 = go.FigureWidget(data=data, layout=layout4)

        def response_test(change):
            x, y, z0, p1, p2 = update_numbers(mt_slider.value - 1)
            with g3.batch_update():
                g3.data[0].x = x
                g3.data[0].y = y
                g3.data[0].marker = dict(
                    color=z0,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g3.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2, p1)
            x_test, y_test, z0_test, p1_test, p2_test = update_numbers_test(
                mt_slider.value - 1
            )
            with g4.batch_update():
                g4.data[0].x = x_test
                g4.data[0].y = y_test
                g4.data[0].marker = dict(
                    color=z0_test,
                    colorscale="Blues_r",
                    reversescale=True,
                    size=10,
                    opacity=0.5,
                    line=dict(width=1),
                )
                g4.layout.annotations[0][
                    "text"
                ] = "{:.1f}% within 20 <br> {:.1f}% within 10".format(p2_test, p1_test)

        mt_slider.observe(response_test, names="value")
        container1 = widgets.VBox([g1, g2])
        container2 = widgets.HBox([container1, g3, g4])
        app = widgets.VBox([widgets.Label("Number of Scenes: "), mt_slider, container2])

    return app


def scatter_plot_agb_retrieval_3g(
    wd,
    mask_file,
    param0_file,
    z0,
    W0_train,
    sm0_train,
    ks0_train,
    z0_test,
    W0_test,
    sm0_test,
    ks0_test,
    out_name,
    x_range=[0, 200],
    init_weights=[1, 1, 1],
    x_txt="AGB (Mg/ha)",
):
    os.chdir(wd)

    if not os.path.exists("output"):
        os.mkdir("output")
    out_prefix = f"output/{out_name}"
    mdl00 = FieldRetrieval(mask_file, out_name=out_prefix)
    SM, W_mean = mdl00.inversion_nobs_sm_03e_wkks(
        z0, ks0_train, sm0_train, param0_file=param0_file, init_weights=init_weights
    )
    # print(W_mean.shape)

    y0 = W0_train.ravel()
    m_t = W_mean.shape[1]
    r2 = []
    rmse = []
    W_mean2 = []
    # for i in range(m_t):
    # print(i)
    for i in [0, 5, 10, 9, 8, 7, 6, 4, 1, 2, 3]:
        x0 = np.nanmean(W_mean[:, : i + 1], axis=1)
        r2_1 = r2_score(x0[~np.isnan(x0), None], y0[~np.isnan(x0), None])
        rmse_1 = np.sqrt(
            mean_squared_error(x0[~np.isnan(x0), None], y0[~np.isnan(x0), None])
        )
        W_mean2.append(x0)
        r2.append(r2_1)
        rmse.append(rmse_1)
    r2 = np.array(r2)
    rmse = np.array(rmse)
    # print(len(W_mean2))
    df = pd.DataFrame(
        {
            "AGB_measured": W0_train.ravel(),
            "AGB_predicted": W_mean2[-1],
            "HH_mean": np.nanmean(z0[:, 0, :, 1], axis=-1),
            "HV_mean": np.nanmean(z0[:, 0, :, 0], axis=-1),
        }
    )
    df.to_csv(f"{out_prefix}_mean_data.csv")

    trace1 = go.Scatter(
        x=np.arange(m_t) + 1,
        y=r2,
        mode="lines+markers",
        name="R<sup>2</sup>",
        marker=dict(size=7),
    )
    trace2 = go.Scatter(
        x=np.arange(m_t) + 1,
        y=rmse,
        mode="lines+markers",
        name="RMSE",
        marker=dict(size=7),
    )
    layout1 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "R<sup>2</sup>",
        },
        width=300,
        height=180,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        hovermode="closest",
    )
    layout2 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "RMSE",
        },
        width=300,
        height=180,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        hovermode="closest",
    )
    g1 = go.FigureWidget(data=[trace1], layout=layout1)
    g2 = go.FigureWidget(data=[trace2], layout=layout2)
    mt_slider = widgets.IntSlider(
        value=1,
        min=1,
        max=m_t,
        step=1,
        disabled=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="80%"),
    )

    def update_numbers(m_ti=0):
        y = W_mean2[m_ti]
        x = y0
        xy = np.vstack([x, y])
        c0 = gaussian_kde(xy)(xy)
        # p2 = np.sum((x - 20 < y) & (y < x + 20) & (x < 100)) / x[x < 100].size * 100
        # p1 = np.sum((x - 10 < y) & (y < x + 10) & (x < 100)) / x[x < 100].size * 100
        p2 = np.sum((x - 2 < y) & (y < x + 2) & (x < 10)) / x[x < 10].size * 100
        p1 = np.sum((x - 1 < y) & (y < x + 1) & (x < 10)) / x[x < 10].size * 100
        return x, y, c0, p1, p2

    x, y, c0, p1, p2 = update_numbers()
    trace3 = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        name="points",
        marker=dict(
            color=c0,
            colorscale="Blues_r",
            reversescale=True,
            size=5,
            opacity=0.5,
            line=dict(width=1),
        ),
    )
    trace4 = go.Scatter(
        x=x_range,
        y=x_range,
        mode="lines",
        line=dict(width=2, color="rgb(0.8, 0.8, 0.8)"),
    )
    data = [trace3, trace4]
    layout3 = go.Layout(
        xaxis={
            "title": f"Measured {x_txt}",
            "range": x_range,
        },
        yaxis={
            "title": f"Predicted {x_txt}",
            "range": x_range,
        },
        width=400,
        height=360,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        showlegend=False,
        hovermode="closest",
        annotations=[
            dict(
                x=x_range[0] + 0.25 * (x_range[1] - x_range[0]),
                y=x_range[0] + 0.9 * (x_range[1] - x_range[0]),
                showarrow=False,
                text="{:.1f}% within 2 (0-10) <br> {:.1f}% within 1 (0-10)".format(
                    p2, p1
                ),
            )
        ],
    )
    g3 = go.FigureWidget(data=data, layout=layout3)

    # test data
    mdl00 = FieldRetrieval(mask_file, out_name=f"tmp/tmp_test_{out_name}")
    SM, W_mean_test = mdl00.inversion_nobs_sm_03e_wkks(
        z0_test, ks0_test, sm0_test, param0_file=param0_file, init_weights=init_weights
    )
    y0_test = W0_test
    y0_test = y0_test.flatten()

    W_mean2_test = []
    for i in range(m_t):
        x0 = np.mean(W_mean_test[:, : i + 1], axis=1)
        W_mean2_test.append(x0)

    def update_numbers_test(m_ti=0):
        y = W_mean2_test[m_ti]
        x = y0_test
        xy = np.vstack([x, y])
        c0 = gaussian_kde(xy)(xy)
        p2 = np.sum((x - 2 < y) & (y < x + 2) & (x < 10)) / x[x < 10].size * 100
        p1 = np.sum((x - 1 < y) & (y < x + 1) & (x < 10)) / x[x < 10].size * 100
        return x, y, c0, p1, p2

    x_test, y_test, c0_test, p1_test, p2_test = update_numbers_test()
    trace5 = go.Scatter(
        x=x_test,
        y=y_test,
        mode="markers",
        name="points",
        marker=dict(
            color=c0_test,
            colorscale="Blues_r",
            reversescale=True,
            size=5,
            opacity=0.5,
            line=dict(width=1),
        ),
    )
    trace6 = go.Scatter(
        x=x_range,
        y=x_range,
        mode="lines",
        line=dict(width=2, color="rgb(0.8, 0.8, 0.8)"),
    )
    data = [trace5, trace6]
    layout4 = go.Layout(
        xaxis={
            "title": f"Measured {x_txt} - Test",
            "range": x_range,
        },
        yaxis={
            "title": f"Predicted {x_txt}",
            "range": x_range,
        },
        width=400,
        height=360,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        showlegend=False,
        hovermode="closest",
        annotations=[
            dict(
                x=x_range[0] + 0.25 * (x_range[1] - x_range[0]),
                y=x_range[0] + 0.9 * (x_range[1] - x_range[0]),
                showarrow=False,
                text="{:.1f}% within 2 (0-10) <br> {:.1f}% within 1 (0-10)".format(
                    p2_test, p1_test
                ),
            )
        ],
    )
    g4 = go.FigureWidget(data=data, layout=layout4)

    def response_test(change):
        x, y, c0, p1, p2 = update_numbers(mt_slider.value - 1)
        with g3.batch_update():
            g3.data[0].x = x
            g3.data[0].y = y
            g3.data[0].marker = dict(
                color=c0,
                colorscale="Blues_r",
                reversescale=True,
                size=5,
                opacity=0.5,
                line=dict(width=1),
            )
            g3.layout.annotations[0][
                "text"
            ] = "{:.1f}% within 2 (0-10) <br> {:.1f}% within 1 (0-10)".format(p2, p1)
        x_test, y_test, c0_test, p1_test, p2_test = update_numbers_test(
            mt_slider.value - 1
        )
        with g4.batch_update():
            g4.data[0].x = x_test
            g4.data[0].y = y_test
            g4.data[0].marker = dict(
                color=c0_test,
                colorscale="Blues_r",
                reversescale=True,
                size=5,
                opacity=0.5,
                line=dict(width=1),
            )
            g4.layout.annotations[0][
                "text"
            ] = "{:.1f}% within 2 (0-10) <br> {:.1f}% within 1 (0-10)".format(
                p2_test, p1_test
            )

    mt_slider.observe(response_test, names="value")
    container1 = widgets.VBox([g1, g2])
    container2 = widgets.HBox([container1, g3, g4])
    app = widgets.VBox([widgets.Label("Number of Scenes: "), mt_slider, container2])

    return app


def scatter_plot_agb_retrieval_3h2(
    wd,
    mask_file,
    param0_file,
    z0,
    W0_train,
    sm0_train,
    ks0_train,
    z0_test,
    W0_test,
    sm0_test,
    ks0_test,
    out_name,
    x_range=[0, 200],
    init_weights=[1, 1, 1],
    x_txt="AGB (Mg/ha)",
):
    os.chdir(wd)

    if not os.path.exists("output"):
        os.mkdir("output")
    out_prefix = f"output/{out_name}"
    mdl00 = FieldRetrieval(mask_file, out_name=out_prefix)
    SM, W_mean = mdl00.inversion_nobs_sm_03h2_wkks(
        z0, ks0_train, sm0_train, param0_file=param0_file, init_weights=init_weights
    )
    # print(W_mean.shape)

    y0 = W0_train.ravel()
    m_t = W_mean.shape[1]
    r2 = []
    rmse = []
    W_mean2 = []
    # for i in range(m_t):
    # print(i)
    for i in [0, 5, 10, 9, 8, 7, 6, 4, 1, 2, 3]:
        x0 = np.nanmean(W_mean[:, : i + 1], axis=1)
        r2_1 = r2_score(x0[~np.isnan(x0), None], y0[~np.isnan(x0), None])
        rmse_1 = np.sqrt(
            mean_squared_error(x0[~np.isnan(x0), None], y0[~np.isnan(x0), None])
        )
        W_mean2.append(x0)
        r2.append(r2_1)
        rmse.append(rmse_1)
    r2 = np.array(r2)
    rmse = np.array(rmse)
    # print(len(W_mean2))
    df = pd.DataFrame(
        {
            "AGB_measured": W0_train.ravel(),
            "AGB_predicted": W_mean2[-1],
            "HH_mean": np.nanmean(z0[:, 0, :, 1], axis=-1),
            "HV_mean": np.nanmean(z0[:, 0, :, 0], axis=-1),
        }
    )
    df.to_csv(f"{out_prefix}_mean_data.csv")

    trace1 = go.Scatter(
        x=np.arange(m_t) + 1,
        y=r2,
        mode="lines+markers",
        name="R<sup>2</sup>",
        marker=dict(size=7),
    )
    trace2 = go.Scatter(
        x=np.arange(m_t) + 1,
        y=rmse,
        mode="lines+markers",
        name="RMSE",
        marker=dict(size=7),
    )
    layout1 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "R<sup>2</sup>",
        },
        width=300,
        height=180,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        hovermode="closest",
    )
    layout2 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "RMSE",
        },
        width=300,
        height=180,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        hovermode="closest",
    )
    g1 = go.FigureWidget(data=[trace1], layout=layout1)
    g2 = go.FigureWidget(data=[trace2], layout=layout2)
    mt_slider = widgets.IntSlider(
        value=1,
        min=1,
        max=m_t,
        step=1,
        disabled=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="80%"),
    )

    def update_numbers(m_ti=0):
        y = W_mean2[m_ti]
        x = y0
        xy = np.vstack([x, y])
        c0 = gaussian_kde(xy)(xy)
        # p2 = np.sum((x - 20 < y) & (y < x + 20) & (x < 100)) / x[x < 100].size * 100
        # p1 = np.sum((x - 10 < y) & (y < x + 10) & (x < 100)) / x[x < 100].size * 100
        p2 = np.sum((x - 2 < y) & (y < x + 2) & (x < 10)) / x[x < 10].size * 100
        p1 = np.sum((x - 1 < y) & (y < x + 1) & (x < 10)) / x[x < 10].size * 100
        return x, y, c0, p1, p2

    x, y, c0, p1, p2 = update_numbers()
    trace3 = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        name="points",
        marker=dict(
            color=c0,
            colorscale="Blues_r",
            reversescale=True,
            size=5,
            opacity=0.5,
            line=dict(width=1),
        ),
    )
    trace4 = go.Scatter(
        x=x_range,
        y=x_range,
        mode="lines",
        line=dict(width=2, color="rgb(0.8, 0.8, 0.8)"),
    )
    data = [trace3, trace4]
    layout3 = go.Layout(
        xaxis={
            "title": f"Measured {x_txt}",
            "range": x_range,
        },
        yaxis={
            "title": f"Predicted {x_txt}",
            "range": x_range,
        },
        width=400,
        height=360,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        showlegend=False,
        hovermode="closest",
        annotations=[
            dict(
                x=x_range[0] + 0.25 * (x_range[1] - x_range[0]),
                y=x_range[0] + 0.9 * (x_range[1] - x_range[0]),
                showarrow=False,
                text="{:.1f}% within 2 (0-10) <br> {:.1f}% within 1 (0-10)".format(
                    p2, p1
                ),
            )
        ],
    )
    g3 = go.FigureWidget(data=data, layout=layout3)

    # test data
    mdl00 = FieldRetrieval(mask_file, out_name=f"tmp/tmp_test_{out_name}")
    SM, W_mean_test = mdl00.inversion_nobs_sm_03h2_wkks(
        z0_test, ks0_test, sm0_test, param0_file=param0_file, init_weights=init_weights
    )
    y0_test = W0_test
    y0_test = y0_test.flatten()

    W_mean2_test = []
    for i in range(m_t):
        x0 = np.mean(W_mean_test[:, : i + 1], axis=1)
        W_mean2_test.append(x0)

    def update_numbers_test(m_ti=0):
        y = W_mean2_test[m_ti]
        x = y0_test
        xy = np.vstack([x, y])
        c0 = gaussian_kde(xy)(xy)
        p2 = np.sum((x - 2 < y) & (y < x + 2) & (x < 10)) / x[x < 10].size * 100
        p1 = np.sum((x - 1 < y) & (y < x + 1) & (x < 10)) / x[x < 10].size * 100
        return x, y, c0, p1, p2

    x_test, y_test, c0_test, p1_test, p2_test = update_numbers_test()
    trace5 = go.Scatter(
        x=x_test,
        y=y_test,
        mode="markers",
        name="points",
        marker=dict(
            color=c0_test,
            colorscale="Blues_r",
            reversescale=True,
            size=5,
            opacity=0.5,
            line=dict(width=1),
        ),
    )
    trace6 = go.Scatter(
        x=x_range,
        y=x_range,
        mode="lines",
        line=dict(width=2, color="rgb(0.8, 0.8, 0.8)"),
    )
    data = [trace5, trace6]
    layout4 = go.Layout(
        xaxis={
            "title": f"Measured {x_txt} - Test",
            "range": x_range,
        },
        yaxis={
            "title": f"Predicted {x_txt}",
            "range": x_range,
        },
        width=400,
        height=360,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        showlegend=False,
        hovermode="closest",
        annotations=[
            dict(
                x=x_range[0] + 0.25 * (x_range[1] - x_range[0]),
                y=x_range[0] + 0.9 * (x_range[1] - x_range[0]),
                showarrow=False,
                text="{:.1f}% within 2 (0-10) <br> {:.1f}% within 1 (0-10)".format(
                    p2_test, p1_test
                ),
            )
        ],
    )
    g4 = go.FigureWidget(data=data, layout=layout4)

    def response_test(change):
        x, y, c0, p1, p2 = update_numbers(mt_slider.value - 1)
        with g3.batch_update():
            g3.data[0].x = x
            g3.data[0].y = y
            g3.data[0].marker = dict(
                color=c0,
                colorscale="Blues_r",
                reversescale=True,
                size=5,
                opacity=0.5,
                line=dict(width=1),
            )
            g3.layout.annotations[0][
                "text"
            ] = "{:.1f}% within 2 (0-10) <br> {:.1f}% within 1 (0-10)".format(p2, p1)
        x_test, y_test, c0_test, p1_test, p2_test = update_numbers_test(
            mt_slider.value - 1
        )
        with g4.batch_update():
            g4.data[0].x = x_test
            g4.data[0].y = y_test
            g4.data[0].marker = dict(
                color=c0_test,
                colorscale="Blues_r",
                reversescale=True,
                size=5,
                opacity=0.5,
                line=dict(width=1),
            )
            g4.layout.annotations[0][
                "text"
            ] = "{:.1f}% within 2 (0-10) <br> {:.1f}% within 1 (0-10)".format(
                p2_test, p1_test
            )

    mt_slider.observe(response_test, names="value")
    container1 = widgets.VBox([g1, g2])
    container2 = widgets.HBox([container1, g3, g4])
    app = widgets.VBox([widgets.Label("Number of Scenes: "), mt_slider, container2])

    return app


def scatter_plot_agb_retrieval_3h1(
    wd,
    mask_file,
    param0_file,
    z0,
    W0_train,
    sm0_train,
    ks0_train,
    z0_test,
    W0_test,
    sm0_test,
    ks0_test,
    out_name,
    x_range=[0, 200],
    init_weights=[1, 1],
    x_txt="AGB (Mg/ha)",
):
    os.chdir(wd)

    if not os.path.exists("output"):
        os.mkdir("output")
    out_prefix = f"output/{out_name}"
    mdl00 = FieldRetrieval(mask_file, out_name=out_prefix)
    SM, W_mean = mdl00.inversion_nobs_sm_03h1_wkks(
        z0, ks0_train, sm0_train, param0_file=param0_file, init_weights=init_weights
    )
    # print(W_mean.shape)

    y0 = W0_train.ravel()
    m_t = W_mean.shape[1]
    r2 = []
    rmse = []
    W_mean2 = []
    # for i in range(m_t):
    # print(i)
    for i in [0, 5, 10, 9, 8, 7, 6, 4, 1, 2, 3]:
        x0 = np.nanmean(W_mean[:, : i + 1], axis=1)
        r2_1 = r2_score(x0[~np.isnan(x0), None], y0[~np.isnan(x0), None])
        rmse_1 = np.sqrt(
            mean_squared_error(x0[~np.isnan(x0), None], y0[~np.isnan(x0), None])
        )
        W_mean2.append(x0)
        r2.append(r2_1)
        rmse.append(rmse_1)
    r2 = np.array(r2)
    rmse = np.array(rmse)
    # print(len(W_mean2))
    df = pd.DataFrame(
        {
            "AGB_measured": W0_train.ravel(),
            "AGB_predicted": W_mean2[-1],
            "HH_mean": np.nanmean(z0[:, 0, :, 1], axis=-1),
            "HV_mean": np.nanmean(z0[:, 0, :, 0], axis=-1),
        }
    )
    df.to_csv(f"{out_prefix}_mean_data.csv")

    trace1 = go.Scatter(
        x=np.arange(m_t) + 1,
        y=r2,
        mode="lines+markers",
        name="R<sup>2</sup>",
        marker=dict(size=7),
    )
    trace2 = go.Scatter(
        x=np.arange(m_t) + 1,
        y=rmse,
        mode="lines+markers",
        name="RMSE",
        marker=dict(size=7),
    )
    layout1 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "R<sup>2</sup>",
        },
        width=300,
        height=180,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        hovermode="closest",
    )
    layout2 = go.Layout(
        xaxis={"title": "Number of Scenes"},
        yaxis={
            "title": "RMSE",
        },
        width=300,
        height=180,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        hovermode="closest",
    )
    g1 = go.FigureWidget(data=[trace1], layout=layout1)
    g2 = go.FigureWidget(data=[trace2], layout=layout2)
    mt_slider = widgets.IntSlider(
        value=1,
        min=1,
        max=m_t,
        step=1,
        disabled=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="80%"),
    )

    def update_numbers(m_ti=0):
        y = W_mean2[m_ti]
        x = y0
        xy = np.vstack([x, y])
        c0 = gaussian_kde(xy)(xy)
        # p2 = np.sum((x - 20 < y) & (y < x + 20) & (x < 100)) / x[x < 100].size * 100
        # p1 = np.sum((x - 10 < y) & (y < x + 10) & (x < 100)) / x[x < 100].size * 100
        p2 = np.sum((x - 2 < y) & (y < x + 2) & (x < 10)) / x[x < 10].size * 100
        p1 = np.sum((x - 1 < y) & (y < x + 1) & (x < 10)) / x[x < 10].size * 100
        return x, y, c0, p1, p2

    x, y, c0, p1, p2 = update_numbers()
    trace3 = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        name="points",
        marker=dict(
            color=c0,
            colorscale="Blues_r",
            reversescale=True,
            size=5,
            opacity=0.5,
            line=dict(width=1),
        ),
    )
    trace4 = go.Scatter(
        x=x_range,
        y=x_range,
        mode="lines",
        line=dict(width=2, color="rgb(0.8, 0.8, 0.8)"),
    )
    data = [trace3, trace4]
    layout3 = go.Layout(
        xaxis={
            "title": f"Measured {x_txt}",
            "range": x_range,
        },
        yaxis={
            "title": f"Predicted {x_txt}",
            "range": x_range,
        },
        width=400,
        height=360,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        showlegend=False,
        hovermode="closest",
        annotations=[
            dict(
                x=x_range[0] + 0.25 * (x_range[1] - x_range[0]),
                y=x_range[0] + 0.9 * (x_range[1] - x_range[0]),
                showarrow=False,
                text="{:.1f}% within 2 (0-10) <br> {:.1f}% within 1 (0-10)".format(
                    p2, p1
                ),
            )
        ],
    )
    g3 = go.FigureWidget(data=data, layout=layout3)

    # test data
    mdl00 = FieldRetrieval(mask_file, out_name=f"tmp/tmp_test_{out_name}")
    SM, W_mean_test = mdl00.inversion_nobs_sm_03h2_wkks(
        z0_test, ks0_test, sm0_test, param0_file=param0_file, init_weights=init_weights
    )
    y0_test = W0_test
    y0_test = y0_test.flatten()

    W_mean2_test = []
    for i in range(m_t):
        x0 = np.mean(W_mean_test[:, : i + 1], axis=1)
        W_mean2_test.append(x0)

    def update_numbers_test(m_ti=0):
        y = W_mean2_test[m_ti]
        x = y0_test
        xy = np.vstack([x, y])
        c0 = gaussian_kde(xy)(xy)
        p2 = np.sum((x - 2 < y) & (y < x + 2) & (x < 10)) / x[x < 10].size * 100
        p1 = np.sum((x - 1 < y) & (y < x + 1) & (x < 10)) / x[x < 10].size * 100
        return x, y, c0, p1, p2

    x_test, y_test, c0_test, p1_test, p2_test = update_numbers_test()
    trace5 = go.Scatter(
        x=x_test,
        y=y_test,
        mode="markers",
        name="points",
        marker=dict(
            color=c0_test,
            colorscale="Blues_r",
            reversescale=True,
            size=5,
            opacity=0.5,
            line=dict(width=1),
        ),
    )
    trace6 = go.Scatter(
        x=x_range,
        y=x_range,
        mode="lines",
        line=dict(width=2, color="rgb(0.8, 0.8, 0.8)"),
    )
    data = [trace5, trace6]
    layout4 = go.Layout(
        xaxis={
            "title": f"Measured {x_txt} - Test",
            "range": x_range,
        },
        yaxis={
            "title": f"Predicted {x_txt}",
            "range": x_range,
        },
        width=400,
        height=360,
        margin={"l": 60, "b": 40, "t": 10, "r": 10},
        showlegend=False,
        hovermode="closest",
        annotations=[
            dict(
                x=x_range[0] + 0.25 * (x_range[1] - x_range[0]),
                y=x_range[0] + 0.9 * (x_range[1] - x_range[0]),
                showarrow=False,
                text="{:.1f}% within 2 (0-10) <br> {:.1f}% within 1 (0-10)".format(
                    p2_test, p1_test
                ),
            )
        ],
    )
    g4 = go.FigureWidget(data=data, layout=layout4)

    def response_test(change):
        x, y, c0, p1, p2 = update_numbers(mt_slider.value - 1)
        with g3.batch_update():
            g3.data[0].x = x
            g3.data[0].y = y
            g3.data[0].marker = dict(
                color=c0,
                colorscale="Blues_r",
                reversescale=True,
                size=5,
                opacity=0.5,
                line=dict(width=1),
            )
            g3.layout.annotations[0][
                "text"
            ] = "{:.1f}% within 2 (0-10) <br> {:.1f}% within 1 (0-10)".format(p2, p1)
        x_test, y_test, c0_test, p1_test, p2_test = update_numbers_test(
            mt_slider.value - 1
        )
        with g4.batch_update():
            g4.data[0].x = x_test
            g4.data[0].y = y_test
            g4.data[0].marker = dict(
                color=c0_test,
                colorscale="Blues_r",
                reversescale=True,
                size=5,
                opacity=0.5,
                line=dict(width=1),
            )
            g4.layout.annotations[0][
                "text"
            ] = "{:.1f}% within 2 (0-10) <br> {:.1f}% within 1 (0-10)".format(
                p2_test, p1_test
            )

    mt_slider.observe(response_test, names="value")
    container1 = widgets.VBox([g1, g2])
    container2 = widgets.HBox([container1, g3, g4])
    app = widgets.VBox([widgets.Label("Number of Scenes: "), mt_slider, container2])

    return app


def colormap_plot_agb_prediction(
    wd, out_radar_list, mask_file, param_file, out_name, ab_range=[0, 5]
):
    os.chdir(wd)
    if not os.path.exists("output"):
        os.mkdir("output")
    mdl00 = FieldRetrieval(mask_file, out_name=out_name)
    z1, _, mask_ws1 = mdl00.inversion_setup(out_radar_list, mask_file=mask_file, n_1=0)
    z1_dim = z1.shape
    z1r = z1.reshape([-1, z1_dim[1] * z1_dim[2] * z1_dim[3]])
    z0 = mdl00.inversion_return_valid(z1r, mask_ws1, mask_file=mask_file)
    z0 = z0.reshape([-1, z1_dim[1], z1_dim[2], z1_dim[3]])

    W_mean, S_mean = mdl00.inversion_recursive_ws(z0, mask_file, param0_file=param_file)

    in0 = gdal.Open(mask_file, gdal.GA_ReadOnly)

    fig, ax = plt.subplots(
        figsize=(5, 4),
    )
    basename = os.path.basename(out_radar_list[0])
    agb_name = f"{out_name}_agb_predictions_mean.tif"
    print(agb_name)
    array0 = in0.GetRasterBand(1).ReadAsArray()
    array0[array0 > 0] = np.mean(W_mean, axis=-1)
    try:
        os.remove(agb_name)
    except OSError:
        pass
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.CreateCopy(
        agb_name,
        in0,
        0,
        ["COMPRESS=LZW", "PREDICTOR=2"],
    )
    ds.GetRasterBand(1).WriteArray(array0)
    ds.FlushCache()  # Write to disk.
    ds = None
    im = ax.imshow(array0, cmap="gist_earth_r", vmin=0, vmax=100)
    ax.set_title("AGB Predictions Mean")
    ax.set_axis_off()
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel("AGB (Mg/ha)")

    fig, ax = plt.subplots(
        nrows=np.ceil(W_mean.shape[1] / 2).astype(int),
        ncols=2,
        figsize=(9, 4 * np.ceil(W_mean.shape[1] / 2)),
        sharex=True,
        sharey=True,
    )
    ax1 = ax.flatten()
    for k in range(len(out_radar_list) // 2):
        basename = os.path.basename(out_radar_list[k * 2])
        agb_name = f"{out_name}_agb_predictions_{basename[ab_range[0]:ab_range[1]]}.tif"
        print(agb_name)
        array0 = in0.GetRasterBand(1).ReadAsArray()
        array0[array0 > 0] = W_mean[:, k]
        try:
            os.remove(agb_name)
        except OSError:
            pass
        driver = gdal.GetDriverByName("GTiff")
        ds = driver.CreateCopy(
            agb_name,
            in0,
            0,
            ["COMPRESS=LZW", "PREDICTOR=2"],
        )
        ds.GetRasterBand(1).WriteArray(array0)
        ds.FlushCache()  # Write to disk.
        ds = None
        im = ax1[k].imshow(
            array0,
            cmap="gist_earth_r",
            vmin=np.quantile(W_mean, 0.01),
            vmax=np.quantile(W_mean, 0.99),
        )
        ax1[k].set_title(basename[ab_range[0] : ab_range[1]])
        ax1[k].set_axis_off()
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.2, 0.03, 0.6])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel("AGB (Mg/ha)")

    # fig, ax = plt.subplots(
    #     nrows=np.ceil(W_mean.shape[1] / 2).astype(int),
    #     ncols=2, figsize=(9, 4 * np.ceil(W_mean.shape[1] / 2)),
    #     sharex=True, sharey=True
    # )
    # ax1 = ax.flatten()
    # for k in range(len(out_radar_list) // 2):
    #     basename = os.path.basename(out_radar_list[k * 2])
    #     agb_name = f"{out_name}_S_predictions_{basename[ab_range[0]:ab_range[1]]}.tif"
    #     print(agb_name)
    #     array0 = in0.GetRasterBand(1).ReadAsArray()
    #     array0[array0 > 0] = S_mean[:, k]
    #     try:
    #         os.remove(agb_name)
    #     except OSError:
    #         pass
    #     driver = gdal.GetDriverByName("GTiff")
    #     ds = driver.CreateCopy(
    #         agb_name,
    #         in0,
    #         0,
    #         ["COMPRESS=LZW", "PREDICTOR=2"],
    #     )
    #     ds.GetRasterBand(1).WriteArray(array0)
    #     ds.FlushCache()  # Write to disk.
    #     ds = None
    #     im = ax1[k].imshow(array0, cmap="gist_earth_r", vmin=np.quantile(S_mean, 0.01), vmax=np.quantile(S_mean, 0.99))
    #     ax1[k].set_title(basename[ab_range[0]:ab_range[1]])
    #     ax1[k].set_axis_off()
    # fig.subplots_adjust(right=0.85)
    # cbar_ax = fig.add_axes([0.88, 0.2, 0.03, 0.6])
    # cbar = fig.colorbar(im, cax=cbar_ax)
    # cbar.ax.set_ylabel("S Term")

    in0 = None

    # return fig


def colormap_plot_agb_sm_prediction(
    wd, out_radar_list, mask_file, param_file, out_name, ab_range, sm0=0.2
):
    os.chdir(wd)
    if not os.path.exists("output"):
        os.mkdir("output")
    mdl00 = FieldRetrieval(mask_file, out_name=out_name)

    in0 = gdal.Open(mask_file, gdal.GA_ReadOnly)
    lc0 = in0.GetRasterBand(1).ReadAsArray()
    lc1 = lc0.ravel()

    s1_lst = []
    for j in range(0, len(out_radar_list)):
        in1 = gdal.Open(out_radar_list[j], gdal.GA_ReadOnly)
        arr0 = in1.GetRasterBand(1).ReadAsArray()
        arr1 = arr0.ravel()[lc1 > 0]
        s1_lst.append(arr1)
        in1 = None
    s1_arr = np.stack(s1_lst, axis=-1)
    z0 = s1_arr.reshape(s1_arr.shape[0], 1, len(out_radar_list) // 2, 2)
    print(z0.shape)
    W_mean, S_mean = mdl00.inversion_recursive_ws_sm(
        z0, mask_file, sm0=sm0, param0_file=param_file
    )

    basename = os.path.basename(out_radar_list[0])
    agb_name = f"{out_name}_agb_predictions_mean.tif"
    print(agb_name)
    array0 = in0.GetRasterBand(1).ReadAsArray()
    array0[array0 > 0] = np.mean(W_mean, axis=-1)
    try:
        os.remove(agb_name)
    except OSError:
        pass
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.CreateCopy(
        agb_name,
        in0,
        0,
        ["COMPRESS=LZW", "PREDICTOR=2"],
    )
    ds.GetRasterBand(1).WriteArray(array0)
    ds.FlushCache()  # Write to disk.
    ds = None
    fig, ax = plt.subplots(
        figsize=(5, 4),
    )
    im = ax.imshow(array0, cmap="gist_earth_r", vmin=0, vmax=100)
    ax.set_title("AGB Predictions Mean")
    ax.set_axis_off()
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel("AGB (Mg/ha)")

    fig, ax = plt.subplots(
        nrows=np.ceil(W_mean.shape[1] / 2).astype(int),
        ncols=2,
        figsize=(9, 4 * np.ceil(W_mean.shape[1] / 2)),
        sharex=True,
        sharey=True,
    )
    ax1 = ax.flatten()
    for k in range(len(out_radar_list) // 2):
        basename = os.path.basename(out_radar_list[k * 2])
        agb_name = f"{out_name}_agb_predictions_{basename[ab_range[0]:ab_range[1]]}.tif"
        print(agb_name)
        array0 = in0.GetRasterBand(1).ReadAsArray()
        array0[array0 > 0] = W_mean[:, k]
        try:
            os.remove(agb_name)
        except OSError:
            pass
        driver = gdal.GetDriverByName("GTiff")
        ds = driver.CreateCopy(
            agb_name,
            in0,
            0,
            ["COMPRESS=LZW", "PREDICTOR=2"],
        )
        ds.GetRasterBand(1).WriteArray(array0)
        ds.FlushCache()  # Write to disk.
        ds = None
        im = ax1[k].imshow(
            array0,
            cmap="gist_earth_r",
            vmin=np.quantile(W_mean, 0.01),
            vmax=np.quantile(W_mean, 0.99),
        )
        ax1[k].set_title(basename[ab_range[0] : ab_range[1]])
        ax1[k].set_axis_off()
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.2, 0.03, 0.6])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel("AGB (Mg/ha)")

    fig, ax = plt.subplots(
        nrows=np.ceil(W_mean.shape[1] / 2).astype(int),
        ncols=2,
        figsize=(9, 4 * np.ceil(W_mean.shape[1] / 2)),
        sharex=True,
        sharey=True,
    )
    ax1 = ax.flatten()
    for k in range(len(out_radar_list) // 2):
        basename = os.path.basename(out_radar_list[k * 2])
        agb_name = f"{out_name}_S_predictions_{basename[ab_range[0]:ab_range[1]]}.tif"
        print(agb_name)
        array0 = in0.GetRasterBand(1).ReadAsArray()
        array0[array0 > 0] = S_mean[:, k]
        try:
            os.remove(agb_name)
        except OSError:
            pass
        driver = gdal.GetDriverByName("GTiff")
        ds = driver.CreateCopy(
            agb_name,
            in0,
            0,
            ["COMPRESS=LZW", "PREDICTOR=2"],
        )
        ds.GetRasterBand(1).WriteArray(array0)
        ds.FlushCache()  # Write to disk.
        ds = None
        im = ax1[k].imshow(
            array0,
            cmap="gist_earth_r",
            vmin=np.quantile(S_mean, 0.01),
            vmax=np.quantile(S_mean, 0.99),
        )
        ax1[k].set_title(basename[ab_range[0] : ab_range[1]])
        ax1[k].set_axis_off()
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.2, 0.03, 0.6])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel("S Term")

    in0 = None

    # return fig


def colormap_plot_sm_prediction_wkagb(
    wd,
    out_radar_list,
    out_vwc_list,
    mask_file,
    param_file,
    out_name,
    ab_range,
    sm0=0.2,
    init_weights=[4, 1],
):
    os.chdir(wd)
    if not os.path.exists("output"):
        os.mkdir("output")
    mdl00 = FieldRetrieval(mask_file, out_name=out_name)

    in0 = gdal.Open(mask_file, gdal.GA_ReadOnly)
    lc0 = in0.GetRasterBand(1).ReadAsArray()
    lc1 = lc0.ravel()

    s1_lst = []
    for j in range(0, len(out_radar_list)):
        in1 = gdal.Open(out_radar_list[j], gdal.GA_ReadOnly)
        arr0 = in1.GetRasterBand(1).ReadAsArray()
        arr1 = arr0.ravel()[lc1 > 0]
        s1_lst.append(arr1)
        in1 = None
    s1_arr = np.stack(s1_lst, axis=-1)
    z0 = s1_arr.reshape(s1_arr.shape[0], 1, len(out_radar_list) // 2, 2)

    s1_lst = []
    for j in range(0, len(out_vwc_list)):
        in1 = gdal.Open(out_vwc_list[j], gdal.GA_ReadOnly)
        arr0 = in1.GetRasterBand(1).ReadAsArray()
        arr1 = arr0.ravel()[lc1 > 0]
        s1_lst.append(arr1)
        in1 = None
    s1_arr = np.stack(s1_lst, axis=-1)
    z1 = s1_arr.reshape(s1_arr.shape[0], 1, len(out_vwc_list), 1)

    S_mean = mdl00.inversion_recursive_ws_sm_wkagb(
        z0, z1, mask_file, sm0=sm0, param0_file=param_file, init_weights=init_weights
    )

    fig, ax = plt.subplots(
        nrows=np.ceil(S_mean.shape[1] / 2).astype(int),
        ncols=2,
        figsize=(9, 4 * np.ceil(S_mean.shape[1] / 2)),
        sharex=True,
        sharey=True,
    )
    ax1 = ax.flatten()
    for k in range(len(out_radar_list) // 2):
        basename = os.path.basename(out_radar_list[k * 2])
        agb_name = f"{out_name}_S_predictions_{basename[ab_range[0]:ab_range[1]]}.tif"
        print(agb_name)
        array0 = in0.GetRasterBand(1).ReadAsArray()
        array0[array0 > 0] = S_mean[:, k]
        try:
            os.remove(agb_name)
        except OSError:
            pass
        driver = gdal.GetDriverByName("GTiff")
        ds = driver.CreateCopy(
            agb_name,
            in0,
            0,
            ["COMPRESS=LZW", "PREDICTOR=2"],
        )
        ds.GetRasterBand(1).WriteArray(array0)
        ds.FlushCache()  # Write to disk.
        ds = None
        im = ax1[k].imshow(
            array0,
            cmap="gist_earth_r",
            vmin=np.quantile(S_mean, 0.01),
            vmax=np.quantile(S_mean, 0.99),
        )
        ax1[k].set_title(basename[ab_range[0] : ab_range[1]])
        ax1[k].set_axis_off()
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.2, 0.03, 0.6])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel("S Term")

    in0 = None

    # return fig
