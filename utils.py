import numpy as np
from scipy import sparse
import torch

def create_sparse_vector(rows=1000, density=0.2,seed=0):
    S = sparse.random(m=rows, n=1, density=density, format='coo', dtype=None, random_state=seed, data_rvs=None)
    sparse_vec = S.A
    beta = [sparse_vec != 0][0].astype(int)
    return beta

def create_conditional_gauss(X, j, mu=None, sigma=None):
    a = np.delete(X, j, 1)
    if mu is None:
        mu = np.mean(X, axis=0)
    mu_1 = np.array([mu[j]])
    mu_2 = np.delete(mu, j, 0)
    if sigma is None:
        sigma = np.cov(X.T)
    sigma_11 = sigma[j, j]
    sigma_12 = np.delete(sigma, j, 1)[j, :]
    sigma_21 = np.delete(sigma, j, 0)[:, j]
    sigma_22 = np.delete(np.delete(sigma, j, 0), j, 1)
    mu_bar_vec = []
    sigma12_22 = sigma_12 @ np.linalg.inv(sigma_22)
    sigma_bar = sigma_11 - sigma12_22 @ sigma_21
    for a_i in a:
        mu_bar = mu_1 + sigma12_22 @ (a_i - mu_2)
        mu_bar_vec.append(mu_bar)

    return mu_bar_vec, np.sqrt(sigma_bar)
    

def create_normal_noise(mu, sigma, shape):
    return np.random.normal(mu, sigma, (shape[0], shape[1]))

def create_labels(X, beta):
    n, p = X.shape
    v = create_normal_noise(mu=0, sigma=1, shape=(n, 1))
    Y = X @ beta + v
    return Y


def generate_fake_data(X, X_mu, X_Sigma, j):
    n, p = X.shape
    mu_tilda, sigma_tilda = create_conditional_gauss(X, j, X_mu, X_Sigma)
    Xj_tilda = create_normal_noise(mu=mu_tilda, sigma=sigma_tilda, shape=(n, 1))
    X_tilda = X.copy()
    X_tilda[:, j] = Xj_tilda.ravel()
    return X_tilda, mu_tilda, sigma_tilda

def generate_conditional_data(X, X_mu, X_Sigma, ftrs_=None):
    n, p = X.shape
    ftrs = range(p) if ftrs_ is None else ftrs_
    if X_mu is None:
        X_mu = np.mean(X, axis=0)
    if X_Sigma is None:
        X_Sigma = np.cov(X.T)
    X_tilda = X.copy()
    for j in ftrs:  # range(p):
        mu_tilda, sigma_tilda = create_conditional_gauss(X, j, X_mu, X_Sigma)
        Xj_tilda = create_normal_noise(mu=mu_tilda, sigma=sigma_tilda, shape=(n, 1))
        X_tilda[:, j] = Xj_tilda.ravel().copy()
    return X_tilda


def create_AR1_Sigma(p, rho=0.5):
    Sigma = np.eye(p)
    for i in range(p):
        for j in range(p):
            Sigma[i, j] = rho ** (abs(i - j))
    mu = np.zeros((p, 1)).ravel()
    return mu, Sigma


def hrt_gauss(model, t, X_test, Y_test, j, mu=None, Sigma=None,
              is_iid=False, K=1000, is_NN=False,scaler=None):
    if scaler is not None:
      X_test=scaler.inverse_transform(X_test)
    n = X_test.shape[0]
    cnt = 0
    if is_iid:
        mu_tilda = 0
        sigma_tilda = 1
    else:
        mu_tilda, sigma_tilda = create_conditional_gauss(X_test, j, mu, Sigma)
    for k in range(K):
        Xj_tilda = create_normal_noise(mu=mu_tilda, sigma=sigma_tilda, shape=(n, 1))
        X_tilda = X_test.copy()
        X_tilda[:, j] = Xj_tilda.ravel()
        if scaler is not None:
          X_tilda=scaler.transform(X_tilda)
        if is_NN:
            model.eval()
            t_tilda = ((model.predict(X_tilda).ravel()-Y_test.ravel())**2).mean()
        else:
            t_tilda = ((Y_test.ravel() - model.predict(X_tilda).ravel()) ** 2).mean()

        if t_tilda <= t:
            cnt += 1
    return (1 + cnt) / float(1 + K)  # P_val_hat


def bh(p, fdr):
    p_orders = np.argsort(p)
    discoveries = []
    m = float(len(p_orders))
    for k, s in enumerate(p_orders):
        if p[s] <= (k + 1) / m * fdr:
            discoveries.append(s)
        else:
            break
    return np.array(discoveries, dtype=int)


def bh_predictions(p_values, fdr_threshold):
    reshape = False
    if len(p_values.shape) > 1:
        reshape = True
        p_shape = p_values.shape
        p_values = p_values.flatten()
    pred = np.zeros(len(p_values), dtype=int)
    disc = bh(p_values, fdr_threshold)
    if len(disc) > 0:
        pred[disc] = 1
    if reshape:
        pred = pred.reshape(p_shape)
    return pred


def calc_power(discoveries, beta):
    denom = np.sum([beta != 0])
    nom = 0
    for i, discovery in enumerate(discoveries):
        if discovery == 1 and beta[i] != 0:
            nom += 1
    return nom / float(denom)


def calc_FDR(discoveries, beta):
    denom = np.sum([discoveries != 0])  
    nom = 0
    for i, discovery in enumerate(discoveries):
        if discovery == 1 and beta[i] == 0:
            nom += 1
    if denom == 0:
        return 0
    return nom / float(denom)