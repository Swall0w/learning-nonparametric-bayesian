##################################################################################
# ノンパラメトリックベイジアンによる正規分布のクラスタリング
# 続・わかりやすいパターン認識の第12章の「クラスタリング法1のアルゴリズム」より
# Python 3 用
##################################################################################

import copy
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from numpy.linalg import det, inv
from scipy.special import gamma
from scipy.stats import wishart, multivariate_normal


# http://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals より共分散行列の楕円の描画
def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


def log_factorial(n):
    """階乗の対数"""
    return sum(math.log(i + 1) for i in range(n))


def log_af(a, n):
    """上昇階乗の対数"""
    return sum(math.log(a + i) for i in range(n))


def log_p_s(alpha, n_i, c, n):
    """イーウェンスの抽出公式の対数"""
    return c * math.log(alpha) + sum(log_factorial(n_i[i] - 1) for i in range(c)) - log_af(alpha, n)


def p_x_theta(x, theta):
    """(12.27) の式"""
    mu, Lambda = theta
    return multivariate_normal.pdf(x, mu, inv(Lambda))


def g0(theta, mu0, beta, nu, S):
    """(12.32) の式"""
    mu, Lambda = theta
    return multivariate_normal.pdf(mu, mu0, inv(beta * Lambda)) * wishart.pdf(Lambda, nu, S)


def test1():
    """クラスタリング法1のアルゴリズム"""
    # データの生成
    data_count = 500
    data = np.zeros([5, data_count // 5, 2])
    for cc in range(data.shape[0]):
        centers = [[0.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, 0.0], [1.0, 1.0]][cc]
        data[cc] = np.random.randn(data.shape[1], 2) + np.array(centers) * 10 - 5
    data = data.reshape([-1, 2])
    n = len(data)

    # (12.33), (12.34) のハイパーパラメータ（少し改ざん）
    alpha = 3  # 書籍は1
    beta = 1 / 3
    nu = 3  # 書籍は15
    S = np.array([[0.1, 0], [0, 0.1]])
    S_inv = inv(S)

    # 初期設定
    s = np.zeros([n], dtype=int)
    c = 1
    n_i = np.array([n], dtype=int)
    mu0 = data.mean(axis=0)
    Lambda_all = inv(np.cov(data.T))
    thetas = [(mu0, Lambda_all)]
    p_max = -np.inf
    best_s = s
    best_thetas = thetas

    # (12.35) の不変部分
    v12_35 = beta / ((1 + beta) * np.pi) * gamma((nu + 1) / 2) / (det(S) ** (nu / 2) * gamma((nu + 1 - 2) / 2))

    for iteration in range(100):
        # 所属クラスタの更新
        for k in range(n):
            Sb = inv(S_inv + beta / (1 + beta) * np.outer(data[k] - mu0, data[k] - mu0))
            probs = [(n_i[i] - (1 if s[k] == i else 0)) * p_x_theta(data[k], thetas[i]) for i in range(c)]
            probs.append(alpha * v12_35 * det(Sb) ** ((nu + 1) / 2))
            probs = np.array(probs, dtype=float)
            probs /= probs.sum()

            new_c = np.random.choice(c + 1, p=probs)
            old_c = s[k]
            s[k] = new_c

            if new_c == c:
                n_i = np.concatenate((n_i, [1]))
                thetas.append((data[k], Lambda_all))
                c += 1
            else:
                n_i[new_c] += 1

            n_i[old_c] -= 1
            if n_i[old_c] == 0:
                n_i = np.delete(n_i, old_c)
                del thetas[old_c]
                c -= 1
                s[np.where(s >= old_c)] -= 1

        # 各クラスタのパラメータの更新
        x_bar = np.zeros([c, 2])
        mu_c = np.zeros([c, 2])
        for k in range(n):
            x_bar[s[k]] += data[k]
        for i in range(c):
            mu_c[i] = (x_bar[i] + beta * mu0) / (n_i[i] + beta)
            x_bar[i] /= n_i[i]

        Sq_inv = np.zeros((c,) + S.shape) + S_inv
        for i in range(c):
            Sq_inv[i] += n_i[i] * beta / (n_i[i] + beta) * np.outer(x_bar[i] - mu0, x_bar[i] - mu0)
        for k in range(n):
            Sq_inv[s[k]] += np.outer(data[k] - x_bar[s[k]], data[k] - x_bar[s[k]])

        for i in range(c):
            nu_c = nu + n_i[i]
            Lambda_i = wishart.rvs(nu_c, inv(Sq_inv[i]))
            Lambda_c = (n_i[i] + beta) * Lambda_i
            mu_i = multivariate_normal.rvs(mu_c[i], inv(Lambda_c))
            thetas[i] = (mu_i, Lambda_i)

        # 事後確率最大化（対数を取っています）
        v = log_p_s(alpha, n_i, c, n)
        v += sum(math.log(g0(theta, mu0, beta, nu, S)) for theta in thetas)
        v += sum(math.log(p_x_theta(data[k], thetas[s[k]])) for k in range(n))
        if v > p_max:
            p_max = v
            best_s = copy.copy(s)
            best_thetas = copy.copy(thetas)
            print("p_max", p_max)
            for i in range(c):
                print("\t", thetas[i][0])

    # Matplotlib で表示
    plt.plot(data[:, 0], data[:, 1], "ro")
    for i in range(len(best_thetas)):
        plot_cov_ellipse(best_thetas[i][1], best_thetas[i][0])
    plt.show()

if __name__ == '__main__':
    test1()
