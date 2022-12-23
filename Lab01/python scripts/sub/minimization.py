import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error


class SolveMinProbl:
    """ 
    Super-class which implements the methods shared by all the 
    regression algorithms.
    
    Attributes
    ----------
    X_train : matrix with the training features
    y_train : vector of the training regressands
    X_val : matrix with the validation features
    y_val : vector of the validation regressands
    gamma : learning coefficient
    Nit : number of iterations
    Np : number of measurements (rows of X's and y's)
    Nf : number of features (columns of X's)
    sol : vector of the weight vector
    err_train : matrix of the mean square error for the train set;
                the 1st column stores the iteration step, the 2nd one the mean square error
    err_val : matrix of the mean square error for the val set;
              the 1st column stores the iteration step, the 2nd one the mean square error
    
    """

    def __init__(self, y_train, X_train, y_val=0, X_val=0, gamma=0, Nit=0):
        """Method for the initialization of the class attributes"""
        self.X_train = X_train
        self.y_train = y_train
        self.y_val = y_val
        self.X_val = X_val
        self.gamma = gamma
        self.Nit = Nit
        self.Np = y_train.shape[0]
        self.Nf = X_train.shape[1]
        self.sol = np.zeros((self.Nf, 1), dtype=float)
        self.err_train = np.zeros((Nit, 2), dtype=float)
        self.err_val = np.zeros((Nit, 2), dtype=float)
        return

    def plot_w(self, title='Weight vector'):
        """Method to plot the vector w"""
        w = self.sol
        n = np.arange(self.Nf)
        plt.figure()
        plt.plot(n, w, marker='o', markerfacecolor='red')
        plt.xlabel('n')
        plt.ylabel('w(n)')
        plt.title(title)
        plt.axhline(lw=0.5, color='black')
        plt.show()
        return

    def plot_y(self, yhat, y, ylabel, xlabel, title='y vector'):
        """Method for the plot of yhat vs y"""
        sns.set(style="darkgrid")
        d = pd.DataFrame(np.hstack([y, yhat]), columns=[xlabel, ylabel])
        sns.jointplot(data=d,
                      x=xlabel,
                      y=ylabel,
                      kind='reg',
                      marginal_kws=dict(bins=50, color='red'),
                      scatter_kws=dict(s=6, alpha=.2, color="red"))
        plt.annotate(title + ':\n' + ylabel + ' vs ' + xlabel, (0, 55))
        plt.savefig('./Figures/' + title + '_plot_' + ylabel + 'vs' + xlabel + '.png', dpi=300)

    def hist_y(self, yhat, y, label1, label2, title='y histogram'):
        """Method to plot the histograms of y - yhat"""
        plt.figure()
        plt.hist(y - yhat, bins=50)
        plt.xlabel('error: ' + label2 + ' - ' + label1)
        plt.ylabel('frequency')
        plt.title(title + ': histogram of ' + label2 + ' - ' + label1)
        plt.savefig('./Figures/' + title + '_histogram_' + label2 + '-' + label1 + '.png', dpi=300)

    def plot_err(self, title, logy, logx):
        """Method to plot the mean square error of the train and val sets"""
        err_train = self.err_train
        err_val = self.err_val
        plt.figure()
        if (logy == 0) & (logx == 0):
            plt.plot(err_train[:, 0], err_train[:, 1], label='Train set error')
            plt.plot(err_val[:, 0], err_val[:, 1], label='Val set error')
        if (logy == 1) & (logx == 0):
            plt.semilogy(err_train[:, 0], err_train[:, 1], label='Train set error')
            plt.semilogy(err_val[:, 0], err_val[:, 1], label='Val set error')
        if (logy == 0) & (logx == 1):
            plt.semilogx(err_train[:, 0], err_train[:, 1], label='Train set error')
            plt.semilogx(err_val[:, 0], err_val[:, 1], label='Val set error')
        if (logy == 1) & (logx == 1):
            plt.loglog(err_train[:, 0], err_train[:, 1], label='Train set error')
            plt.loglog(err_val[:, 0], err_val[:, 1], label='Val set error')
        plt.legend()
        plt.xlabel('n')
        plt.ylabel('mean square error')
        plt.title(title)
        plt.show()
        return

    def print_result(self, title='Result'):
        """Method to print the values of the vector w"""
        print(title, ': ')
        print('the optimum weight vector is: ')
        print(self.sol)
        return


class SolveLLS(SolveMinProbl):
    """
    Class to perform LLS;
    Inherit all the attributes and methods of the SolveMinProbl class
    """

    def run(self):
        X = self.X_train
        y = self.y_train
        w = np.dot(np.linalg.pinv(X), y)
        self.sol = w
        return w


class SolveGrad(SolveMinProbl):
    """
    Class to perform the gradient algorithm;
    Inherit all the attributes and methods of the SolveMinProbl class
    """

    def run(self):
        X = self.X_train
        y = self.y_train
        X_val = self.X_val
        y_val = self.y_val

        gamma = self.gamma
        w = np.random.rand(self.Nf, 1)

        for it in range(self.Nit):
            grad = 2 * np.dot(X.T, (np.dot(X, w) - y))
            w = w - gamma * grad
            self.err_train[it, 0] = it
            self.err_val[it, 0] = it
            self.err_train[it, 1] = mean_squared_error(np.dot(X, w), y)
            self.err_val[it, 1] = mean_squared_error(np.dot(X_val, w), y_val)

        self.sol = w
        return w


class SolveSteepDesc(SolveMinProbl):
    """
    Class to perform the steepest descent;
    Inherit all the attributes and methods of the SolveMinProbl class
    """

    def run(self):
        X = self.X_train
        y = self.y_train
        X_val = self.X_val
        y_val = self.y_val

        H = 2 * np.dot(X.T, X)  # Hessian matrix
        w = np.random.rand(self.Nf, 1)

        for it in range(self.Nit):
            grad = 2 * np.dot(X.T, (np.dot(X, w) - y))
            gamma = np.linalg.norm(grad) ** 2 / np.dot(np.dot(grad.T, H), grad)
            w = w - gamma * grad
            self.err_train[it, 0] = it
            self.err_val[it, 0] = it
            self.err_train[it, 1] = mean_squared_error(np.dot(X, w), y)
            self.err_val[it, 1] = mean_squared_error(np.dot(X_val, w), y_val)

        self.sol = w
        return w


class SolveStochGrad(SolveMinProbl):
    """
    Class to perform the stochastic gradient;
    Inherit all the attributes and methods of the SolveMinProbl class
    """

    def run(self):
        X = self.X_train
        y = self.y_train
        X_val = self.X_val
        y_val = self.y_val

        gamma = self.gamma
        w = np.random.rand(self.Nf, 1)

        for it in range(int(self.Nit / self.Np)):
            for i in range(self.Np):
                x = X[i]
                x.shape = (x.size, 1)
                grad = (np.dot(x.T, w) - y[i]) * x
                w = w - gamma * grad
            self.err_train[it, 0] = it
            self.err_val[it, 0] = it
            self.err_train[it, 1] = mean_squared_error(np.dot(X, w), y)
            self.err_val[it, 1] = mean_squared_error(np.dot(X_val, w), y_val)

        self.sol = w
        return w


class SolveConjGrad(SolveMinProbl):
    """
    Class to perform the conjugate gradient;
    Inherit all the attributes and methods of the SolveMinProbl class
    """

    def run(self):
        X = self.X_train
        y = self.y_train
        Q = np.dot(X.T, X)
        self.err_train = np.zeros((self.Nf, 2), dtype=float)
        b = np.dot(X.T, y)
        w = np.zeros((self.Nf, 1), dtype=float)
        d = b
        g = -b

        for it in range(self.Nf):
            a = -np.dot(d.T, g) / np.dot(np.dot(d.T, Q), d)
            w = w + a * d
            g = np.dot(Q, w) - b
            beta = np.dot(np.dot(g.T, Q), d) / np.dot(np.dot(d.T, Q), d)
            d = -g + beta * d
            self.err_train[it, 0] = it
            self.err_train[it, 1] = mean_squared_error(np.dot(X, w), y)

        self.sol = w
        return w


class SolveRidge(SolveMinProbl):
    """
    Class to perform the ridge regression;
    Inherit all the attributes and methods of the SolveMinProbl class
    """

    def run(self):
        X = self.X_train
        y = self.y_train
        X_val = self.X_val
        y_val = self.y_val

        I = np.identity(self.Nf)
        self.err_train = np.zeros((5, 2), dtype=float)
        self.err_val = np.zeros((5, 2), dtype=float)

        for lamb in range(5):
            w = np.linalg.inv(np.dot(X.T, X) + lamb * I).dot(X.T).dot(y)
            self.err_train[lamb, 0] = lamb
            self.err_val[lamb, 0] = lamb
            self.err_train[lamb, 1] = mean_squared_error(np.dot(X, w), y)
            self.err_val[lamb, 1] = mean_squared_error(np.dot(X_val, w), y_val)

        self.sol = w
        return w
