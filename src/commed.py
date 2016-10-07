# coding: utf-8

from __future__ import with_statement, print_function
import numpy as np
import scipy as sp
import warnings
from collections import deque
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC, LinearSVC

class CMVmedSolver_binary:
    """
    
         The Consensus-based Multi-View maximum entropy discrimination solver for binary classification
    
         :param Xtrains_l: a list of labeled train sets. 
         :type Xtrains_l: list of length nVs = number of views
             :param Xtrains_l[k]: the labeled data for view k
             :type Xtrain_l[k]: numpy.ndarray with shape [n_labels, n_dims[k]]
         :param  Xtrains_u: a list of unlabeled train sets.
         :type Xtrains_u: the same length as Xtrain_l
             :param Xtrains_u[k]: numpy.adarray with shape [n_ulabels, n_dims[k]]
         :param  Xtests: a list of test sets.
         :type Xtests: the same length as Xtrain_l
             :param Xtests[k]: numpy.adarray with shape [n_test, n_dims[k]]
         :param Y: numpy.array [n_labels,]
         :param options: dictionary of solver parameters
             :param max_iters: max iteration for the solver
             :param tol: tolerance value for MAP iterations
             :param kern: kernel methods
                    :param kern_name: ={'linear', 'rbf'} by default = 'rbf'
                    :param kern_param: a dictionary that depends on the kernel method
    
    """

    def __init__(self, Xtrains_l, Y, Xtrains_u, **kwargs):
        assert len(Xtrains_l) > 0 and len(Xtrains_u) > 0, 'The input data not empty!'
        assert len(Xtrains_l) == len(Xtrains_u), 'The number of views should match.'
        self.nVs = len(Xtrains_l)
        self.n_dims = [ d.shape[1] for d in Xtrains_l ]
        self.n_labels, _ = Xtrains_l.shape
        for d in Xtrains_l:
            assert d.shape[0] == self.n_labels, 'The number of samples for all views should be the same.'

        self.Xtrains_l = Xtrains_l
        self.n_ulabels, _ = Xtrains_u.shape
        for k, d in enumerate(Xtrains_u):
            assert d.shape[0] == self.n_ulabels, 'The number of samples for all views should be the same.'
            assert d.shape[1] == self.n_dims[k], 'The number of dimensions for unlabeled and labeled samples should be the same'

        self.Xtrains_u = Xtrains_u
        self.n_total = self.n_labels + self.n_ulabels
        assert self.n_labels == Y.shape[0], 'The number of labels should match.'
        self.Y = Y
        self.max_iters = kwargs.get('max_iters')
        if self.max_iters is None:
            self.max_iters = 300
        self.max_iters = kwargs.get('tol')
        if self.tol is None:
            self.tol = 0.0001
        self.kern = kwargs.get('kern')
        if self.kern is None:
            self.kern = rbf_kernel
            self.kern_name = 'rbf'
            self.kern_param = {'gamma': 1}
        else:
            self.kern_name = kwargs.get('kern_name')
            self.kern_param = kwargs.get('kern_param')
            if self.kern_name is None or self.kern_param is None:
                raise KeyError('Not found the name of kernel or the kernel parameters')
            elif not isinstance(self.kern_param, dict):
                raise TypeError('The kernel parameter is in a dictionary format')
        self.alpha = kwargs.get('alpha')
        if self.alpha is None:
            self.alpha = 1
        self.subsample = kwargs.get('subsample')
        if self.subsample is None:
            self.subsample = 0.7
        self.n_subsample = np.ceil(self.label * self.subsample)
        self.models_single = []
        self.flag_pre_train = False
        self.result = np.zeros([n_total])
        self.error_var = np.zeros([n_total])
        self.pseudo_label = np.zeros([n_ulabels])
        self.full_label = np.zeros([n_total])
        self.approx_cross_entropy_loss = np.zeros([n_ulabels])
        self.queue_hist_likelihoods = deque([])
        self.queue_hist_dual_variables = deque([])
        self.queue_hist_predict_local = deque([])
        self.queue_hist_prodict_joint = deque([])

    def optimize():
        """
            The main procedure for CMV-MED solver
        """
        if not self.flag_pre_train:
            self.precompute()
        for i in range(max_iters):
            print('Step {0:4d}: '.format(i + 1))

    def quadratic_optimizor():
        """
            The main solver for the quadratic programming problem in CMV-MED 
        
            
        
        """
        pass

    def consensus_view(self, Xs, **kwargs):
        """
        
            Construct the consensus view function by merging nVs different functions 
            :param nVs: the number of different views
            :param *args: nVs different view-specific log-likelihood functions 
            :param **kwargs: auxilary parameters for log-likelihood functions
        """
        assert len(Xs) == self.nVs, 'nV = len of Xs.'
        if not self.flag_pre_train:
            print('Precompute single view p.d.f. first')
            self.precompute()
        weight = kwargs.get('weight')
        if weight is None:
            weight = np.ones([self.nVs]) / self.nV
        n, _ = Xs[0].shape
        y_prob = np.zeros([n, 2])
        for k, d in enumerate(Xs):
            y_prob = y_prob + loglikelihood_binary(self.models_single[k].decision_function, d, n) / weight[k]

        return (y_prob, np.exp(y_prob))

    def predict(self, Xtests):
        """
         
           Prediction on Test datasets
        """
        assert len(Xtests) == self.nVs, 'The number of views should match.'
        n_test, _ = Xtests[0].shape
        for k, d in enumerate(Xtests):
            assert d.shape[0] == n_test, 'The number of samples for all views should be the same.'
            assert d.shape[1] == self.n_dims[k], 'The number of dimensions for training and test samples should be the same'

        y_test = np.zeros([n_test])
        return y_test

    def precompute(self):
        """
           
           Precompute the initial p.d.f. via C-SVM in sklearn package
        """
        self.subIndex = np.random.permutation(np.arange(self.n_labels))[0:self.n_subsample + 1]
        Xs_sub_l = []
        for k, d in enumerate(self.Xtrains_l):
            Xs_sub_l.append(d[self.subIndex, :])

        Y_sub = Y[self.subIndex]
        self.models_single = []
        self.flag_pre_train = False
        ifExists = True
        if self.kern_name == 'rbf':
            svc_kern_name = 'rbf'
        elif self.kern_name == 'linear':
            svc_kern_name = 'linear'
        elif self.kern_name == 'poly':
            svc_kern_name = 'poly'
        else:
            ifExists = False
        print('Initialization:')
        for k, ds in enumerate(Xs_sub_l):
            if self.kern_name == 'rbf' or self.kern_name == 'poly':
                gamma = self.kern_param['gamma']
                if self.kern_name == 'poly':
                    try:
                        degree = self.kern_param['degree']
                    except KeyError:
                        degree = 3

                    svc_model = SVC(kernel=svc_kern_name, gamma=gamma, degree=degree)
                else:
                    svc_model = SVC(kernel=svc_kern_name, gamma=gamma)
            elif self.kern_name == 'linear':
                svc_model = LinearSVC()
            else:
                svc_model = SVC(kernel=self.kern)
            svc_model.fit(ds, Y_sub)
            self.models_single.append(svc_model)

        self.flag_pre_train = True


def loglikelihood_binary(fun, X, n):
    """
        return the p.d.f. for binary classifer. It is a sigmoid(fun) in essence
    """
    loglikelihood = np.zeros([n, 2])
    loglikelihood[:, 0] = 0.5 * np.ones([n]) * fun(X) - np.log(2 * np.cosh(0.5 * fun(X)))
    loglikelihood[:, 1] = -0.5 * np.ones([n]) * fun(X) - np.log(2 * np.cosh(0.5 * fun(X)))
    return loglikelihood


def compute_approx_cross_entropy(x, x_ref, y_ref):
    off_set = -np.log(2 * np.cosh(0.5 * x_ref)) + 0.5 * sigmoid(x_ref) * x_ref
    linear = 0.5 * dsigmoid(x_ref, y_ref) * x
    quadratic = -1 / 8 * sigmoid2(x_ref) * (x - x_ref) ** 2
    return off_set + linear + quadratic


def sigmoid(x):
    return np.tanh(0.5 * x)


def dsigmoid(x, y):
    r"""
       compute \hat{y} - sigmoid(x)
    """
    return y - np.tanh(0.5 * x)


def sigmoid2(x):
    """
        1- sigmoid(x)**2
    """
    return 1 - np.tanh(0.5 * x) ** 2
