

from cma import purecma as pcma
import sys
import os
from numpy import argsort
from math import log, exp

import sklearn.mixture as skmix # for GMM


def fmin(objective_fct, xstart, sigma,
         args=(),
         maxfevals='1e3 * N**2', ftarget=None,
         verb_disp=100, verb_log=1, verb_save=1000):
    """non-linear non-convex minimization procedure, a functional
    interface to CMA-ES.

    Parameters
    ==========
        `objective_fct`: `callable`
            a function that takes as input a `list` of floats (like
            [3.0, 2.2, 1.1]) and returns a single `float` (a scalar).
            The objective is to find ``x`` with ``objective_fct(x)``
            to be as small as possible.
        `xstart`: `list` or sequence
            list of numbers (like `[3.2, 2, 1]`), initial solution vector,
            its length defines the search space dimension.
        `sigma`: `float`
            initial step-size, standard deviation in any coordinate
        `args`: `tuple` or sequence
            additional (optional) arguments passed to `objective_fct`
        `ftarget`: `float`
            target function value
        `maxfevals`: `int` or `str`
            maximal number of function evaluations, a string
            is evaluated with ``N`` as search space dimension
        `verb_disp`: `int`
            display on console every `verb_disp` iteration, 0 for never
        `verb_log`: `int`
            data logging every `verb_log` iteration, 0 for never
        `verb_save`: `int`
            save logged data every ``verb_save * verb_log`` iteration

    Return
    ======
    The `tuple` (``xmin``:`list`, ``es``:`CMAES`), where ``xmin`` is the
    best seen (evaluated) solution and ``es`` is the correspoding `CMAES`
    instance. Consult ``help(es.result)`` of property `result` for further
    results.

    Example
    =======
    The following example minimizes the function `ff.elli`:

    >>> try: import cma.purecma as purecma
    ... except ImportError: import purecma
    >>> def felli(x):
    ...     return sum(10**(6 * i / (len(x)-1)) * xi**2
    ...                for i, xi in enumerate(x))
    >>> res = purecma.fmin(felli, 3 * [0.5], 0.3, verb_disp=100)  # doctest:+SKIP
    evals: ax-ratio max(std)   f-value
        7:     1.0  3.4e-01  240.2716966
       14:     1.0  3.9e-01  2341.50170536
      700:   247.9  2.4e-01  0.629102574062
     1400:  1185.9  5.3e-07  4.83466373808e-13
     1421:  1131.2  2.9e-07  5.50167024417e-14
    termination by {'tolfun': 1e-12}
    best f-value = 2.72976881789e-14
    solution = [5.284564665206811e-08, 2.4608091035303e-09, -1.3582873173543187e-10]
    >>> print(res[0])  # doctest:+SKIP
    [5.284564665206811e-08, 2.4608091035303e-09, -1.3582873173543187e-10]
    >>> res[1].result[1])  # doctest:+SKIP
    2.72976881789e-14
    >>> res[1].logger.plot()  # doctest:+SKIP

    Details
    =======
    After importing `purecma`, this call:

    >>> es = purecma.fmin(pcma.ff.elli, 10 * [0.5], 0.3, verb_save=0)[1]  # doctest:+SKIP

    and these lines:

    >>> es = purecma.CMAES(10 * [0.5], 0.3)
    >>> es.optimize(purecma.ff.elli, callback=es.logger.add)  # doctest:+SKIP

    do pretty much the same. The `verb_save` parameter to `fmin` adds
    the possibility to plot the saved data *during* the execution from a
    different Python shell like ``pcma.CMAESDataLogger().load().plot()``.
    For example, with ``verb_save == 3`` every third time the logger
    records data they are saved to disk as well.

    :See: `CMAES`, `OOOptimizer`.
    """
    #es = CMAES(xstart, sigma, maxfevals=maxfevals, ftarget=ftarget)
    es = GMM_CMAES(xstart, sigma, maxfevals=maxfevals, ftarget=ftarget) # substitute single gaussian for multiple
    if verb_log:  # prepare data logging
        es.logger = pcma.CMAESDataLogger(verb_log).add(es, force=True)
    while not es.stop():
        X = es.ask()  # get a list of sampled candidate solutions
        fit = [objective_fct(x, *args) for x in X]  # evaluate candidates
        es.tell(X, fit)  # update distribution parameters

    # that's it! The remainder is managing output behavior only.
        es.disp(verb_disp)
        if verb_log:
            if es.counteval / es.params.lam % verb_log < 1:
                es.logger.add(es)
            if verb_save and (es.counteval / es.params.lam
                              % (verb_save * verb_log) < 1):
                es.logger.save()

    if verb_disp:  # do not print by default to allow silent verbosity
        es.disp(1)
        print('termination by', es.stop())
        print('best f-value =', es.result[1])
        print('solution =', es.result[0])
    if verb_log:
        es.logger.add(es, force=True)
        es.logger.save() if verb_save else None
    return [es.best.x if es.best.f < objective_fct(es.xmean) else
            es.xmean, es]


# GMM_CMAES is a CMAES with one difference: 
# the gaussian used is substituted by a multiple gaussian model
class GMM_CMAES(pcma.CMAES):
    
    def __init__(self, xstart, sigma,  # mandatory
                 popsize=pcma.CMAESParameters.default_popsize,
                 ftarget=None,
                 maxfevals='100 * popsize + '  # 100 iterations plus...
                           '150 * (N + 3)**2 * popsize**0.5'):

        pcma.CMAES.__init__(self, xstart, sigma, popsize, ftarget, maxfevals)

        self.mixture = skmix.GaussianMixture(3)
        self.mixture.fit(pcma.CMAES.ask(self))
    
    def ask(self):
        samples, _ = self.mixture.sample(self.params.lam)
        return samples
    

    def tell(self, arx, fitvals):
        """update the evolution paths and the distribution parameters m,
        sigma, and C within CMA-ES.

        Parameters
        ----------
            `arx`: `list` of "row vectors"
                a list of candidate solution vectors, presumably from
                calling `ask`. ``arx[k][i]`` is the i-th element of
                solution vector k.
            `fitvals`: `list`
                the corresponding objective function values, to be
                minimised
        """
        ### bookkeeping and convenience short cuts
        self.counteval += len(fitvals)  # evaluations used within tell
        N = len(self.xmean)
        par = self.params
        xold = self.xmean  # not a copy, xmean is assigned anew later

        ### Sort by fitness
        arx = [arx[k] for k in argsort(fitvals)]  # sorted arx
        self.fitvals = sorted(fitvals)  # used for termination and display only
        self.best.update(arx[0], self.fitvals[0], self.counteval)

        ### recombination, compute new mixture
        self.mixture.fit(arx[0:par.mu]) # currently unweighted
        
        print(self.best.f)
        """
        ### recombination, compute new weighted mean value
        self.xmean = dot(arx[0:par.mu], par.weights[0:par.mu], transpose=True)
        #          = [sum(self.weights[k] * arx[k][i] for k in range(self.mu))
        #                                             for i in range(N)]

        ### Cumulation: update evolution paths
        y = minus(self.xmean, xold)
        z = dot(self.C.invsqrt, y)  # == C**(-1/2) * (xnew - xold)
        csn = (par.cs * (2 - par.cs) * par.mueff)**0.5 / self.sigma
        for i in range(N):  # update evolution path ps
            self.ps[i] = (1 - par.cs) * self.ps[i] + csn * z[i]
        ccn = (par.cc * (2 - par.cc) * par.mueff)**0.5 / self.sigma
        # turn off rank-one accumulation when sigma increases quickly
        hsig = (sum(x**2 for x in self.ps) / N  # ||ps||^2 / N is 1 in expectation
                / (1-(1-par.cs)**(2*self.counteval/par.lam))  # account for initial value of ps
                < 2 + 4./(N+1))  # should be smaller than 2 + ...
        for i in range(N):  # update evolution path pc
            self.pc[i] = (1 - par.cc) * self.pc[i] + ccn * hsig * y[i]

        ### Adapt covariance matrix C
        # minor adjustment for the variance loss from hsig
        c1a = par.c1 * (1 - (1-hsig**2) * par.cc * (2-par.cc))
        self.C.multiply_with(1 - c1a - par.cmu * sum(par.weights))  # C *= 1 - c1 - cmu * sum(w)
        self.C.addouter(self.pc, par.c1)  # C += c1 * pc * pc^T, so-called rank-one update
        for k, wk in enumerate(par.weights):  # so-called rank-mu update
            if wk < 0:  # guaranty positive definiteness
                wk *= N * (self.sigma / self.C.mahalanobis_norm(minus(arx[k], xold)))**2
            self.C.addouter(minus(arx[k], xold),  # C += wk * cmu * dx * dx^T
                            wk * par.cmu / self.sigma**2)
        """        

        ### Adapt step-size sigma
        cn, sum_square_ps = par.cs / par.damps, sum(x**2 for x in self.ps)
        self.sigma *= exp(min(1, cn * (sum_square_ps / N - 1) / 2))
        # self.sigma *= exp(min(1, cn * (sum_square_ps**0.5 / par.chiN - 1)))
        
        

if __name__ == "__main__":
    fmin(pcma.ff.elli, 2 * [0.5], 0.5)