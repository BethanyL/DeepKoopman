import numpy as np
from scipy.special import factorial2
from scipy.special import factorial


def kfn(theta0):
    return np.sin(np.float64(theta0) / 2)


def periodTerm(n, k):
    numerator = np.float64((factorial2(2 * n - 1)) ** 2)
    denominator = np.float64((2 ** n) * factorial(n) * factorial2(2 * n))
    return (numerator / denominator) * (k ** (2 * n)) * (2 * np.pi)


def periodTermNextStep(n, k, prevTerm):
    nextTerm = ((2 * n - 1) ** 2) * prevTerm / (4 * n ** 2) * k ** 2
    return nextTerm


def periodPendulum(theta0, tol, maxN, printFlag=0):
    periodApprox = 2 * np.pi  # n = 0 term
    k = kfn(theta0)
    for n in (np.arange(maxN) + 1):  # n = 1, 2, ..., maxN
        if n < 3:
            nextTerm = periodTerm(n, k)
        else:
            # nextTermOldFormula = periodTerm(n, k)
            nextTerm = periodTermNextStep(n, k, prevTerm)
            # if (np.abs(nextTermOldFormula - nextTerm) > 10**(-10)):
            #    print "discrepancy at n = %f, k = %f: prev. formula said nextTerm = %f, new formula says nextTerm = %f" % (n, k, nextTermOldFormula, nextTerm)
        if nextTerm < 0:
            print("nextTerm < 0: %f at n = %f, k = %f" % (nextTerm, n, k))
        if nextTerm > tol:  # nextTerm gives sense of error (really, lower bound, since all terms are non-neg)
            periodApprox += nextTerm
        else:
            if printFlag:
                print("reached tol (lower bound on error) after n = %d" % (n - 1))
            break
        prevTerm = nextTerm.copy()
    return periodApprox


def FindTheta0(theta, thetadot):
    # print "find theta0 for theta = %f and thetadot = %f" % (theta, thetadot)
    potential = (1.0 / 2.0) * (thetadot ** 2) - np.cos(theta) + 1  # H
    # E = 1 - np.cos(theta0)
    # domain for real #s for arrcos is [-1,1]
    # so want 1-potential in [-1, 1]
    # so want potential in [0, 2]
    if ((potential < 0) or (potential > 2)):
        # TODO: do something smarter here 
        potential = 0
    theta0 = np.arccos(1 - potential)
    return theta0


# def f(t, x):
#    return [x[1], -np.sin(x[0])]

def FindOmega(point, tol=10 ** (-7), maxTerms=100):
    theta0 = FindTheta0(point[0], point[1])
    period = periodPendulum(theta0, tol, maxTerms)
    omega = (2 * np.pi) / period
    # eigenval = np.exp(omega*1j*deltat)
    return omega


def AddFrequency(prefix, suffix, tol=10 ** (-7), maxTerms=100):
    fname = prefix + suffix
    print("loading %s" % fname)
    data = np.loadtxt(fname, delimiter=',')
    # each row is an example, we add extra column
    data_freq = np.zeros((data.shape[0], data.shape[1] + 1), dtype=np.float32)
    data_freq[:, 0:2] = data.copy()
    for j in range(data.shape[0]):
        data_freq[j, 2] = FindOmega(data[j, 0:2])
    newfname = prefix + 'Freq' + suffix
    print("saving %s" % newfname)
    np.savetxt(newfname, data_freq, delimiter=',')
