import numpy, scipy.linalg, scipy.optimize

# angular correlation function of systematic errors (two possible choices):
# x is the angular distance between two points in degrees,
# return the covariance of each component of proper motion in (mas/yr)^2.
# non-oscillatory choice (conservative):
covfnc1 = lambda x:  0.0036* numpy.exp(-x / 0.25      ) + 0.0008 * numpy.exp(-x / 20.)
# oscillatory choice (more optimistic):
covfnc2 = lambda x:  0.004 * numpy.sinc(x / 0.5 + 0.25) + 0.0008 * numpy.exp(-x / 20.)


def angular_distance(ra0, dec0, ra1, dec1):
    '''
    Compute the angular distance between two points on a sphere (coordinates expressed in degrees)
    '''
    d2r = numpy.pi/180  # degrees to radians
    return 2 * numpy.arcsin( (numpy.sin( (dec0-dec1)*0.5 * d2r )**2 +
        numpy.cos(dec0 * d2r) * numpy.cos(dec1 * d2r) * numpy.sin( (ra0-ra1)*0.5 * d2r )**2 )**0.5 ) / d2r


def get_mean_pm(ra, dec, pmra, pmdec, pmra_error, pmdec_error, pmra_pmdec_corr, sigma=0, rsigma=numpy.inf, covfnc=None):
    '''
    Compute the mean proper motion (PM) and possibly the internal PM dispersion for a cluster of stars.

    Input consists of several arrays of length N (the number of stars), as given in the Gaia catalogue:
    ra -- right ascension (azimuthal angle, or longitude) of stars, measured in degrees
    dec -- declination (latitude), measured in degrees [-90 to +90]
    pmra -- PM in the ra direction: d(ra)/dt * cos(dec), measured in mas/yr
    pmdec -- PM in the dec direction
    pmra_error -- statistical uncertainty of pmra [mas/yr]
    pmdec_error -- same for pmdec
    pmra_pmdec_corr -- correlation coefficient between errors in pmra and pmdec [-1 to 1]

    Additional input parameters:
    sigma -- internal PM dispersion. If None, then this quantity will be inferred from the input data,
    if the value is provided, it will be used without modifications.
    rsigma -- spatial radius of the internal PM dispersion profile: sigma(R) = sigma / (1 + (R/rsigma)**2)**0.25,
    where R is the angular distance from the cluster center. If set to infinity, the internal dispersion
    is assumed to be the same everywhere, otherwise it declines with radius.
    covfnc -- spatial covariance function of systematic errors.
    If None, the errors are assumed to be uncorrelated; possible choices are covfnc1 or covfnc2.

    Output is a tuple with the following values:
    mean_pmra, mean_pmdec -- the components of the mean PM of the cluster center-of-mass;
    mean_pmra_error, mean_pmdec_error -- uncertainties on these mean PM;
    mean_pmra_pmdec_corr -- correlation coefficient of the uncertainties;
    sigma_value -- inferred internal PM dispersion (if sigma==None), or a copy of the input sigma if it was provided;
    sigma_error -- uncertainty on sigma_value, or 0 if it was fixed to the input value.
    '''

    nstar  = len(ra)
    pmboth = numpy.column_stack((pmra, pmdec)).reshape(-1)  # interleaved pmra,pmdec for each star

    meanra = numpy.median(ra)
    meandec= numpy.median(dec)
    # assumed functional form of PM dispersion (may be arbitrary, here use a particular choice;
    # note that the amplitude of dispersion may be a free parameter, only the spatial profile is fixed
    sig2mul= 1 / (1 + angular_distance(ra, dec, meanra, meandec)**2 / rsigma**2)**0.5

    if sigma is None and covfnc is None:  # should have a separate (simplified) treatment for this case,
        covfnc = lambda x: x*0            # but instead use the general machinery with covfnc set to zero

    if not covfnc is None:
        # covariance matrix of systematic errors depending on the distance between two stars:
        # [ [ C11  C12  ...  C1n ]
        #   [ C12  C22  ...  C2n ]
        #   [ ...  ...  ...  ... ]
        #   [ C1n  C2n  ...  Cnn ] ]
        # where Cij = covfnc(angular_distance(star_i, star_j))
        covmat = numpy.array([ covfnc( angular_distance(ra, dec, ra0, dec0) )
            for ra0, dec0 in zip(ra, dec) ])

        # convert to a 2x larger matrix with diagonal 2x2 blocks (checkerboard pattern):
        # [ [ C11   0   C12   0   ...  C1n   0  ]
        #   [  0   C11   0   C12  ...   0   C1n ]
        #   [ C12   0   C22   0   ...  C2n   0  ]
        #   [  0   C12   0   C22  ...   0   C2n ]
        #   [ ...  ...  ...  ...  ...  ...  ... ]
        #   [ C1n   0   C2n   0   ...  Cnn   0  ]
        #   [  0   C1n   0   C2n  ...   0   Cnn ] ]
        covmat2= numpy.kron(covmat, numpy.eye(2))

        # add 2x2 blocks along the main diagonal:
        # [ [ E1xx E1xy  0    0   ...   0    0  ]
        #   [ E1xy E1yy  0    0   ...   0    0  ]
        #   [  0    0   E2xx E2xy ...   0    0  ]
        #   [ ...  ...  ...  ...  ...  ...  ... ]
        #   [  0    0    0    0   ...  Enxy Enyy] ]
        # where the the 2x2 blocks are the covariance matrices of random errors
        # for each star, and optionally the internal velocity dispersion (sigma):
        # [ [  pmra_err^2 + sigma^2          corr * pmra_err * pmdec_err ]
        #   [  corr * pmra_err * pmdec_err   pmdec_err^2 + sigma^2       ] ]

        # distance between two subsequent diagonal elements in the flattened corrmat2
        stride = 2*(2*nstar+1)
        covmat2.flat[        0::stride] += pmra_error **2
        covmat2.flat[2*nstar+1::stride] += pmdec_error**2
        covmat2.flat[        1::stride] += pmra_error * pmdec_error * pmra_pmdec_corr
        covmat2.flat[2*nstar  ::stride] += pmra_error * pmdec_error * pmra_pmdec_corr
        P = numpy.tile(numpy.eye(2), (nstar,1))   # [ [1,0], [0,1], [1,0], [0,1], ... ]

        if not sigma is None:
            covmat2.flat[::stride//2] += numpy.repeat(sigma**2 * sig2mul, 2)

            # obtain the Cholesky decomposition of covariance matrix
            chol  = numpy.linalg.cholesky(covmat2)
            CinvP = scipy.linalg.cho_solve((chol, True), P)
            covar = numpy.linalg.inv(numpy.dot(CinvP.T, P))        # covariance of uncertainties in mean pmra,pmdec
            sol   = numpy.dot(covar, numpy.dot(CinvP.T, pmboth))   # mean pmra,pmdec (the solution)
            sigval= sigma  # known value provided as input
            sigerr= 0.     # with no uncertainty

        else:

            Lambda, Q = scipy.linalg.eigh(covmat2, numpy.diag(numpy.repeat(sig2mul, 2)))
            y = numpy.dot(pmboth, Q)
            R = numpy.dot(Q.T, P)

            # if we do not know sigma, we find it from the condition that the derivative of log-likelihood
            # w.r.t. sigma^2 is zero; this means solving a 1d equation for sigma^2, in which for any trial
            # value of sigma^2 we first obtain the solution of a linear system for (meanpmra, meanpmdec),
            # and then substitute it into d(ln L)/d(sigma^2), which is a nonlinear function of sigma^2.
            def rootfindersigma(sigma2):
                D    = (Lambda + sigma2)**-1                 # diag(D) = diag[ 1 / (lambda_k + sigma^2) ]
                RTDR = numpy.einsum('ki,k,kj->ij', R, D, R)  # R^T  diag(D)  R
                RTDy = numpy.einsum('ki,k,k->i',   R, D, y)  # R^T  diag(D)  y
                mubar= numpy.linalg.solve(RTDR, RTDy)
                return numpy.sum( ( (y - numpy.dot(R, mubar)) * D )**2 - D)

            # first need to bracket the root from above (assuming that the lower bracket is at 0)
            maxsigma2 = 1.0
            while rootfindersigma(maxsigma2)>0:
                maxsigma2 *= 4
            # then find the exact root
            try:
                sigma2 = scipy.optimize.brentq(rootfindersigma, 0, maxsigma2)
            except:  # may fail if the best-fit sigma^2 is less than 0, then replace it by 0
                sigma2 = 0.

            # after having found the best-fit sigma, we again obtain the solution for (meanpmra, meanpmdec)
            # from a linear system, and then compute the hessian of -ln(L) w.r.t. all three parameters
            # (meanpmra, meanpmdec, sigma).
            # The inverse of this hessian is the covariance matrix for the best-fit parameters.
            D      = (Lambda + sigma2)**-1
            RTDR   = numpy.einsum('ki,k,kj->ij', R, D, R)
            RTDy   = numpy.einsum('ki,k,k->i',   R, D, y)
            sol    = numpy.linalg.solve(RTDR, RTDy)  # inferred mean PM of the cluster
            sigval = sigma2**0.5                     # inferred internal PM dispersion
            z      = y - numpy.dot(R, sol)           # difference between model prediction and actual PM
            H      = numpy.zeros((3, 3))  # overall hessian
            H[:2,:2] = RTDR               # top-left 2x2 block of the hessian is [ R^T  diag(D)  R ]
            H[2,2] = numpy.sum( 4 * sigma2 * z**2 * D**3 - (2 * sigma2 + z**2) * D**2 + D )  # bottom-right element
            H[2,:2]= numpy.dot(R.T, z * D**2) * 2*sigval   # two remaining values in the bottom row
            H[:2,2]= H[2,:2]                               # ..and symmetrically transposed ones in the right column
            invH   = numpy.linalg.inv(H)
            covar  = invH[0:2, 0:2]    # covariance matrix for meanpmra, meanpmdec
            sigerr = invH[2, 2]**0.5   # standard deviation of sigma (don't output its correlations with meanpmra, meanpmdec)

    else:  # a simpler case with no spatial correlations and known sigma
        covmat = numpy.zeros((nstar, 2, 2))
        covmat[:,0,0] = pmra_error **2 + sigma**2 * sig2mul
        covmat[:,1,1] = pmdec_error**2 + sigma**2 * sig2mul
        covmat[:,0,1] = pmra_error * pmdec_error * pmra_pmdec_corr
        covmat[:,1,0] = pmra_error * pmdec_error * pmra_pmdec_corr
        CinvA = numpy.linalg.inv(covmat).reshape(2*nstar, 2)
        covar = numpy.linalg.inv(numpy.sum(CinvA.reshape(nstar, 2, 2), axis=0))
        sol   = numpy.dot(covar, numpy.dot(CinvA.T, pmboth))
        sigval= sigma
        sigerr= 0.

    return sol[0], sol[1], covar[0,0]**0.5, covar[1,1]**0.5, covar[0,1] / (covar[0,0]*covar[1,1])**0.5, sigval, sigerr


def create_cluster(nstar, radius, sigma=0, covfnc=None):
    '''
    Create a mock cluster of stars.

    Input:
    nstar -- number of stars.
    radius -- radius of the cluster (in degrees); the surface density profile is Gaussian with this radius.
    sigma -- internal PM dispersion of stars in the cluster; its spatial dependence is
    sigma(R) = sigma / (1 + (R/radius)**2)**0.25
    covfnc -- spatial covariance function of systematic errors; if None then no correlated systematic errors are added,
    only the random (statistical) errors with a realistic distribution of magnitudes.

    Output:
    five arrays that may be provided as input for the routine get_mean_pm:
    ra, dec, pmra, pmdec, pmra_error, pmdec_error, pm_corr
    '''
    rsigma     = radius
    ra         = numpy.random.normal(size=nstar) * radius
    dec        = numpy.random.normal(size=nstar) * radius
    sigvar     = sigma / (1 + (ra**2 + dec**2) / rsigma**2)**0.25   # spatially variable dispersion
    pmra       = numpy.random.normal(size=nstar) * sigvar           # true values
    pmdec      = numpy.random.normal(size=nstar) * sigvar
    pm_err_mag = numpy.random.gamma(shape=2, scale=0.5, size=nstar) # size of error ellipse
    pm_err_rat = numpy.random.uniform(0.2, 1, size=nstar)           # axis ratio of --"--
    pm_err_dir = numpy.random.uniform(0, numpy.pi, size=nstar)      # orientation of --"--
    pm_err_maj = pm_err_mag * pm_err_rat**-0.5                      # semimajor axis
    pm_err_min = pm_err_mag * pm_err_rat**0.5                       # semiminor axis
    tan        = numpy.tan(pm_err_dir)
    pmra_error = ( (pm_err_maj**2 + pm_err_min**2 * tan**2) / (1 + tan**2) )**0.5
    pmdec_error= ( (pm_err_min**2 + pm_err_maj**2 * tan**2) / (1 + tan**2) )**0.5
    pm_corr    = tan / (1 + tan**2) * (pm_err_maj**2 - pm_err_min**2) / (pmra_error * pmdec_error)
    # add errors to the true values of pmra, pmdec
    if not covfnc is None:
        # same steps as in the above routine
        covmat = numpy.array([ covfnc( angular_distance(ra, dec, ra0, dec0) )
            for ra0, dec0 in zip(ra, dec) ])
        covmat2= numpy.zeros((2*nstar, 2*nstar))
        covmat2[0::2, 0::2] = covmat
        covmat2[1::2, 1::2] = covmat
        stride = 2*(2*nstar+1)
        covmat2.flat[        0::stride] += pmra_error **2
        covmat2.flat[2*nstar+1::stride] += pmdec_error**2
        covmat2.flat[        1::stride] += pmra_error * pmdec_error * pm_corr
        covmat2.flat[2*nstar  ::stride] += pmra_error * pmdec_error * pm_corr
        chol  = numpy.linalg.cholesky(covmat2)
        errs  = numpy.dot(chol, numpy.random.normal(size=2*nstar))   # noise/measurement errors
        pmra += errs[0::2]
        pmdec+= errs[1::2]
    else:  # simpler case with no correlated errors
        noise1, noise2 = numpy.random.normal(size=(2,nstar))
        pmra += pmra_error  *  noise1
        pmdec+= pmdec_error * (noise2 * (1-pm_corr**2)**0.5 + noise1 * pm_corr)

    return ra, dec, pmra, pmdec, pmra_error, pmdec_error, pm_corr


##### test/example of the usage for the above code #####

if __name__ == '__main__':
    import matplotlib.patches, matplotlib.pyplot as plt

    def ellipse(meanx, meany, sigmax, sigmay, corr, **kw):
        # draw error ellipse defined by standard deviations and correlation coefficient
        if corr==0:
            return matplotlib.patches.Ellipse((meanx,meany), width=2*sigmax, height=2*sigmay, **kw)
        sum = sigmax**2+sigmay**2
        dif = sigmax**2-sigmay**2
        det = (dif**2 + (2*sigmax*sigmay*corr)**2)**0.5
        ang = numpy.arctan2(2*corr*sigmax*sigmay, dif+det)
        return matplotlib.patches.Ellipse((meanx,meany), width=(2*(sum+det))**0.5, height=(2*(sum-det))**0.5, angle=180/numpy.pi*ang, **kw)


    def test(nstar, radius, sigma=0, covfnc=None, plot=False, fix_sigma=False):
        ra, dec, pmra, pmdec, pmra_err, pmdec_err, pm_corr = create_cluster(nstar, radius, sigma, covfnc)
        meanpmra, meanpmdec, errpmra, errpmdec, corr, sigma, sigerr = \
            get_mean_pm(ra, dec, pmra, pmdec, pmra_err, pmdec_err, pm_corr, sigma if fix_sigma else None, radius, covfnc)
        print("pmra=%.3f  pmdec=%.3f  pmra_err=%.3f  pmdec_err=%.3f  corr=%.3f  sigma=%.3f  sigma_err=%.3f" % \
            (meanpmra, meanpmdec, errpmra, errpmdec, corr, sigma, sigerr))
        if plot:
            for i in range(nstar):
                plt.gca().add_artist(ellipse(pmra[i], pmdec[i], pmra_err[i], pmdec_err[i], pm_corr[i],
                    alpha=0.1, lw=0, color=numpy.random.rand(3,1)))
            plt.plot(0, 0, 'o', color='b', markeredgewidth=0)  # true value
            plt.errorbar(meanpmra, meanpmdec, xerr=errpmra, yerr=errpmdec, color='r', markeredgewidth=0)
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            plt.show()

    numpy.random.seed(142)
    print('sigma=0, with sys.err.: fix sigma or infer it from data')
    test(nstar=500, radius=0.5, sigma=0.0, covfnc=covfnc1, fix_sigma=True)
    test(nstar=500, radius=0.5, sigma=0.0, covfnc=covfnc1)
    print('sigma=0.3, with sys.err.')
    test(nstar=500, radius=0.5, sigma=0.3, covfnc=covfnc1, fix_sigma=True)
    test(nstar=500, radius=0.5, sigma=0.3, covfnc=covfnc1)
    print('sigma=0, no sys.err.')
    test(nstar=500, radius=0.5, sigma=0.0, covfnc=None, fix_sigma=True)
    test(nstar=500, radius=0.5, sigma=0.0, covfnc=None)
    print('sigma=0.3, no sys.err.')
    test(nstar=500, radius=0.5, sigma=0.3, covfnc=None, fix_sigma=True)
    test(nstar=500, radius=0.5, sigma=0.3, covfnc=None)
