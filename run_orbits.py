import numpy, agama

# STEP 1: create Monte Carlo realizations of position and velocity of each cluster,
# sampling from their measured uncertainties.

# this file should have been produced by run_fit.py
tab   = numpy.loadtxt('result.txt', dtype=str)
names = tab[:,0]   # 0th column is the cluster name (string)
tab   = tab[:,1:].astype(float)  # remaining columns are numbers
ra0   = tab[:,0]   # coordinates of cluster centers [deg]
dec0  = tab[:,1]
dist0 = tab[:,2]   # distance [kpc]
diste = tab[:,3]   # distance error [kpc]
vlos0 = tab[:,4]   # line-of-sight velocity [km/s]
vlose = tab[:,5]   # its error estimate
pmra0 = tab[:,8]   # mean proper motion [mas/yr]
pmdec0= tab[:,9]
pmrae = tab[:,11]  # its uncertainty
pmdece= tab[:,12]
pmcorr= tab[:,13]  # correlation coefficient for errors in two PM components
vlose = numpy.maximum(vlose, 2.0)  # assumed error of at least 2 km/s on line-of-sight velocity
diste[~numpy.isfinite(diste)] = dist0[~numpy.isfinite(diste)] * 0.05   # assumed error of 0.1 mag in distance modulus when not known
vlos0[~numpy.isfinite(vlos0)] = 0.0   # replace non-existent line-of-sight velocity measurements
vlose[~numpy.isfinite(vlose)] = 150.0 # with dummy values and large uncertainties

# create bootstrap samples
numpy.random.seed(42)  # ensure repeatability of random samples
nboot = 100            # number of bootstrap samples for each cluster
nclust= len(tab)
ra    = numpy.repeat(ra0,   nboot)
dec   = numpy.repeat(dec0,  nboot)
pmra  = numpy.repeat(pmra0, nboot)
pmdec = numpy.repeat(pmdec0,nboot)
for i in range(nclust):
    # draw PM realizations from a correlated 2d gaussian for each cluster
    A = numpy.random.normal(size=nboot)
    B = numpy.random.normal(size=nboot) * (1-pmcorr[i]**2)**0.5 + A * pmcorr[i]
    pmra [i*nboot:(i+1)*nboot] += pmrae [i] * A
    pmdec[i*nboot:(i+1)*nboot] += pmdece[i] * B
vlos  = numpy.repeat(vlos0, nboot) + numpy.hstack([numpy.random.normal(scale=e, size=nboot) for e in vlose])
dist  = numpy.repeat(dist0, nboot) + numpy.hstack([numpy.random.normal(scale=e, size=nboot) for e in diste])

# convert coordinates from heliocentric (ra,dec,dist,PM,vlos) to Galactocentric (kpc and km/s)
d2r = numpy.pi/180
l, b, pml, pmb = agama.transformCelestialCoords(agama.fromICRStoGalactic, ra*d2r, dec*d2r, pmra, pmdec)
posvel = numpy.column_stack(agama.getGalactocentricFromGalactic(l, b, dist, pml*4.74, pmb*4.74, vlos))
numpy.savetxt('posvel.txt', posvel, fmt='%.6g')

try:
    nanpercentile = numpy.nanpercentile
except AttributeError:  # older numpy versions don't have this function
    def nanpercentile(x, *args, **namedargs):  # an approximation which is not quite the same though..
        return numpy.array(numpy.percentile(numpy.nan_to_num(x), *args, **namedargs))

# STEP 2: compute the orbits, min/max galactocentric radii, and actions, for all Monte Carlo samples
agama.setUnits(length=1, velocity=1, mass=1)   # units: kpc, km/s, Msun; time unit ~ 1 Gyr
potential = agama.Potential('McMillan17.ini')  # MW potential from McMillan(2017) - may choose another one

# compute orbits for each realization of initial conditions,
# integrated for 100 dynamical times or 20 Gyr (whichever is lower)
print("Computing orbits for %d realizations of cluster initial conditions" % len(posvel))
inttime= numpy.minimum(20., potential.Tcirc(posvel)*100)
orbits = agama.orbit(ic=posvel, potential=potential, time=inttime, trajsize=1000)[:,1]
rmin = numpy.zeros(len(orbits))
rmax = numpy.zeros(len(orbits))
for i,o in enumerate(orbits):
    r = numpy.sum(o[:,0:3]**2, axis=1)**0.5
    rmin[i] = numpy.min(r) if len(r)>0 else numpy.nan
    rmax[i] = numpy.max(r) if len(r)>0 else numpy.nan
# replace nboot samples rmin/rmax with their median and 68% confidence intervals for each cluster
rmin = nanpercentile(rmin.reshape(nclust, nboot), [16,50,84], axis=1)
rmax = nanpercentile(rmax.reshape(nclust, nboot), [16,50,84], axis=1)

# compute actions for the same initial conditions
actfinder = agama.ActionFinder(potential)
actions = actfinder(posvel)
# again compute the median and 68% confidence intervals for each cluster
actions = nanpercentile(actions.reshape(nclust, nboot, 3), [16,50,84], axis=1)

# compute the same confidence intervals for the total energy
energy  = potential.potential(posvel[:,0:3]) + 0.5 * numpy.sum(posvel[:,3:6]**2, axis=1)
energy  = numpy.array(numpy.percentile(energy.reshape(nclust, nboot), [16,50,84], axis=1))

# write the orbit parameters, actions and energy - one line per cluster, with the median and uncertainties
fileout = open("result_orbits.txt", "w")
fileout.write("# Name         \t     pericenter[kpc]   \t     apocenter[kpc]    \t" +
      "       Jr[kpc*km/s]    \t       Jz[kpc*km/s]    \t      Jphi[kpc*km/s]   \t    Energy[km^2/s^2]   \n")
for i in range(nclust):
    fileout.write( ("%-15s" + "\t%7.2f" * 6 + "\t%7.0f" * 12 + "\n") %
        (names[i], rmin[0,i], rmin[1,i], rmin[2,i], rmax[0,i], rmax[1,i], rmax[2,i],
        actions[0,i,0], actions[1,i,0], actions[2,i,0],
        actions[0,i,1], actions[1,i,1], actions[2,i,1],
        actions[0,i,2], actions[1,i,2], actions[2,i,2],
        energy[0,i], energy[1,i], energy[2,i]) )
fileout.close()
