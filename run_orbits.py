import numpy
from astropy import units as u
from astropy import coordinates as coord

# STEP 1: create Monte Carlo realizations of position and velocity of each cluster,
# sampling from their measured uncertainties.

# this file should have been produced by run_fit.py
tab   = numpy.loadtxt('result.txt', dtype=str)
names = tab[:,0]   # 0th column is the cluster name (string)
tab   = tab[:,1:].astype(float)  # remaining columns are numbers
ra0   = tab[:,0]   # coordinates of cluster centers [deg]
dec0  = tab[:,1]
dist0 = tab[:,2]   # distance [kpc]
vlos0 = tab[:,3]   # line-of-sight velocity [km/s]
vlose = tab[:,4]   # its error estimate
pmra0 = tab[:,7]   # mean proper motion [mas/yr]
pmdec0= tab[:,8]
pmrae = tab[:,9]   # its uncertainty
pmdece= tab[:,10]
pmcorr= tab[:,11]  # correlation coefficient for errors in two PM components
vlose = numpy.maximum(vlose, 2.0)  # assumed error of at least 2 km/s on line-of-sight velocity
diste = dist0 * 0.46*0.1           # assumed error of 0.1 mag in distance modulus

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
u.kms = u.km/u.s
c_sky = coord.ICRS(ra=ra*u.degree, dec=dec*u.degree, pm_ra_cosdec=pmra*u.mas/u.yr, pm_dec=pmdec*u.mas/u.yr, distance=dist*u.kpc, radial_velocity=vlos*u.kms)
c_gal = c_sky.transform_to(coord.Galactocentric(galcen_distance=8.2*u.kpc, galcen_v_sun=coord.CartesianDifferential([10., 248., 7.]*u.kms)))
pos   = numpy.column_stack((c_gal.  x/u.kpc, c_gal.  y/u.kpc, c_gal.  z/u.kpc))
vel   = numpy.column_stack((c_gal.v_x/u.kms, c_gal.v_y/u.kms, c_gal.v_z/u.kms))
# add uncertainties from the solar position and velocity
pos[:,0] += numpy.random.normal(scale=0.1, size=nboot*nclust)  # uncertainty in solar distance from Galactic center
vel[:,0] += numpy.random.normal(scale=1.0, size=nboot*nclust)  # uncertainty in solar velocity
vel[:,1] += numpy.random.normal(scale=3.0, size=nboot*nclust)
vel[:,2] += numpy.random.normal(scale=1.0, size=nboot*nclust)
pos[:,0] *= -1  # revert back to normal orientation of coordinate system (solar position at x=+8.2)
vel[:,0] *= -1  # same for velocity
posvel= numpy.column_stack((pos,vel))
numpy.savetxt('posvel.txt', posvel, fmt='%.6g')


# STEP 2: compute the orbits, min/max galactocentric radii, and actions, for all Monte Carlo samples
import agama
print(agama.setUnits(length=1, velocity=1, mass=1))  # units: kpc, km/s, Msun; time unit ~ 1 Gyr
potential = agama.Potential('McMillan17.ini')        # MW potential from McMillan(2017)

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
rmin = numpy.nanpercentile(rmin.reshape(nclust, nboot), [16,50,84], axis=1)
rmax = numpy.nanpercentile(rmax.reshape(nclust, nboot), [16,50,84], axis=1)

# compute actions for the same initial conditions
actfinder = agama.ActionFinder(potential)
actions = actfinder(posvel)
# again compute the median and 68% confidence intervals for each cluster
actions = numpy.nanpercentile(actions.reshape(nclust, nboot, 3), [16,50,84], axis=1)

# compute the same confidence intervals for the total energy
energy  = potential.potential(posvel[:,0:3]) + 0.5 * numpy.sum(posvel[:,3:6]**2, axis=1)
energy  = numpy.percentile(energy.reshape(nclust, nboot), [16,50,84], axis=1)

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
