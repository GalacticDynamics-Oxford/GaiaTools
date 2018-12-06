import numpy

# download the file with renormalized unit weight error correction tables from the Gaia website
ruwefile = 'DR2_RUWE_V1/table_u0_2D.txt'
import subprocess, os
if not os.path.isfile(ruwefile):
    subprocess.call('curl https://www.cosmos.esa.int/documents/29201/1769576/DR2_RUWE_V1.zip/d90f37a8-37c9-81ba-bf59-dd29d9b1438f > temp.zip', shell=True)
    subprocess.call('unzip temp.zip '+ruwefile, shell=True)
    os.remove('temp.zip')
if not os.path.isdir('data'):
    os.mkdir('data')

# construct the interpolator for the renormalized unit weight error correction table
import scipy.interpolate
rtab=numpy.loadtxt(ruwefile, delimiter=',', skiprows=1)
# correction factor as a function of g_mag and bp_rp
rint=scipy.interpolate.RectBivariateSpline(
    x=rtab[:,0],
    y=numpy.linspace(-1., 10., 111),
    z=rtab[:,2:], kx=1, ky=1)
# correction factor in case of no bp/rp, as a function of g_mag only
rint0=scipy.interpolate.UnivariateSpline(x=rtab[:,0], y=rtab[:,1], k=1, s=0)

# this is the basic interface for querying the Gaia archive
from astroquery.utils.tap.core import Tap
gaia = Tap(url="http://gea.esac.esa.int/tap-server/tap")

# silence some irrelevant warnings
import warnings, astropy
warnings.filterwarnings('ignore', category=astropy.io.votable.VOWarning)


def retrieve(ra, dec, radius, filename, parallax_limit):
    """
    Query the Gaia archive for all sources within a certain radius from the given point,
    which have parallax below the given limit (within 3 sigma),
    and save the result as a numpy zip archive.
    """
    job = gaia.launch_job("select top 999999 "+
        "ra, dec, pmra, pmra_error, pmdec, pmdec_error, pmra_pmdec_corr, "+
        "phot_g_mean_mag, bp_rp, "+
        "sqrt(astrometric_chi2_al/(astrometric_n_good_obs_al-5)) as uwe, "+
        "astrometric_excess_noise, phot_bp_rp_excess_factor "+
        "FROM gaiadr2.gaia_source WHERE "+
        "parallax is not null and "+
        "parallax-"+str(parallax_limit)+"<3*parallax_error and "+
        "contains(point('icrs',gaiadr2.gaia_source.ra,gaiadr2.gaia_source.dec), "+
        "circle('icrs',"+str(ra)+","+str(dec)+","+str(radius)+"))=1")
    table = job.get_results()
    print("%s => %d" % (filename, len(table)))
    # compute the renormalized unit weight error from the interpolation tables
    g_mag = numpy.array(table['phot_g_mean_mag'])
    bp_rp = numpy.array(table['bp_rp'])
    rfac  = rint(g_mag, bp_rp, grid=False)
    rfac[numpy.isnan(bp_rp)] = rint0(g_mag[numpy.isnan(bp_rp)])
    # save the data as a numpy zip archive
    numpy.savez_compressed("data/"+filename,
        ra = numpy.array(table['ra']).astype(numpy.float32),
        dec = numpy.array(table['dec']).astype(numpy.float32),
        pmra = numpy.array(table['pmra']).astype(numpy.float32),
        pmdec= numpy.array(table['pmdec']).astype(numpy.float32),
        pmra_error = numpy.array(table['pmra_error']).astype(numpy.float32),
        pmdec_error = numpy.array(table['pmdec_error']).astype(numpy.float32),
        pmra_pmdec_corr = numpy.array(table['pmra_pmdec_corr']).astype(numpy.float32),
        phot_g_mean_mag = g_mag.astype(numpy.float32),
        bp_rp = bp_rp.astype(numpy.float32),
        ruwe = (numpy.array(table['uwe']) / rfac).astype(numpy.float32),
        astrometric_excess_noise = numpy.array(table['astrometric_excess_noise']).astype(numpy.float32),
        phot_bp_rp_excess_factor = numpy.array(table['phot_bp_rp_excess_factor']).astype(numpy.float32)
    )

# read the list of clusters and query the Gaia archive for each of them
lst = numpy.genfromtxt('input.txt', dtype=str)
for l in lst:
    retrieve(ra=float(l[1]), dec=float(l[2]), radius=float(l[7])/60,  # convert from arcmin to degrees
        filename=l[0], parallax_limit=1./float(l[3]))
