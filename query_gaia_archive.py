import numpy, os
if not os.path.isdir('data'):
    os.mkdir('data')

# this is the basic interface for querying the Gaia archive
from astroquery.utils.tap.core import Tap
gaia = Tap(url="https://gea.esac.esa.int/tap-server/tap")

# parallax zero-point correction from Lindegren+2020
try:
    from zero_point import zpt
    zpt.load_tables()
except Exception as ex:
    print("Parallax zero-point correction not available: "+str(ex))
    zpt = None

def retrieve(ra, dec, radius, filename, parallax0):
    """
    Query the Gaia archive for all sources within a certain radius from the given point,
    which have parallax below the given limit (within 3 sigma),
    and save the result as a numpy zip archive.
    """
    job = gaia.launch_job("select top 999999 "+
        "ra, dec, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error, pmra_pmdec_corr, "+
        "phot_g_mean_mag, bp_rp, ruwe, astrometric_excess_noise, phot_bp_rp_excess_factor, "+
        "nu_eff_used_in_astrometry, pseudocolour, ecl_lat, astrometric_params_solved "+
        "FROM gaiaedr3.gaia_source WHERE "+
        "phot_g_mean_mag<=21 and parallax is not null and "+
        "abs(parallax%+.3f"%(-parallax0)+")<3*parallax_error and "+
        "contains(point('icrs',gaiaedr3.gaia_source.ra,gaiaedr3.gaia_source.dec), "+
        "circle('icrs',"+str(ra)+","+str(dec)+","+str(radius)+"))=1")
    table = job.get_results()
    print("%s => %d" % (filename, len(table)))
    # apply parallax zero-point correction
    phot_g_mean_mag = numpy.array(table['phot_g_mean_mag'])
    parallax = numpy.array(table['parallax'])
    if zpt is not None:
        parallax -= numpy.nan_to_num(zpt.get_zpt(phot_g_mean_mag,
            table['nu_eff_used_in_astrometry'], table['pseudocolour'], table['ecl_lat'],
            table['astrometric_params_solved'], _warnings=True))
    # save the data as a numpy zip archive
    numpy.savez_compressed("data/"+filename,
        ra = numpy.array(table['ra']).astype(numpy.float32),
        dec = numpy.array(table['dec']).astype(numpy.float32),
        parallax = parallax.astype(numpy.float32),
        parallax_error = numpy.array(table['parallax_error']).astype(numpy.float32),
        pmra = numpy.array(table['pmra']).astype(numpy.float32),
        pmdec= numpy.array(table['pmdec']).astype(numpy.float32),
        pmra_error = numpy.array(table['pmra_error']).astype(numpy.float32),
        pmdec_error = numpy.array(table['pmdec_error']).astype(numpy.float32),
        pmra_pmdec_corr = numpy.array(table['pmra_pmdec_corr']).astype(numpy.float32),
        phot_g_mean_mag = phot_g_mean_mag.astype(numpy.float32),
        bp_rp = numpy.array(table['bp_rp']).astype(numpy.float32),
        ruwe = numpy.array(table['ruwe']).astype(numpy.float32),
        astrometric_excess_noise = numpy.array(table['astrometric_excess_noise']).astype(numpy.float32),
        phot_bp_rp_excess_factor = numpy.array(table['phot_bp_rp_excess_factor']).astype(numpy.float32),
        fiveparam = numpy.array(table['astrometric_params_solved'])==31
    )

# read the list of clusters and query the Gaia archive for each of them
lst = numpy.genfromtxt('input.txt', dtype=str)
for l in lst:
    retrieve(ra=float(l[1]), dec=float(l[2]), radius=float(l[8])/60,  # convert from arcmin to degrees
        filename=l[0], parallax0=1./float(l[3])-0.03)
