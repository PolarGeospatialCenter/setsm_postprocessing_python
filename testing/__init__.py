import os
from platform import system


# Establish directory for saving and reading of test files (mainly images).

TESTDIR = os.path.join(os.path.expanduser('~'), 'setsm_postprocessing_testFiles')

# SYSTYPE = system()
# if SYSTYPE == 'Windows':
#     TESTDIR = 'D:/test_s2s/testFiles/'
# elif SYSTYPE == 'Linux':
#     TESTDIR = '/mnt/pgc/data/scratch/erik/test_s2s/testFiles/'

if not os.path.isdir(TESTDIR):
    print("Creating 'testFiles' directory: {}".format(TESTDIR))
    print("Modify `testing` module init file to change directory location: {}".format(
        os.path.realpath(__file__))
    )
    os.makedirs(TESTDIR)

# Prefix of index file used for concurrent-run comparison mode of scenes2strips
PREFIX_RUNNUM = 'CURRENT_RUNNUM_'


PROJREF_POLAR_STEREO = """PROJCS["unnamed",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-70],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]]]"""
