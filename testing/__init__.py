import os
from platform import system


# Establish directory for saving and reading of test files (mainly images).

TESTDIR = os.path.join(os.path.expanduser('~'), 'scratch', 'setsm_postprocessing_testFiles')

# SYSTYPE = system()
# if SYSTYPE == 'Windows':
#     # TESTDIR = r'D:\setsm_postprocessing_tests\testFiles'
#     TESTDIR = r'V:\pgc\data\scratch\erik\setsm_postprocessing_tests\testFiles'
# elif SYSTYPE == 'Linux':
#     TESTDIR = r'/mnt/pgc/data/scratch/erik/setsm_postprocessing_tests/testFiles'


# Prefix of index file used for concurrent-run comparison mode of scenes2strips
PREFIX_RUNNUM = 'CURRENT_RUNNUM_'
