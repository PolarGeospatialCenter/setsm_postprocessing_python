import os
from platform import system


# Establish directory for saving and reading of test files (mainly images).

TESTDIR = os.path.join(os.path.expanduser('~'), 'scratch', 'setsm_postprocessing_testFiles')

# SYSTYPE = system()
# if SYSTYPE == 'Windows':
#     TESTDIR = r'E:\scratch\test\s2s\testFiles'
#     # TESTDIR = r'V:\pgc\data\scratch\erik\test\s2s\testFiles'
# elif SYSTYPE == 'Linux':
#     TESTDIR = r'/mnt/pgc/data/scratch/erik/test/s2s/testFiles'


# Prefix of index file used for concurrent-run comparison mode of scenes2strips
PREFIX_RUNNUM = 'CURRENT_RUNNUM_'
