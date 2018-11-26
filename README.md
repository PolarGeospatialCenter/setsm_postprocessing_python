# setsm_postprocessing_python
Methods for filtering and mosaicking DEMs produced by [SETSM](https://github.com/setsmdeveloper/SETSM "SETSM on GitHub").

The methodology for both filtering and mosaicking is largely ported from [MATLAB code written by Ian Howat](https://github.com/ihowat/setsm_postprocessing "setsm_postprocessing on GitHub"), glaciologist and professor of Earth Sciences at The Ohio State University.


## Python Requirements
Scripts in this repo were originally written to work with Python 2.7, but support was recently added for Python 3.6/3.7+. If you get an error when attempting to run the code in any of these Python versions, please create an Issue.

### Python package dependencies
* NumPy
* SciPy
* scikit-image
* OpenCV
* GDAL (including OSGeo, OGR, OSR)
* Shapely
* tifffile (required only for debugging)

### Recommendations on Python environment setup with *conda*
You will likely have issues getting all of the dependencies installed and working together properly without using [conda](https://conda.io/docs/index.html "conda landing page"). Even if you do use conda, you may still run into issues with conflicting dependencies depending on your system (more on that later). Conda is an open source package manager that generally works very well at quickly spinning up the particular Python environment that you need to get a program running. I recommend installing the [Miniconda](https://conda.io/miniconda.html "Miniconda installers") distribution (a clearner install that starts you off only with the conda package manager and Python + [Standard Library](https://docs.python.org/3/library/ "The Python Standard Library") modules) over the [Anaconda](https://www.anaconda.com/download/ "Anaconda installers") distribution (which installs ~100 scientific Python packages, most of which you will never use). You will have to pick between distros with Python 2.7 or Python 3.7 for your base install, but regardless of which version you choose now (and if you chose Anaconda or Miniconda), later you can set up an environment with the particular Python version and packages you need to get the job done. [This page offers both a good introduction to conda and a reference to basic conda commands.](https://conda.io/docs/user-guide/getting-started.html "Getting started with conda") 

Once conda has been installed and you've created the environment into which you will install the required packages (or you may decide to not create a new environment and just install the packages into your default 'base' environment), You can install all of the packages at once or individually in whatever order you like, but I recommend installing them in the following order by running these commands:
1. `conda install numpy scipy`
2. `conda install shapely`
3. `conda install scikit-image`
4. `conda install gdal`
5. `conda install opencv`
6. (optional) `conda install -c conda-forge tifffile`

Between running and completing these install commands, I would check to make sure you can successfully import the appropriate modules into Python. As you progress, try running the following commands in your environment's Python interpreter:
1. `import numpy, scipy`
2. `import shapely`
3. `import skimage`
4. `import gdal, osgeo, ogr, osr`
5. `import cv2`
6. `import tifffile`

If any of the imports raise an error with a mysterious message, you may want to note what the error is and try Googling the error message to see what other people have said about it. Sometimes an issue with the package install (likely conflicting dependencies) will cause a module name to not be recognized at all in the Python interpreter. Regardless, these issues have a good chance of automatically being resolved on their own as you progress (since conda will change the versions of installed packages automatically in its deconflicting step), so attempt to push forward until you have successfully installed all of the packages with `conda install` (again, you shouldn't need to install them in any specific order). If you continue to have issues importing all of the listed modules in the Python interpreter -- or worse, the `conda install` command doesn't complete successfully -- try installing the latest version of the dysfunctional package(s) through the [conda-forge](https://conda-forge.org/#about "conda-forge 'About' page") by adding the `-c conda-forge` optional argument to the `conda install` command. You may want to uninstall a broken package before upgrading, but I can't offer an exact solution for every system. If all else fails, try pinning the versions of installed packages that you know work and maybe do a [`conda install --force --no-update-dependencies`](https://conda.io/docs/commands/conda-install.html "`conda install` documentation") to install the remainder.


## batch_scenes2strips.py
Located in the root folder of the repo, this is the main post-processing script.

### Turning scenes into strips
In the context of this repo, a "scene" is what we call the set of result raster images and a metadata text file that are produced by SETSM when run on a pair of overlapping chunks of stereo DigitalGlobe satellite images. Each scene is composed of rasters with filenames ending with *dem.tif*/*dem_smooth.tif*, *matchtag.tif*/*matchtag_mt.tif*, and *ortho.tif* and an auxilary metadata text file ending with *meta.txt*. Since DigitalGlobe customers often don't receive a single collect as a whole long, gigantic image but instead can receive slightly-overlapping image chunks that together make up the whole collect, there is often a large number of combinations for overlapping stereo images to process between a pair of stereo collects. After these overlapping stereo image "scenes" (some call these "subscenes" since the word "scene" is often attributed to the whole satellite collect) have been processed with SETSM, they should be stitched together to create the best possible representation of the entire area of overlap between the stereo collects -- the "strip" -- whence they came. That is what this script aims to do, in batch.

### Step 1: Source scene selection by "strip-pair ID"
(text)

### Step 2: Scene filtering and creation of the *bitmask.tif* scene component
(text)

### Step 3: Scene ordering and coregistration
(text)

### Step 3/4: Mosaicking of scenes into strip segment(s)
(text)

### Completion of strip processing and creation of the *.fin* file
(text)

### Notes on batch job submission to scheduler
(text)

### Commentary on script arguments

```
usage: batch_scenes2strips.py [-h] [--dst DST]
                              [--meta-trans-dir META_TRANS_DIR]
                              [--mask-ver {maskv1,maskv2,rema2a,mask8m,bitmask}]
                              [--noentropy] [--nowater] [--nocloud]
                              [--nofilter-coreg]
                              [--save-coreg-step {off,meta,all}]
                              [--rmse-cutoff RMSE_CUTOFF]
                              [--scheduler {pbs,slurm}]
                              [--jobscript JOBSCRIPT] [--logdir LOGDIR]
                              [--dryrun] [--stripid STRIPID]
                              src res

Filters scene DEMs in a source directory, then mosaics them into strips and saves the results. 
Batch work is done in units of strip-pair IDs, as parsed from scene dem filenames (see --stripid argument for how this is parsed).

positional arguments:
  src                   Path to source directory containing scene DEMs to process. If --dst is not specified, this path should contain the folder 'tif_results'.
  res                   Resolution of target DEMs in meters.

optional arguments:
  -h, --help            show this help message and exit
  --dst DST             Path to destination directory for output mosaicked strip data. (default is src.(reverse)replace('tif_results', 'strips')) (default: None)
  --meta-trans-dir META_TRANS_DIR
                        Path to directory of old strip metadata from which translation values will be parsed to skip scene coregistration step. (default: None)
  --mask-ver {maskv1,maskv2,rema2a,mask8m,bitmask}
                        Filtering scheme to use when generating mask raster images, to classify bad data in scene DEMs. 
                        'maskv1': Two-component (edge, data density) filter to create separate edgemask and datamask files for each scene. 
                        'maskv2': Three-component (edge, water, cloud) filter to create classical 'flat' binary masks for 2m DEMs. 
                        'bitmask': Same filter as 'maskv2', but distinguish between the different filter components by creating a bitmask. 
                        'rema2a': Filter designed specifically for 8m Antarctic DEMs. 
                        'mask8m': General-purpose filter for 8m DEMs. 
                         (default: bitmask)
  --noentropy           Use filter without entropy protection. Can only be used when --mask-ver=maskv1. (default: False)
  --nowater             Use filter without water masking. Can only be used when --mask-ver=bitmask. (default: False)
  --nocloud             Use filter without cloud masking. Can only be used when --mask-ver=bitmask. (default: False)
  --nofilter-coreg      If --nowater/--nocloud, turn off the respective filter(s) during coregistration step in addition to mosaicking step. Can only be used when --mask-ver=bitmask. (default: False)
  --save-coreg-step {off,meta,all}
                        If --nowater/--nocloud, save output from coregistration step in directory '`dstdir`_coreg_filtXXX' where [XXX] is the bit-code corresponding to filter components ([cloud, water, edge], respectively) applied during the coregistration step. By default, all three filter components are applied so this code is 111. 
                        If 'off', do not save output from coregistration step. 
                        If 'meta', save only the *_meta.txt component of output strip segments. (useful for subsequent runs with --meta-trans-dir argument) 
                        If 'all', save all output from coregistration step, including both metadata and raster components. 
                        Can only be used when --mask-ver=bitmask, and has no affect if neither --nowater nor --nocloud arguments are provided, or either --meta-trans-dir or --nofilter-coreg arguments are provided since then the coregistration and mosaicking steps are effectively rolled into one step. 
                         (default: off)
  --rmse-cutoff RMSE_CUTOFF
                        Maximum RMSE from coregistration step tolerated for scene merging. A value greater than this causes a new strip segment to be created. (default: 1.0)
  --scheduler {pbs,slurm}
                        Submit tasks to job scheduler. (default: None)
  --jobscript JOBSCRIPT
                        Script to run in job submission to scheduler. (default scripts are found in /mnt/pgc/data/scratch/erik/repos/setsm_postprocessing_python/jobscripts) (default: None)
  --logdir LOGDIR       Directory to which standard output/error log files will be written for batch job runs. 
                        If not provided, default scheduler (or jobscript #CONDOPT_) options will be used. 
                        **Note that due to implementation difficulties, this directory will also become the working directory for the job process. Since relative path inputs are always changed to absolute paths in this script, this should not be an issue. (default: None)
  --dryrun              Print actions without executing. (default: False)
  --stripid STRIPID     Run filtering and mosaicking for a single strip with strip-pair ID as parsed from scene DEM filenames using the following regex: '(^[A-Z0-9]{4}_.*?_?[0-9A-F]{16}_.*?_?[0-9A-F]{16}).*$' (default: None)
  ```

* `src` :: In the OSU-PGC processing scheme, this is a path to a `*/tif_results/8m` or `*/tif_results/2m` folder for 8-meter or 2-meter DEMs, respectively. For 50-centimeter processing, it doesn't matter if the name of the lowest folder is "50cm" or "0.5m" or whatever.
* `res` :: NUMERIC INPUT; This value must be provided in METERS! The input resolution value is used to (1) make sure that the selected scenes in the `src` source directory are indicated by their filenames to be of the same resolution, (2) check if strip results already exist in the `--dst` destination directory *and skip processing the strip if any results already exist*, (3) make sure the selected `--mask-ver` filter scheme works with the resolution, and (4) is the resolution included in the filenames of the output strip results files. For more information on how condition (1) is enforced (as )
* `--dst` :: In the OSU-PGC processing scheme, this will be a path to a `*/strips/8m` or `*/strips/2m` folder where the `*` is the same path as given in the commentary for `src`. If this argument isn't provided an attempt is made to determine this path by default (as specified in the `--help` text), so if you're following the OSU-PGC processing scheme you don't ever need to provide this argument. However, if you think you will be creating both filtered and unfiltered versions of strips (through the `--nowater`/`--nocloud` arguments), I recommend making this path `*/strips/2m_filtXXX` where \[`XXX`\] is the bit-code corresponding to filter components (\[cloud, water, edge\], respectively) applied during the final mosaicking step of the scenes2strips process. The bit-code for completely filtered strips is thus `111` while for completely "unfiltered" strips (both `--nowater` and `--nocloud` provided) it is `001`.
* `--meta-trans-dir` :: This option is useful if either (1) you need to reprocess/recreate strips for some reason and you trust that the coregistration of the scenes in the earlier processing is still sound or (2) you were running scenes2strips with `--save-coreg-step` set to either `meta` or `all` when a crash/abort caused processing to end prematurely and you want to skip redoing the processing of the coregistration step. If provided, the program will attempt to read the translation values for scenes from old strip segment *meta.txt* files in the provided directory and use these values for assembling strips in lieu of running the coregistration step. If it happens that the scenes2strips process decides to break the new strip into segments differently than it broke for the old strip results in the `--meta-trans-dir` directory, the program will fall back to running as if the argument was never provided *for only that particular strip*.
* `--mask-ver` :: Improvements to the scenes2strips filtering step are currently focusing solely on the *bitmask* version that creates the *bitmask.tif* scene/strip mask component raster. Only change this option from its default value (`bitmask`) if for some reason you're interested in seeing what the old masks look like.
* `--noentropy` :: As noted in the `--help` text, this argument can only be provided along with `--mask-ver=maskv1`. You probably won't ever use it.
* `--nowater`/`--nocloud` :: These arguments allow for producing unfiltered strip results, which is what really differentiates this version of the scenes2strips process from the earlier MATLAB version. See subsection "Step 2: Scene filtering and creation of the *bitmask.tif* scene component" above for more information.
* `--nofilter-coreg` :: By default, all filters (edge, water, and cloud) are applied during the coregistration step with the assumption that the filters remove only bad data and lead to a better coregistration. If the filters are instead removing more good data than bad data, a better coregistration may be achieved by providing this argument to turn off the offending filters during the coregistration step. (Note that the edge filter must be applied in both the coregistration and mosaicking steps because it is known that DEMs produced by SETSM have a border of bad data around their edges that needs to be cropped off before merging.)
* `--save-coreg-step` :: When the scenes2strips process is split into separate coregistration and mosaicking steps due to providing `--nowater`/`--nocloud`, this option allows for caching the results of the coregistration step. Set this option to `all` if you want the full unfiltered version of strip output in addition to a filtered version. Set it to `meta` if you want to make sure to have a backup of the translation values for `--meta-trans-dir` in case of a crash. Also note that the strip segment *meta.txt* files saved with this option will have different values for the the RMSE component of the "Mosaicking Alignment Statistics" than the output strip meta files saved in the `--dst` directory because the RMSE statistic is properly recalculated for the unfiltered version of the strip during the mosaicking step. For conditions on when this argument applies, see the `--help` text.
* `--rmse-cutoff` :: After the iterative coregistration step is complete, the final RMSE value for the coregistration is reported. If that RMSE value is greater than the value specified by this argument, the scene that failed to register to the strip will become the start of a new strip segment.
* `--scheduler` :: Currently only the PBS and SLURM job schedulers are supported. If you provide this argument, note that if you do not specify a particular PBS/SLURM jobscript to run with the `--jobscript` argument a default jobscript will be selected from the 'jobscripts' folder in the repo root directory that corresponds to this script and the indicated scheduler type.
* `--jobscript` :: REQUIREMENTS: The jobscript MUST (1) be readable by the provided `--scheduler` job scheduler type, (2) load the Python environment that includes all required packages as specified above under "Python package dependencies" before it (3) executes the main command that runs the script for a single `--stripid` by means of substituting the entire command into the jobscript through the environment variable `$p1`.
* `--logdir` :: If this argument is not provided and you are using the default jobscripts from this repo, the default output log file directory for SLURM is the directory where the command to run the script was submitted, while for PBS it is the `$HOME` directory of the user who submitted the command.
* `--dryrun` :: Useful for testing which strips will be built where, without actually starting the process of building them.
* `--stripid` :: During normal batch usage of this script, you only specify the `src` source directory of all scenes that you aim to turn into multiple strips. This argument is then typically used only internally by the batch execution logic to specify which single strip each instance of the scenes2strips program will work on. If you have a ton of scenes (that can be made into multiple strips) in the `src` directory and wish to only create/recreate a specific strip, you can specify that strip using the OSU-PGC strip-pair ID naming convention described above under "Step 1: Source scene selection by "strip-pair ID"".

### Example commands & output

Activate the Python environment (if using conda and package dependencies are installed in an environment other than your 'base' environment) or load all necessary modules.
```
ehusby@pgc:~/scratch/repos/setsm_postprocessing_python$ source activate my_root
(my_root) ehusby@pgc:~/scratch/repos/setsm_postprocessing_python$
````

Try running `python batch_scenes2strips.py --help` to make sure the core dependencies are working together without issues.

Let's say you have a directory containing 2-meter source SETSM scenes that you want to process into completely filtered strips (removing bad data from both water and clouds using the automated filtering methods). This is the most basic type of run and does not need any optional arguments specified. Try running the command first in "safe mode" with the `--dryrun` option:
```
(my_root) ehusby@pgc:~/scratch/repos/setsm_postprocessing_python$ python batch_scenes2strips.py ~/scratch/data/setsm_results/tif_results/2m/ 2 --dryrun
--dst dir set to: /home/ehusby/scratch/data/setsm_results/strips/2m
Found 3 *dem.tif strip-pair IDs, 3 unfinished
Sleeping 5 seconds before task submission
1, python -u /att/gpfsfs/hic101/ppl/ehusby/scratch/repos/setsm_postprocessing_python/batch_scenes2strips.py --dst "/home/ehusby/scratch/data/setsm_results/strips/2m" --mask-ver "bitmask" --save-coreg-step "off" --rmse-cutoff 1.0 --dryrun --stripid "WV01_20170717_102001006264A100_1020010066A25800" "/home/ehusby/scratch/data/setsm_results/tif_results/2m/" 2.0
2, python -u /att/gpfsfs/hic101/ppl/ehusby/scratch/repos/setsm_postprocessing_python/batch_scenes2strips.py --dst "/home/ehusby/scratch/data/setsm_results/strips/2m" --mask-ver "bitmask" --save-coreg-step "off" --rmse-cutoff 1.0 --dryrun --stripid "WV03_20160731_1040010020191100_104001001F543C00" "/home/ehusby/scratch/data/setsm_results/tif_results/2m/" 2.0
3, python -u /att/gpfsfs/hic101/ppl/ehusby/scratch/repos/setsm_postprocessing_python/batch_scenes2strips.py --dst "/home/ehusby/scratch/data/setsm_results/strips/2m" --mask-ver "bitmask" --save-coreg-step "off" --rmse-cutoff 1.0 --dryrun --stripid "WV03_20161014_10400100231DFD00_104001002363A300" "/home/ehusby/scratch/data/setsm_results/tif_results/2m/" 2.0
```
The program found three sets of scenes that it can attempt to merge together to create (segments of) three distinct strips. Since we did not provide the `--scheduler` argument to specify a job scheduler for processing these strips, the program will simply run each printed command in serial from the parent process. Notice how the most meaningful (non-`None`) script arguments are passed from the parent process to each child command, even default script arguments that you did not specify. This should not be a cause for concern because the child process runs the same script as the parent and would pick up the same default script arguments anyways.

Let's try removing the `--dryrun` option and giving it a go:
<details>
<summary>[click to view output]</summary>
<br/>
  
```
(my_root) ehusby@pgc:~/scratch/repos/setsm_postprocessing_python$ python batch_scenes2strips.py ~/scratch/data/setsm_results/tif_results/2m/ 2
--dst dir set to: /home/ehusby/scratch/data/setsm_results/strips/2m
Found 3 *dem.tif strip-pair IDs, 3 unfinished
Sleeping 5 seconds before task submission
1, python -u /att/gpfsfs/hic101/ppl/ehusby/scratch/repos/setsm_postprocessing_python/batch_scenes2strips.py --dst "/home/ehusby/scratch/data/setsm_results/strips/2m" --mask-ver "bitmask" --save-coreg-step "off" --rmse-cutoff 1.0 --stripid "WV01_20170717_102001006264A100_1020010066A25800" "/home/ehusby/scratch/data/setsm_results/tif_results/2m/" 2.0

stripid: WV01_20170717_102001006264A100_1020010066A25800
res: 2m
srcdir: /home/ehusby/scratch/data/setsm_results/tif_results/2m
dstdir: /home/ehusby/scratch/data/setsm_results/strips/2m
dstdir for coreg step: None
metadir: None
mask version: bitmask
mask name: bitmask
coreg filter options: ()
mask filter options: ()
rmse cutoff: 1.0
dryrun: False

Processing strip-pair ID: WV01_20170717_102001006264A100_1020010066A25800, 15 scenes

Filtering 1 of 15: /home/ehusby/scratch/data/setsm_results/tif_results/2m/WV01_20170717_102001006264A100_1020010066A25800_501591396070_01_P001_501591395050_01_P001_2_dem.tif
rescaled to: 0 to 2047.0
radiance value range: 7.00 to 331.54
/att/gpfsfs/hic101/ppl/ehusby/scratch/repos/setsm_postprocessing_python/lib/filter_scene.py:1163: RuntimeWarning: invalid value encountered in greater
  mask = (mean_slope_array > 1)
/att/gpfsfs/hic101/ppl/ehusby/scratch/repos/setsm_postprocessing_python/lib/filter_scene.py:1417: RuntimeWarning: invalid value encountered in less
  stdev_elev_array[stdev_elev_array < 0] = 0
20/80 percentile elevation difference: 986.9, sigma-z threshold: 50.0
/att/gpfsfs/hic101/ppl/ehusby/scratch/repos/setsm_postprocessing_python/lib/filter_scene.py:1447: RuntimeWarning: invalid value encountered in greater
  mask_stdev = (dem_data & (stdev_elev_array > stdev_thresh))
Saving Geotiff /home/ehusby/scratch/data/setsm_results/tif_results/2m/WV01_20170717_102001006264A100_1020010066A25800_501591396070_01_P001_501591395050_01_P001_2_bitmask.tif ... GDAL data type: Byte, NoData value: 0, Creation Options: BIGTIFF=IF_SAFER COMPRESS=LZW TILED=YES, Projection (Proj4): +proj=utm +zone=39 +south +datum=WGS84 +units=m +no_defs ... creating file ... writing array values ... finishing file ... done!
Filtering 2 of 15: /home/ehusby/scratch/data/setsm_results/tif_results/2m/WV01_20170717_102001006264A100_1020010066A25800_501591396070_01_P002_501591395050_01_P001_2_dem.tif
...
```
You can safely ignore the `RuntimeWarning` messages that appear at the start of the scene filtering process (they are only printed once). They appear when certain numerical comparison operations involve NaN values in the raster matrices, but currently these boolean operations return an acceptable value of `False`.
```
...
Filtering 15 of 15: /home/ehusby/scratch/data/setsm_results/tif_results/2m/WV01_20170717_102001006264A100_1020010066A25800_501591396070_01_P008_501591395050_01_P008_2_dem.tif
rescaled to: 0 to 2047.0
radiance value range: -3.92 to 337.13
20/80 percentile elevation difference: 386.5, sigma-z threshold: 50.0
Saving Geotiff /home/ehusby/scratch/data/setsm_results/tif_results/2m/WV01_20170717_102001006264A100_1020010066A25800_501591396070_01_P008_501591395050_01_P008_2_bitmask.tif ... GDAL data type: Byte, NoData value: 0, Creation Options: BIGTIFF=IF_SAFER COMPRESS=LZW TILED=YES, Projection (Proj4): +proj=utm +zone=39 +south +datum=WGS84 +units=m +no_defs ... creating file ... writing array values ... finishing file ... done!

All *_bitmask.tif scene masks have been created in source scene directory

Running scenes2strips

Building segment 1
Running s2s with coregistration filter options: None
Ordering 15 scenes
Scene 1 of 15: /home/ehusby/scratch/data/setsm_results/tif_results/2m/WV01_20170717_102001006264A100_1020010066A25800_501591396070_01_P008_501591395050_01_P008_2_dem.tif
Ortho had wv_correct applied, rescaling values to range [0, 2047.0]
Scene 2 of 15: /home/ehusby/scratch/data/setsm_results/tif_results/2m/WV01_20170717_102001006264A100_1020010066A25800_501591396070_01_P008_501591395050_01_P007_2_dem.tif
Ortho had wv_correct applied, rescaling values to range [0, 2047.0]
Planimetric Correction Iteration 0
Offset (z,x,y): 0.000, 0.000, 0.000
/att/gpfsfs/hic101/ppl/ehusby/scratch/repos/setsm_postprocessing_python/lib/scenes2strips.py:589: RuntimeWarning: invalid value encountered in less_equal
  & (np.abs(dz - np.nanmedian(dz)) <= np.nanstd(dz)))
RMSE = 0.5270087499035556
Planimetric Correction Iteration 1
Offset (z,x,y): -0.084, -0.050, -0.020
RMSE = 0.518523311804853
Planimetric Correction Iteration 2
Offset (z,x,y): -0.082, -0.062, -0.026
RMSE = 0.5183195659664994
RMSE step in this iteration (-0.00020) is above threshold (-0.001), stopping and returning values of prior iteration
Final offset (z,x,y): -0.084, -0.050, -0.020
Final RMSE = 0.518523311804853
Scene 3 of 15: /home/ehusby/scratch/data/setsm_results/tif_results/2m/WV01_20170717_102001006264A100_1020010066A25800_501591396070_01_P007_501591395050_01_P007_2_dem.tif
...
```

```
...
Scene 14 of 15: /home/ehusby/scratch/data/setsm_results/tif_results/2m/WV01_20170717_102001006264A100_1020010066A25800_501591396070_01_P002_501591395050_01_P001_2_dem.tif
Ortho had wv_correct applied, rescaling values to range [0, 2047.0]
Planimetric Correction Iteration 0
Offset (z,x,y): 0.000, 0.000, 0.000
RMSE = 1.2451133001688222
Planimetric Correction Iteration 1
Offset (z,x,y): -0.575, -0.201, -0.206
RMSE = 1.0759884256897048
Planimetric Correction Iteration 2
Offset (z,x,y): -0.552, -0.273, -0.278
RMSE = 1.0733802282249287
Planimetric Correction Iteration 3
Offset (z,x,y): -0.544, -0.299, -0.303
RMSE = 1.0727809259739933
RMSE step in this iteration (-0.00060) is above threshold (-0.001), stopping and returning values of prior iteration
Final offset (z,x,y): -0.552, -0.273, -0.278
Final RMSE = 1.0733802282249287
Final RMSE is greater than cutoff value (1.0733802282249287 > 1.0), segment break
DEM: /home/ehusby/scratch/data/setsm_results/strips/2m/WV01_20170717_102001006264A100_1020010066A25800_seg1_2m_dem.tif
                # of verts              perimeter               area
in              54                      122283.38                       543235306.50
out             40                      122283.36                       543231644.50
-----------------------------------------------------
change  -25.93%                 -0.00%                  -0.00%

Saving Geotiff /home/ehusby/scratch/data/setsm_results/strips/2m/WV01_20170717_102001006264A100_1020010066A25800_seg1_2m_dem.tif ... GDAL data type: Float32, NoData value: -9999, Creation Options: BIGTIFF=IF_SAFER COMPRESS=LZW TILED=YES, Projection (Proj4): +proj=utm +zone=39 +south +datum=WGS84 +units=m +no_defs ... creating file ... writing array values ... finishing file ... done!
/att/gpfsfs/hic101/ppl/ehusby/scratch/repos/setsm_postprocessing_python/lib/raster_array_tools.py:462: UserWarning: NumPy array data type (bool) does not have equivalent GDAL data type and is not supported, but can be safely promoted to uint8
  "supported, but can be safely promoted to {}".format(dtype_np, promote_dtype(1).dtype))
Saving Geotiff /home/ehusby/scratch/data/setsm_results/strips/2m/WV01_20170717_102001006264A100_1020010066A25800_seg1_2m_matchtag.tif ... GDAL data type: Byte, NoData value: 0, Creation Options: BIGTIFF=IF_SAFER COMPRESS=LZW TILED=YES, Projection (Proj4): +proj=utm +zone=39 +south +datum=WGS84 +units=m +no_defs ... creating file ... writing array values ... finishing file ... done!
/att/gpfsfs/hic101/ppl/ehusby/scratch/repos/setsm_postprocessing_python/lib/raster_array_tools.py:669: UserWarning: Input array NumPy data type (uint16) differs from output NumPy data type (int16)
  "NumPy data type ({})".format(dtype_in, dtype_out(1).dtype))
Saving Geotiff /home/ehusby/scratch/data/setsm_results/strips/2m/WV01_20170717_102001006264A100_1020010066A25800_seg1_2m_ortho.tif ... GDAL data type: Int16, NoData value: 0, Creation Options: BIGTIFF=IF_SAFER COMPRESS=LZW TILED=YES, Projection (Proj4): +proj=utm +zone=39 +south +datum=WGS84 +units=m +no_defs ... creating file ... writing array values ... finishing file ... done!
Saving Geotiff /home/ehusby/scratch/data/setsm_results/strips/2m/WV01_20170717_102001006264A100_1020010066A25800_seg1_2m_bitmask.tif ... GDAL data type: Byte, NoData value: 0, Creation Options: BIGTIFF=IF_SAFER COMPRESS=LZW TILED=YES, Projection (Proj4): +proj=utm +zone=39 +south +datum=WGS84 +units=m +no_defs ... creating file ... writing array values ... finishing file ... done!
```
Notice that the first strip segment break occurs due to a final RMSE value from the coregistration step that is greater than the allowed maximum RMSE as specified by the script argument `--rmse-cutoff`.
```
Building segment 2
Running s2s with coregistration filter options: None
Ordering 2 scenes
Scene 1 of 2: /home/ehusby/scratch/data/setsm_results/tif_results/2m/WV01_20170717_102001006264A100_1020010066A25800_501591396070_01_P001_501591395050_01_P001_2_dem.tif
Ortho had wv_correct applied, rescaling values to range [0, 2047.0]
Scene 2 of 2: /home/ehusby/scratch/data/setsm_results/tif_results/2m/WV01_20170717_102001006264A100_1020010066A25800_501591396070_01_P002_501591395050_01_P001_2_dem.tif
Ortho had wv_correct applied, rescaling values to range [0, 2047.0]
Planimetric Correction Iteration 0
Offset (z,x,y): 0.000, 0.000, 0.000
RMSE = 1.282425844029874
Planimetric Correction Iteration 1
Offset (z,x,y): -0.193, -0.051, 0.018
RMSE = 1.2663068445114265
Planimetric Correction Iteration 2
Offset (z,x,y): -0.188, -0.070, 0.025
RMSE = 1.2660253821993022
RMSE step in this iteration (-0.00028) is above threshold (-0.001), stopping and returning values of prior iteration
Final offset (z,x,y): -0.193, -0.051, 0.018
Final RMSE = 1.2663068445114265
Final RMSE is greater than cutoff value (1.2663068445114265 > 1.0), segment break
DEM: /home/ehusby/scratch/data/setsm_results/strips/2m/WV01_20170717_102001006264A100_1020010066A25800_seg2_2m_dem.tif
                # of verts              perimeter               area
in              62                      34723.49                        80326893.00
out             44                      34723.44                        80320557.00
-----------------------------------------------------
change  -29.03%                 -0.00%                  -0.01%

Saving Geotiff /home/ehusby/scratch/data/setsm_results/strips/2m/WV01_20170717_102001006264A100_1020010066A25800_seg2_2m_dem.tif ... GDAL data type: Float32, NoData value: -9999, Creation Options: BIGTIFF=IF_SAFER COMPRESS=LZW TILED=YES, Projection (Proj4): +proj=utm +zone=39 +south +datum=WGS84 +units=m +no_defs ... creating file ... writing array values ... finishing file ... done!
...
```
Another segment break occurs soon after the first, resulting in a total of three strip segments created for the whole overlap area of stereo collections referenced by strip-pair ID WV01_20170717_102001006264A100_1020010066A25800.
```
...
Building segment 3
Running s2s with coregistration filter options: None
Ordering 1 scenes
Scene 1 of 1: /home/ehusby/scratch/data/setsm_results/tif_results/2m/WV01_20170717_102001006264A100_1020010066A25800_501591396070_01_P002_501591395050_01_P001_2_dem.tif
Ortho had wv_correct applied, rescaling values to range [0, 2047.0]
DEM: /home/ehusby/scratch/data/setsm_results/strips/2m/WV01_20170717_102001006264A100_1020010066A25800_seg3_2m_dem.tif
                # of verts              perimeter               area
in              25                      26046.56                        25047709.00
out             19                      26046.54                        25045397.00
-----------------------------------------------------
change  -24.00%                 -0.00%                  -0.01%

Saving Geotiff /home/ehusby/scratch/data/setsm_results/strips/2m/WV01_20170717_102001006264A100_1020010066A25800_seg3_2m_dem.tif ... GDAL data type: Float32, NoData value: -9999, Creation Options: BIGTIFF=IF_SAFER COMPRESS=LZW TILED=YES, Projection (Proj4): +proj=utm +zone=39 +south +datum=WGS84 +units=m +no_defs ... creating file ... writing array values ... finishing file ... done!
/att/gpfsfs/hic101/ppl/ehusby/scratch/repos/setsm_postprocessing_python/lib/raster_array_tools.py:462: UserWarning: NumPy array data type (bool) does not have equivalent GDAL data type and is not supported, but can be safely promoted to uint8
  "supported, but can be safely promoted to {}".format(dtype_np, promote_dtype(1).dtype))
Saving Geotiff /home/ehusby/scratch/data/setsm_results/strips/2m/WV01_20170717_102001006264A100_1020010066A25800_seg3_2m_matchtag.tif ... GDAL data type: Byte, NoData value: 0, Creation Options: BIGTIFF=IF_SAFER COMPRESS=LZW TILED=YES, Projection (Proj4): +proj=utm +zone=39 +south +datum=WGS84 +units=m +no_defs ... creating file ... writing array values ... finishing file ... done!
/att/gpfsfs/hic101/ppl/ehusby/scratch/repos/setsm_postprocessing_python/lib/raster_array_tools.py:669: UserWarning: Input array NumPy data type (uint16) differs from output NumPy data type (int16)
  "NumPy data type ({})".format(dtype_in, dtype_out(1).dtype))
Saving Geotiff /home/ehusby/scratch/data/setsm_results/strips/2m/WV01_20170717_102001006264A100_1020010066A25800_seg3_2m_ortho.tif ... GDAL data type: Int16, NoData value: 0, Creation Options: BIGTIFF=IF_SAFER COMPRESS=LZW TILED=YES, Projection (Proj4): +proj=utm +zone=39 +south +datum=WGS84 +units=m +no_defs ... creating file ... writing array values ... finishing file ... done!
Saving Geotiff /home/ehusby/scratch/data/setsm_results/strips/2m/WV01_20170717_102001006264A100_1020010066A25800_seg3_2m_bitmask.tif ... GDAL data type: Byte, NoData value: 0, Creation Options: BIGTIFF=IF_SAFER COMPRESS=LZW TILED=YES, Projection (Proj4): +proj=utm +zone=39 +south +datum=WGS84 +units=m +no_defs ... creating file ... writing array values ... finishing file ... done!

Fin!
2, python -u /att/gpfsfs/hic101/ppl/ehusby/scratch/repos/setsm_postprocessing_python/batch_scenes2strips.py --dst "/home/ehusby/scratch/data/setsm_results/strips/2m" --mask-ver "bitmask" --save-coreg-step "off" --rmse-cutoff 1.0 --stripid "WV03_20160731_1040010020191100_104001001F543C00" "/home/ehusby/scratch/data/setsm_results/tif_results/2m/" 2.0
...
```

</details>
<br/>

Now that strip results exist in the `--dst` directory, rerunning the previous command should yield no new results. We receive the following output:
```
(my_root) ehusby@pgc:~/scratch/repos/setsm_postprocessing_python$ python batch_scenes2strips.py ~/scratch/data/setsm_results/tif_results/2m/ 2
--dst dir set to: /home/ehusby/scratch/data/setsm_results/strips/2m
Found 3 *dem.tif strip-pair IDs, 0 unfinished
No unfinished strip DEMs found to process, exiting
```

Let's say that next we would like to create completely *unfiltered* strips (only doing the bare minimum crop to remove bad data around the edges of each scene). Recognize that our previous run has already done the brunt of the work to align the scenes for each strip during the iterative coregistration step. Instead of doing this intensive process over again (which should achieve the same results) we will use the `--meta-trans-dir` argument to pull the translation values from the "Mosaicking Alignment Statistics" section of the *meta.txt* files in the strip results folder of the previous run. We run the following command with a new `--dst` directory and test our processing options with `--dryrun` again:
```
(my_root) ehusby@pgc:~/scratch/repos/setsm_postprocessing_python$ python batch_scenes2strips.py ~/scratch/data/setsm_results/tif_results/2m/ 2 --nowater --nocloud --dst ~/scratch/data/setsm_results/strips/2m_filt001 --meta-trans-dir ~/scratch/data/setsm_results/strips/2m/ --dryrun
Found 3 *dem.tif strip-pair IDs, 3 unfinished
Sleeping 5 seconds before task submission
1, python -u /att/gpfsfs/hic101/ppl/ehusby/scratch/repos/setsm_postprocessing_python/batch_scenes2strips.py --dst "/home/ehusby/scratch/data/setsm_results/strips/2m_filt001" --meta-trans-dir "/home/ehusby/scratch/data/setsm_results/strips/2m/" --mask-ver "bitmask" --nowater --nocloud --save-coreg-step "off" --rmse-cutoff 1.0 --dryrun --stripid "WV01_20170717_102001006264A100_1020010066A25800" "/home/ehusby/scratch/data/setsm_results/tif_results/2m/" 2.0
2, python -u /att/gpfsfs/hic101/ppl/ehusby/scratch/repos/setsm_postprocessing_python/batch_scenes2strips.py --dst "/home/ehusby/scratch/data/setsm_results/strips/2m_filt001" --meta-trans-dir "/home/ehusby/scratch/data/setsm_results/strips/2m/" --mask-ver "bitmask" --nowater --nocloud --save-coreg-step "off" --rmse-cutoff 1.0 --dryrun --stripid "WV03_20160731_1040010020191100_104001001F543C00" "/home/ehusby/scratch/data/setsm_results/tif_results/2m/" 2.0
3, python -u /att/gpfsfs/hic101/ppl/ehusby/scratch/repos/setsm_postprocessing_python/batch_scenes2strips.py --dst "/home/ehusby/scratch/data/setsm_results/strips/2m_filt001" --meta-trans-dir "/home/ehusby/scratch/data/setsm_results/strips/2m/" --mask-ver "bitmask" --nowater --nocloud --save-coreg-step "off" --rmse-cutoff 1.0 --dryrun --stripid "WV03_20161014_10400100231DFD00_104001002363A300" "/home/ehusby/scratch/data/setsm_results/tif_results/2m/" 2.0
```

Remove the `--dryrun` option and let's run this:
<details>
<summary>[click to view output]</summary>
<br/>
  
```
(my_root) ehusby@pgc:~/scratch/repos/setsm_postprocessing_python$ python batch_scenes2strips.py ~/scratch/data/setsm_results/tif_results/2m/ 2 --nowater --nocloud --dst ~/scratch/data/setsm_results/strips/2m_filt001 --meta-trans-dir ~/scratch/data/setsm_results/strips/2m/
Found 3 *dem.tif strip-pair IDs, 3 unfinished
Sleeping 5 seconds before task submission
1, python -u /att/gpfsfs/hic101/ppl/ehusby/scratch/repos/setsm_postprocessing_python/batch_scenes2strips.py --dst "/home/ehusby/scratch/data/setsm_results/strips/2m_filt001" --meta-trans-dir "/home/ehusby/scratch/data/setsm_results/strips/2m/" --mask-ver "bitmask" --nowater --nocloud --save-coreg-step "off" --rmse-cutoff 1.0 --stripid "WV01_20170717_102001006264A100_1020010066A25800" "/home/ehusby/scratch/data/setsm_results/tif_results/2m/" 2.0

stripid: WV01_20170717_102001006264A100_1020010066A25800
res: 2m
srcdir: /home/ehusby/scratch/data/setsm_results/tif_results/2m
dstdir: /home/ehusby/scratch/data/setsm_results/strips/2m_filt001
dstdir for coreg step: None
metadir: /home/ehusby/scratch/data/setsm_results/strips/2m
mask version: bitmask
mask name: bitmask
coreg filter options: ()
mask filter options: ('nowater', 'nocloud')
rmse cutoff: 1.0
dryrun: False

Processing strip-pair ID: WV01_20170717_102001006264A100_1020010066A25800, 15 scenes


All *_bitmask.tif scene masks have been created in source scene directory

Running scenes2strips

Building segment 1
Running s2s with masking filter options: nowater, nocloud
Scene 1 of 13: /home/ehusby/scratch/data/setsm_results/tif_results/2m/WV01_20170717_102001006264A100_1020010066A25800_501591396070_01_P008_501591395050_01_P008_2_dem.tif
Ortho had wv_correct applied, rescaling values to range [0, 2047.0]
Scene 2 of 13: /home/ehusby/scratch/data/setsm_results/tif_results/2m/WV01_20170717_102001006264A100_1020010066A25800_501591396070_01_P008_501591395050_01_P007_2_dem.tif
Ortho had wv_correct applied, rescaling values to range [0, 2047.0]
Planimetric Correction Iteration 0
Offset (z,x,y): -0.084, -0.050, -0.020
/att/gpfsfs/hic101/ppl/ehusby/scratch/repos/setsm_postprocessing_python/lib/scenes2strips.py:589: RuntimeWarning: invalid value encountered in less_equal
  & (np.abs(dz - np.nanmedian(dz)) <= np.nanstd(dz)))
RMSE = 0.518482315820321
Holding trans guess, stopping
Final offset (z,x,y): -0.084, -0.050, -0.020
Final RMSE = 0.518482315820321
Scene 3 of 13: /home/ehusby/scratch/data/setsm_results/tif_results/2m/WV01_20170717_102001006264A100_1020010066A25800_501591396070_01_P007_501591395050_01_P007_2_dem.tif
Ortho had wv_correct applied, rescaling values to range [0, 2047.0]
Planimetric Correction Iteration 0
Offset (z,x,y): 0.036, 0.073, 0.044
RMSE = 0.5986345972889032
Holding trans guess, stopping
Final offset (z,x,y): 0.036, 0.073, 0.044
Final RMSE = 0.5986345972889032
Scene 4 of 13: /home/ehusby/scratch/data/setsm_results/tif_results/2m/WV01_20170717_102001006264A100_1020010066A25800_501591396070_01_P007_501591395050_01_P006_2_dem.tif
Ortho had wv_correct applied, rescaling values to range [0, 2047.0]
Planimetric Correction Iteration 0
Offset (z,x,y): 0.268, 0.120, 0.094
RMSE = 0.5143300607857831
Holding trans guess, stopping
Final offset (z,x,y): 0.268, 0.120, 0.094
Final RMSE = 0.5143300607857831
`rmse` out of `coregisterdems` does not match `rmse_guess` when rounded to 2 decimals
`rmse_guess`:
[[0.     0.5185 0.5987 0.5209 0.6018 0.4558 0.4091 0.3627 0.4855 0.5172 0.5606 0.7244 0.7292]]
`rmse`
[[0.     0.5185 0.5986 0.5143 0.     0.     0.     0.     0.     0.     0.     0.     0.    ]]
Scene 5 of 13: /home/ehusby/scratch/data/setsm_results/tif_results/2m/WV01_20170717_102001006264A100_1020010066A25800_501591396070_01_P006_501591395050_01_P006_2_dem.tif
...
```
The `--meta-trans-dir` option pulls translation values out of the *meta.txt* files in the specified directory of old strip results. This allows the program to skip the intensive iteration procedure in the coregistration step as indicated by the statement `Holding trans guess, stopping`. However, an RMSE value is still calculated as normal using the provided translation values on the scene rasters as they are mosaicked with *the particular filter options applied in this run*. Since we are creating unfiltered strips in this run using the translation values from our previous run that created the filtered versions (which is the same procedure that is done under the hood when going straight to creating unfiltered strips if the `--meta-trans-dir` and `--nofilter-coreg` arguments are not provided), it is not only reasonable but *expected* that the RMSE values will differ between the filtered and unfiltered versions. The program only warns the user when the RMSE differs beyond two decimals in value and prints a list of all given and newly-calculated RMSE values for the current strip segment up to and including the current scene. In this warning, `'rmse_guess'` and `'rmse'` refer to RMSE values pulled from the *meta.txt* files and the new RMSE values calculated during this run, respectively.
```
...
Scene 12 of 13: /home/ehusby/scratch/data/setsm_results/tif_results/2m/WV01_20170717_102001006264A100_1020010066A25800_501591396070_01_P003_501591395050_01_P002_2_dem.tif
Ortho had wv_correct applied, rescaling values to range [0, 2047.0]
Planimetric Correction Iteration 0
Offset (z,x,y): 0.297, -0.038, -0.041
RMSE = 0.7254524828191551
Holding trans guess, stopping
Final offset (z,x,y): 0.297, -0.038, -0.041
Final RMSE = 0.7254524828191551
`rmse` out of `coregisterdems` does not match `rmse_guess` when rounded to 2 decimals
`rmse_guess`:
[[0.     0.5185 0.5987 0.5209 0.6018 0.4558 0.4091 0.3627 0.4855 0.5172 0.5606 0.7244 0.7292]]
`rmse`
[[0.     0.5185 0.5986 0.5143 0.6025 0.4558 0.4109 0.3633 0.4852 0.5172 0.5602 0.7255 0.    ]]
Scene 13 of 13: /home/ehusby/scratch/data/setsm_results/tif_results/2m/WV01_20170717_102001006264A100_1020010066A25800_501591396070_01_P002_501591395050_01_P002_2_dem.tif
Ortho had wv_correct applied, rescaling values to range [0, 2047.0]
Planimetric Correction Iteration 0
Offset (z,x,y): 0.060, -0.090, -0.073
RMSE = 0.7272856924697136
Holding trans guess, stopping
Final offset (z,x,y): 0.060, -0.090, -0.073
Final RMSE = 0.7272856924697136
DEM: /home/ehusby/scratch/data/setsm_results/strips/2m_filt001/WV01_20170717_102001006264A100_1020010066A25800_seg1_2m_dem.tif
                # of verts              perimeter               area
in              58                      122307.30                       543250098.50
out             41                      122307.26                       543246404.50
-----------------------------------------------------
change  -29.31%                 -0.00%                  -0.00%

Saving Geotiff /home/ehusby/scratch/data/setsm_results/strips/2m_filt001/WV01_20170717_102001006264A100_1020010066A25800_seg1_2m_dem.tif ... GDAL data type: Float32, NoData value: -9999, Creation Options: BIGTIFF=IF_SAFER COMPRESS=LZW TILED=YES, Projection (Proj4): +proj=utm +zone=39 +south +datum=WGS84 +units=m +no_defs ... creating file ... writing array values ... finishing file ... done!
...
```
Notice this run of the program knew to use only the first 13 scenes when building the first strip segment. This is because the strip segmentation and per-segment scene composition is already spelled out in the *meta.txt* files found in the provided `--meta-trans-dir` folder. This structure for the output strip results must be followed exactly during this run of the program for the old translation values parsed from the *meta.txt* files to be utilized. As mentioned in the "Commentary on script arguments" section above concerning the `--meta-trans-dir` argument, if a segment break occurs in this run where it did not occur in the previous run from which the *meta.txt* files originated, the program will fall back to running as if the argument was never provided for all remaining segments that will be created for this strip-pair ID.
```
...
Building segment 2
Running s2s with masking filter options: nowater, nocloud
Scene 1 of 1: /home/ehusby/scratch/data/setsm_results/tif_results/2m/WV01_20170717_102001006264A100_1020010066A25800_501591396070_01_P001_501591395050_01_P001_2_dem.tif
...
```
```
...
Building segment 3
Running s2s with masking filter options: nowater, nocloud
Scene 1 of 1: /home/ehusby/scratch/data/setsm_results/tif_results/2m/WV01_20170717_102001006264A100_1020010066A25800_501591396070_01_P002_501591395050_01_P001_2_dem.tif
Ortho had wv_correct applied, rescaling values to range [0, 2047.0]
DEM: /home/ehusby/scratch/data/setsm_results/strips/2m_filt001/WV01_20170717_102001006264A100_1020010066A25800_seg3_2m_dem.tif
                # of verts              perimeter               area
in              26                      26049.52                        25054109.00
out             19                      26049.50                        25051461.00
-----------------------------------------------------
change  -26.92%                 -0.00%                  -0.01%

Saving Geotiff /home/ehusby/scratch/data/setsm_results/strips/2m_filt001/WV01_20170717_102001006264A100_1020010066A25800_seg3_2m_dem.tif ... GDAL data type: Float32, NoData value: -9999, Creation Options: BIGTIFF=IF_SAFER COMPRESS=LZW TILED=YES, Projection (Proj4): +proj=utm +zone=39 +south +datum=WGS84 +units=m +no_defs ... creating file ... writing array values ... finishing file ... done!
/att/gpfsfs/hic101/ppl/ehusby/scratch/repos/setsm_postprocessing_python/lib/raster_array_tools.py:462: UserWarning: NumPy array data type (bool) does not have equivalent GDAL data type and is not supported, but can be safely promoted to uint8
  "supported, but can be safely promoted to {}".format(dtype_np, promote_dtype(1).dtype))
Saving Geotiff /home/ehusby/scratch/data/setsm_results/strips/2m_filt001/WV01_20170717_102001006264A100_1020010066A25800_seg3_2m_matchtag.tif ... GDAL data type: Byte, NoData value: 0, Creation Options: BIGTIFF=IF_SAFER COMPRESS=LZW TILED=YES, Projection (Proj4): +proj=utm +zone=39 +south +datum=WGS84 +units=m +no_defs ... creating file ... writing array values ... finishing file ... done!
/att/gpfsfs/hic101/ppl/ehusby/scratch/repos/setsm_postprocessing_python/lib/raster_array_tools.py:669: UserWarning: Input array NumPy data type (uint16) differs from output NumPy data type (int16)
  "NumPy data type ({})".format(dtype_in, dtype_out(1).dtype))
Saving Geotiff /home/ehusby/scratch/data/setsm_results/strips/2m_filt001/WV01_20170717_102001006264A100_1020010066A25800_seg3_2m_ortho.tif ... GDAL data type: Int16, NoData value: 0, Creation Options: BIGTIFF=IF_SAFER COMPRESS=LZW TILED=YES, Projection (Proj4): +proj=utm +zone=39 +south +datum=WGS84 +units=m +no_defs ... creating file ... writing array values ... finishing file ... done!
Saving Geotiff /home/ehusby/scratch/data/setsm_results/strips/2m_filt001/WV01_20170717_102001006264A100_1020010066A25800_seg3_2m_bitmask.tif ... GDAL data type: Byte, NoData value: 0, Creation Options: BIGTIFF=IF_SAFER COMPRESS=LZW TILED=YES, Projection (Proj4): +proj=utm +zone=39 +south +datum=WGS84 +units=m +no_defs ... creating file ... writing array values ... finishing file ... done!

Fin!
2, python -u /att/gpfsfs/hic101/ppl/ehusby/scratch/repos/setsm_postprocessing_python/batch_scenes2strips.py --dst "/home/ehusby/scratch/data/setsm_results/strips/2m_filt001" --meta-trans-dir "/home/ehusby/scratch/data/setsm_results/strips/2m/" --mask-ver "bitmask" --nowater --nocloud --save-coreg-step "off" --rmse-cutoff 1.0 --stripid "WV03_20160731_1040010020191100_104001001F543C00" "/home/ehusby/scratch/data/setsm_results/tif_results/2m/" 2.0
...
```
  
</details>
<br/>

*** It is important to note that the scene filtering step was skipped during this second run of the program because once a particular mask version (such as the *bitmask*) raster file is built alongside each scene in the `src` directory those mask files will remain in the directory past any termination of the program.

It should also be noted that the same two sets of filtered and unfiltered strip results could have been created during a single run of the program by utilizing the `--save-coreg-step` argument like so:
```
(my_root) ehusby@pgc:~/scratch/repos/setsm_postprocessing_python$ python batch_scenes2strips.py ~/scratch/data/setsm_results/tif_results/2m/ 2 --nowater --nocloud --save-coreg-step all
--dst dir set to: /home/ehusby/scratch/data/setsm_results/strips/2m
Found 3 *dem.tif strip-pair IDs, 3 unfinished
Sleeping 5 seconds before task submission
1, python -u /att/gpfsfs/hic101/ppl/ehusby/scratch/repos/setsm_postprocessing_python/batch_scenes2strips.py --dst "/home/ehusby/scratch/data/setsm_results/strips/2m" --mask-ver "bitmask" --nowater --nocloud --save-coreg-step "all" --rmse-cutoff 1.0 --stripid "WV01_20170717_102001006264A100_1020010066A25800" "/home/ehusby/scratch/data/setsm_results/tif_results/2m/" 2.0

stripid: WV01_20170717_102001006264A100_1020010066A25800
res: 2m
srcdir: /home/ehusby/scratch/data/setsm_results/tif_results/2m
dstdir: /home/ehusby/scratch/data/setsm_results/strips/2m
dstdir for coreg step: /home/ehusby/scratch/data/setsm_results/strips/2m_coreg_filt111
metadir: None
mask version: bitmask
mask name: bitmask
coreg filter options: ()
mask filter options: ('nowater', 'nocloud')
rmse cutoff: 1.0
dryrun: False

Processing strip-pair ID: WV01_20170717_102001006264A100_1020010066A25800, 15 scenes
...
```
After this run is complete, the unfiltered set of strip results is located in `/home/ehusby/scratch/data/setsm_results/strips/2m`. The location of the filtered set of (intermediate, yet complete) strip results was set automatically as `/home/ehusby/scratch/data/setsm_results/strips/2m_coreg_filt111`.


## batch_mask.py
Documentation forthcoming.


## diff_strips.py (WIP)
Documentation forthcoming.
