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

### Commentary on script usage

```
usage: batch_scenes2strips.py [-h] [--dst DST]
                              [--meta-trans-dir META_TRANS_DIR]
                              [--mask-ver {maskv1,maskv2,rema2a,mask8m,bitmask}]
                              [--noentropy] [--nowater] [--nocloud]
                              [--nofilter-coreg]
                              [--save-coreg-step {off,meta,all}]
                              [--rmse-cutoff RMSE_CUTOFF]
                              [--scheduler {pbs,slurm}]
                              [--jobscript JOBSCRIPT] [--dryrun]
                              [--stripid STRIPID]
                              src res

Filters scene dems in a source directory, then mosaics them into strips and saves the results.
 Batch work is done in units of strip-pair ID, as parsed from scene dem filenames (see --stripid argument for how this is parsed).

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
                         (default: meta)
  --rmse-cutoff RMSE_CUTOFF
                        Maximum RMSE from coregistration step tolerated for scene merging. A value greater than this causes a new strip segment to be created. (default: 1.0)
  --scheduler {pbs,slurm}
                        Submit tasks to job scheduler. (default: None)
  --jobscript JOBSCRIPT
                        Script to run in job submission to scheduler. (default scripts are found in [REPO-ROOT-DIR]/jobscripts)
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
* `--dryrun` :: Useful for testing which strips will be built where, without actually starting the process of building them.
* `--stripid` :: During normal batch usage of this script, you only specify the `src` source directory of all scenes that you aim to turn into multiple strips. This argument is then typically used only internally by the batch execution logic to specify which single strip each instance of the scenes2strips program will work on. If you have a ton of scenes (that can be made into multiple strips) in the `src` directory and wish to only create/recreate a specific strip, you can specify that strip using the OSU-PGC strip-pair ID naming convention described above under "Step 1: Source scene selection by "strip-pair ID"".

### Example commands & output

```
~/scratch/repos/setsm_postprocessing_python$ python batch_scenes2strips.py ~/scratch/data/setsm_results/tif_results/2m 2
Found 3 *dem.tif strip-pair IDs, 0 unfinished
No unfinished strip DEMs found to process, exiting
```


## batch_mask.py
Documentation coming soon.


## diff_strips.py (WIP)
Documentation forthcoming.
