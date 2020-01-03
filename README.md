# setsm_postprocessing_python
Methods for filtering and merging DEMs produced by [SETSM](https://github.com/setsmdeveloper/SETSM "SETSM on GitHub") with stereo pairs of DigitalGlobe imagery.

The code for both filtering and merging (also referred to as "mosaicking" here) is largely ported from [MATLAB code written by Ian Howat](https://github.com/ihowat/setsm_postprocessing "setsm_postprocessing on GitHub"), glaciologist and professor of Earth Sciences at The Ohio State University (OSU).

Please direct any questions to the current manager of this repo, Erik Husby, by email at husby036 (a+) umn (d<>t) edu.

[(Click here for a shortcut to information on the *bitmask.tif* raster)](https://github.com/PolarGeospatialCenter/setsm_postprocessing_python#the-bitmasktif-raster-explained)


## Python Requirements
Scripts in this repo are intended to work with both Python 2.7 and Python 3.6/3.7+. If you get an error when attempting to run the code in any of these Python versions, please create an Issue in this repo.

### Python package dependencies
* NumPy
* SciPy
* scikit-image
* OpenCV
* GDAL (including OSGeo, OGR, OSR)
* Shapely
* tifffile (required only for debugging)

### Recommendations on Python environment setup with *conda*
You will likely have issues getting all of the dependencies installed and working together properly without using [conda](https://conda.io/docs/index.html "conda landing page"). Even if you do use conda, you may still run into issues with conflicting dependencies depending on your system (more on that later). Conda is an open source package manager that generally works very well at quickly spinning up the particular Python environment that you need to get a program running. I recommend installing the [Miniconda](https://conda.io/miniconda.html "Miniconda installers") distribution (a cleaner install that starts you off with only the conda package manager and Python + [Standard Library](https://docs.python.org/3/library/ "The Python Standard Library") modules) over the [Anaconda](https://www.anaconda.com/download/ "Anaconda installers") distribution (which installs ~100 scientific Python packages, most of which you will never use). You will have to pick between distros with Python 2.7 or Python 3.7 for your base install, but regardless of which version you choose now (and if you chose Anaconda or Miniconda), later you can set up an environment with the particular Python version and packages you need to get the job done. [This page offers both a good introduction to conda and a reference to basic conda commands.](https://conda.io/docs/user-guide/getting-started.html "Getting started with conda") 

Once conda has been installed and you've created the environment into which you will install the required packages (or you may decide to not create a new environment and just install the packages into your default 'base' environment), you can install all of the packages at once or individually in whatever order you like. If you'd like some steps to follow, I recommend installing them in the following order by running these commands:
1. `conda install numpy scipy`
2. `conda install shapely`
3. `conda install scikit-image`
4. `conda install gdal`
5. `conda install opencv`
6. (optional) `conda install -c conda-forge tifffile`

Between running and completing these install commands, I would check to make sure you can successfully import the appropriate modules into Python. As you progress through package installation, try running the following commands in your environment's Python interpreter:
1. `import numpy, scipy`
2. `import shapely`
3. `import skimage`
4. `import gdal, osgeo, ogr, osr`
5. `import cv2`
6. `import tifffile`

If any of the imports raise an error with a mysterious message, you may want to note what the error is and try Googling the error message to see what other people have said about it. Sometimes an issue with the package install (likely conflicting dependencies) will cause a module name to not be recognized at all in the Python interpreter. Regardless, these issues have a good chance of automatically being resolved on their own as you progress (since conda will change the versions of installed packages automatically in its deconflicting step), so if you're starting with a clean install and have nothing to lose if it doesn't work in the end, attempt to push forward until you have successfully installed all of the packages with `conda install` (again, you shouldn't need to install them in any specific order). If you continue to have issues importing all of the listed modules in the Python interpreter -- or worse, the `conda install` command doesn't complete successfully -- try installing the latest version of the dysfunctional package(s) through the [conda-forge](https://conda-forge.org/#about "conda-forge 'About' page") by adding the `-c conda-forge` optional argument to the `conda install` command. You may want to uninstall a broken package before upgrading, but I can't offer an exact solution for every system. If all else fails, try pinning the versions of installed packages that you know work and maybe do a [`conda install --force --no-update-dependencies`](https://conda.io/docs/commands/conda-install.html "`conda install` documentation") to install the remainder.


## batch_scenes2strips.py
Located in the root folder of the repo, this is the main post-processing script.

```
usage: batch_scenes2strips.py [-h] [--dst DST]
                              [--meta-trans-dir META_TRANS_DIR]
                              [--skip-browse] [--dem-type {lsf,non-lsf}]
                              [--mask-ver {maskv1,maskv2,rema2a,mask8m,bitmask}]
                              [--noentropy] [--nowater] [--nocloud] [--unf]
                              [--nofilter-coreg]
                              [--save-coreg-step {off,meta,all}]
                              [--rmse-cutoff RMSE_CUTOFF] [--scheduler {pbs}]
                              [--jobscript JOBSCRIPT] [--logdir LOGDIR]
                              [--email [EMAIL]] [--restart]
                              [--remove-incomplete] [--use-old-masks]
                              [--old-org] [--dryrun] [--stripid STRIPID]
                              src res
```

**Note:** Run `python batch_scenes2strips.py --help` to print all script options with basic descriptions of their usage.

### Version History

#### 3.0
The [3.0 branch of Ian Howat's setsm_postprocessing GitHub repo](https://github.com/ihowat/setsm_postprocessing/tree/3.0 "setsm_postprocessing, 3.0 branch") provided the source MATLAB code from which this Python translation of the scenes2strips process was born. All features of the original scenes2strips process as of "Latest commit 8e4fc3b on Aug 21, 2018" were considered in the translation. Running this Python script is comparable to running the original MATLAB scripts [batch_mask.m](https://github.com/ihowat/setsm_postprocessing/blob/3.0/batch_mask.m "setsm_postprocessing, batch_mask.m") (for 2-meter resolution DEMs) and [batch_scenes2strips.m](https://github.com/ihowat/setsm_postprocessing/blob/3.0/batch_scenes2strips.m "setsm_postprocessing, batch_scenes2strips.m"), in that order.

#### 3.1
* The first working version of the Python scenes2strips process as it was translated from the original MATLAB source code. Some small tweaks to the logic of the scenes2strips process in translation were necessary to get the results of the translated and original processes to match as closely as possible. Rigorous attempts were made to recreate MATLAB image processing functions in Python with correct handling of edge cases and comparable run times.
* Added options to leverage the PBS or SLURM job schedulers to run the scenes2strips process in parallel on a Linux compute cluster (with scripting paradigm borrowed from Claire Porter, thanks!).
* Added capability to create "unfiltered" strips (easiest done by passing the `--unf` script argument) in multiple processing scenarios in addition to the ability to create normal filtered strips.
* Added mosaicking of scene masks into a mask strip component along with the normal dem/matchtag/ortho strip components.
* Added ability to use different algorithms for scene filtering by invoking the `--mask-ver` script argument.
* Restructured filtering code that produced the default 2-meter resolution FLAT binary *mask.tif* scene mask from [setsm_postprocessing 3.0](https://github.com/ihowat/setsm_postprocessing/tree/3.0 "setsm_postprocessing, 3.0 branch") to create the component-ized [*bitmask.tif*](https://github.com/PolarGeospatialCenter/setsm_postprocessing_python#the-bitmasktif-raster-explained) mask, which is now the default mask for 2-meter SETSM DEMs.

#### 3.2
* Updated the coregistration function to match recent changes to the MATLAB source code. The coregistration function now reports error values for each (x,y,z) component of the returned translation vector. This information will be necessary in future efforts to create a new tile mosaicking process for SETSM DEMs.

#### 4
When building cross-track ("xtrack") strips:
* Added requirement that the second scene ortho image exists in source scene pairname folder.
* For each DEM scene, separate masks will be generated using each of the scene ortho images, and the resulting masks will be combined using logical OR before being written out as the expected single scene mask in the source scene directory.
* A new *ortho2.tif* second strip ortho image will be created in the output strip pairname folder.


### Turning scenes into strips

In the context of SETSM post-processing under the OSU-PGC co-developed processing scheme, a "scene" is what we call the set of result raster files and a metadata text file that are produced by SETSM after the program has been run on a pair of overlapping chunks of stereo DigitalGlobe satellite images. Each scene is composed of rasters with the filename suffixes *dem.tif*/*dem_smooth.tif*, *matchtag.tif*/*matchtag_mt.tif*, and *ortho.tif*, plus an auxiliary metadata text file ending with *meta.txt*. Since DigitalGlobe customers often don't receive a single collect as a whole long, gigantic image but instead can receive slightly-overlapping image chunks that together make up the whole collect, there is often a large number of combinations for overlapping stereo images to process between a pair of stereo collects (in DigitalGlobe terms, a pair of "catalog IDs"). After these overlapping stereo image "scenes" (some call these "subscenes" since the word "scene" is often attributed to the whole satellite collect) have been processed with SETSM, they should be stitched together to create the best possible representation of the entire area of overlap between the stereo collects -- the "strip" -- whence they came. That is what this script aims to do, in batch.


### Step 1: Source scene selection by "strip-pair ID"

In the OSU-PGC processing scheme, SETSM results have filenames like "WV01_20170717_102001006264A100_1020010066A25800_501591396070_01_P001_501591395050_01_P001_2_dem.tif", where this filename contains pieces of information from the (DigitalGlobe) source images. The beginning of this filename, "WV01_20170717_102001006264A100_1020010066A25800", contains the catalog IDs of the two stereo image collects from which two overlapping chunks were processed with SETSM to generate this scene DEM. (The other half of the filename contains the order numbers of the two collects and identifies which chunks -- otherwise known as the "part", indicated by "P001" for both collects in this example -- were processed in particular.) This filename prefix, which I refer to as the "strip-pair ID", is common to all scene DEMs generated from those two stereo collects and thus it is the base unit of batch processing with scenes2strips. When batch_scenes2strips.py is run on a source directory `src` with specified resolution `res`,  all unique strip-pair IDs are identified from the contents of that source folder (non-recursively) and each strip-pair ID that has scenes of the indicated resolution is sent off to the core scenes2strips program where its group of scene DEMs will be filtered and merged together into strip DEMs.


### Step 2: Scene filtering and creation of the *bitmask.tif* scene component

SETSM DEMs often contain "blunders" -- areas of bad data that result from the presence of occlusions (most often clouds), areas of limited bit-depth (dark spots like water and shadow), or differences between the input stereo images (could be from choppy water or fast-moving clouds). Automated filtering of the scene DEMs should be performed so that (1) blunders may be masked out before the scenes are coregistered and (2) blunders may be masked out of the resulting strip DEMs. This is where we often see bad data in DEM scenes:

* Around the edges of the scene, there will almost always be a thin border (with more or less straight internal edges) of bad data with a steep slope up or down from the edges of the scene to the region of good data. This border of bad data must be removed before merging.
* The dark and glossy nature of water, combined with random reflection effects from waves that change randomly between collects, causes areas of water cover in a scene DEM to contain random bad data.
* Cloud cover (such as opaque "popcorn" clouds) can cause bad data to show up in otherwise good DEMs. SETSM has been shown to create decent DEMs when looking through fog and other semi-transparent cloud cover, but they usually have a blanket of error shown as an artificial rough appearance. It should also be noted that it is possible for cloud cover to change between stereo collects, though usually the small time difference of ~2 minutes is too small of a window for effects of cloud movement to be apparent.
* Areas in shadow are generally too dark (too low bit-depth) for SETSM to perform well in its pattern-matching. The inside of shadow-covered areas may show up in the DEM with artificial (erroneous) roughness similar to fog-covered areas, and the edges of shadow-covered areas will often show up as apparent discontinuities (jumps) in the DEM.

To attempt to identify bad data, a filtering method that is based on thresholding of the scene DEM, matchtag, and ortho image (in the OSU-PGC processing scheme, the ortho is always the "left" or first input image to the SETSM program) as follows:

1. A "match point density" map is calculated from the matchtag raster. The calculation uses a square kernel of ones (as are all kernels henceforth) with side length (henceforth "size") that scales with the resolution of the input scene rasters as `int(math.floor(21*2/image_res))`.
2. All scene rasters (including the map calculated in (1)) are downsampled with bicubic interpolation to 8-meter resolution (or upsampled... but let's assume your scenes have resolution <= 8-meter).
3. Various calculations and thresholds are used on the downsampled rasters to derive a three-component bitmask identifying areas of bad data over EDGE, WATER, and CLOUD:

    * The EDGE mask is derived from a concave hull applied to the inverse of a filter for steep slopes in the DEM. Slope grade is calculated using `numpy.gradient`, averaged using the same kernel from (1), thresholded where pixels with mean slope greater than 1 are classified as high slopes, then high slopes are dilated by a kernel of size 8 to complete the high-slope filter. (The concave hull algorithm was created by me out of sheer desperation to obtain a Python function equivalent to MATLAB's `boundary` function. My algorithm uses what should be the side length compliment of a more traditional circular alpha shape to eat out the edges of the Delaunay Triangulation of the data points.) Since the high-slope filter picks up the bad edges of the scene DEM, we inverse this mask and find the concave hull the data we want to retain. The concave hull uses an alpha shape with size that is halfway between a (large) size that would barely leave the convex hull intact and a (small) size that would barely keep the hull from splitting into two pieces.
    * The WATER mask uses the panchromatic scene ortho image from SETSM to classify areas as water with the following logic: (low radiance AND match point density < 98%) OR low entropy in radiance. Radiance values are calculated directly from the digital numbers in the original ortho image by using the following conversion: ```L = GAIN * DN * (ABSCALFACTOR / EFFECTIVEBANDWIDTH) + OFFSET``` where `L` is radiance value, `DN` is the ortho image value, `GAIN` and `OFFSET` are obtained from a lookup table by satellite sensor, and `ABSCALFACTOR` and `EFFECTIVEBANDWIDTH` are read from the scene metadata file. Entropy is defined as `-sum(p.*log2(p))`, where `p` contains the normalized histogram counts of the filter window in `skimage.filters.rank.entropy`, given a kernel of size 5. A smoothed version of the radiance differential is fed to the entropy filter by taking the difference between a maximum filter and minimum filter with a kernel size of 5. Pixels with calculated entropy < 0.2 are then classified as low entropy. The low-entropy filter is post-processed to remove clusters with total # of pixels < 500, then dilated by a kernel of size 7. The low-radiance filter classifies as low radiance values < (5 if `MEANSUNEL < 30` else 20), and is also post-processed to remove clusters with total # of pixels < 500. After the logical combination of these filters, the final water mask is post-processed to remove small clusters and fill voids with total # of pixels < 500.
    * The CLOUD mask uses the scene DEM and ortho image to classify areas as cloud with the following logic: (radiance > 70 AND match point density < 90%) OR match point density < 60% OR high relative standard deviation in elevation. Standard deviation of elevation is calculated by use of moving average filters with a kernel of size 21. The threshold value for classification of high standard devation is adaptive; if the difference between the 80th and 20th percentile elevation values in the unfiltered scene DEM is <= (40, 50, 75, 100, inf) the threshold value is (10.5, 15, 19, 27, 50), respectively, matching on the first true result of <=. After the logical combination of these filters, the final cloud mask is post-processed to remove small clusters with total # of pixels < 1000. It is then eroded and dilated by kernels of size 31 and 21, respectively, to "remove thin borders caused by cliffs/ridges" (comment from original MATLAB code) before it is dilated again by a kernel of size 61. Finally, voids ("good data" in this context) with total # of pixels < 10,000 are filled.

4. All three masks are resampled with nearest interpolation to the original resolution of the scene rasters.
5. If the `--mask-ver=maskv2` script argument is provided, all three mask components are combined with logical OR to create a flat binary mask. If instead `--mask-ver=bitmask` (the default option), the three mask components are instead combined by essentially converting the (EDGE, WATER, CLOUD) mask values from binary True/False to integer (1, 2, 4)/0 and combining into a single bitmask using `numpy.bitwise_or`.
6. The output mask (with either *mask.tif* or *bitmask.tif* suffix if provided `--mask-ver` is either `maskv2` or `bitmask`) is saved in the same folder as its corresponding scene rasters whence it was derived.

When it comes time to load the mask into the scene merging part of the program (in either the coregistration or mosaicking steps) a final post-processing step of the flattened mask (with particular toggling of water and cloud filters for the current run of the program) is to fill voids ("good data" in this context) with total # of pixels < `500 * (8 / image_res)**2`.

***Important!***  Scene *(bit)mask.tif* files are always created during the filtering step of the program and are stored alongside the scene files in the `src` scene directory, so make sure you have write permission for `src`.  These masks are also mosaicked together so that strip `*(bit)mask.tif` files are created and stored in the `dst` directory alongside strip results.


### Step 3: Scene ordering and coregistration

Due to reliance on satellite sensor RPC models to determine the geolocation of the stereo images that are provided as input to SETSM, [the geolocation of the output scene DEMs has an absolute accuracy of approximately 4 meters in the horizontal and vertical planes](https://www.pgc.umn.edu/guides/arcticdem/introduction-to-arcticdem/ "Introduction to ArcticDEM: Weaknesses") (which is on the order of -- if not greater than -- the resolution of the DEMs themselves). Additionally, [SETSM applies an RPC bias removal that](https://mjremotesensing.wordpress.com/setsm/ "What is SETSM: RPC errors") is likely to disrupt any internal consistency in geolocation of scenes within the same strip. This means that in order to merge the scene DEMs into a strip, a coregistration procedure must be be employed to align the *overlapping* scenes before they can be stitched together. The particular method used in the scenes2strips program is an iterative coregistration procedure from the academic paper ["Co-registration and bias corrections of satellite elevation data sets for quantifying glacier thickness change" by C. Nuth and A. Kaab, 2011](https://www.the-cryosphere.net/5/271/2011/tc-5-271-2011.pdf "Online PDF of the paper"). The COREGISTRATION STEP of the program goes roughly as follows:

1. Scenes are queued using the geolocation information they came with out of SETSM (described above). Assuming all scenes are in the same projected coordinate system, they are ordered from grid south to north or from west to east -- whichever direction corresponds to the longest side of the rectangular extent of the scenes -- under the condition that the scene with the largest area of overlap to the cumulative rectangular extent of all scenes added up to that point* is picked (meaning that the ordering will only be from grid south to north or from west to east in a general sense).
2. Scene raster data is read in as NumPy arrays and is immediately masked (always with the edge filter and by default with the water and cloud filters, but the latter may be side-stepped by passing the `--nofilter-coreg` script option in conjunction with one or both of the `--nowater`/`--nocloud` options). If all DEM pixels have been masked, this scene is skipped and the next scene is loaded.
3. Scenes are loaded into the program sequentially in the order that was determined in step (1) and, granted that there is enough area of overlap with the scene(s) that were loaded in prior*, the area of overlap between the new scene and the existing strip is isolated. If the new scene does not have enough new data to add to the strip, it is skipped. Otherwise, if the overlapping sections of both layers have areas of at least 90% match point density*, the areas of overlap are fed to the coregistration routine designed by Nuth and Kaab as mentioned above.
4. In the coregistration routine, a difference of DEM values in the area of overlap between the new scene and the existing strip is calculated for pixels that are marked as match points in both layers. An RMSE value is then calculated for this difference map and is reported. Alignment correction of the two layers is done by calculating DEM slope in the x and y directions for the difference points and placing these values in the coefficient matrix of a least squares equation with the difference values as the dependent variable. Solving this least squares equation yields a new set of x, y, and z offset values (also known as "translation"/"trans" values from interpretation as a translation vector) for shifting the new scene into better alignment with the strip. This shift is applied through a 2D grid linear interpolation of the overlap area for the new scene and the two layers are differenced again. This process is repeated in an iterative fashion until change in RMSE between two consecutive iterations is > -0.001, at which point the RMSE and offset values for the penultimate iteration are returned.
5. If the RMSE value is not greater than the script argument value `--rmse-cutoff` (with default value of 1.0)*, the offset vector is subtracted from the entirety of the new scene through 2D grid interpolation and the aligned output rasters are combined with the strip rasters. For the DEM and ortho image rasters, a feathering approach with weighting that scales linearly over the area of overlap is used to seamlessly blend the two pieces together. For the matchtag and mask rasters, a bitwise OR combination is used.
6. Once all scenes that can be added to the strip (segment) have been added*, the output strip rasters are written to the `--dst` destination directory.
7. If for any of the general reasons indicated by (*) a scene cannot be added to a strip, a "segment break" is enacted and strip-building halts while what has been built of the current strip rasters, which now make up one of multiple strip *segments*, are written to disk as in step (6) (with the exception of a segment break detected in the queueing process in step (1), where processing simply begins with a partial queue). All scenes that were either skipped or successfully added to the saved strip are then removed from the processing queue and the strip-building process begins again at step (1) with the reduced queue and an incremented strip segment number. *In the worst case, there can be N number of output strip segments for N number of input scenes matching the same strip-pair ID.*


### Step 3/4: Merging of scenes into strip segment(s)

If you've read everything up to this point, (1) you are awesome and (2) you should be a bit confused by the title of this "Step 3/4" because you would expect the strip-building process to be complete after completion of the sub-steps outlined in "Step 3: Scene ordering and coregistration". The problem is this: What if you want the same strip coregistration and merging process applied as in Step 3, but without the masking out of "bad data" (which *may not* be bad in actuality) in the output strip rasters?

It turns out the easiest way to modify the scenes2strips routine to support the creation of "unfiltered" strips was to barely change it at all and simply run the routine twice: the first pass (the "coregistration step") gets the scene ordering and corresponding offset values from coregistration, then a second pass (the "mosaicking step") applies the same scene ordering and offset correction to scenes with different filtering options applied (specified by script options `--nowater` and/or `--nocloud`). (A modification that would allow for creating unfiltered strips in a single pass would either require significantly more memory or smarter memory management.) With the script argument `--save-coreg-step` you may choose to save the results of the first pass in full (by specifying `--save-coreg-step=all`) or only the metadata text files (with `--save-coreg-step=meta`).

If you have already created *filtered* strips during an earlier run of the program and would like to use the same ordering and offset values to create unfiltered strips, you can do so by providing the `--meta-trans-dir` script argument. This will allow the program to skip the coregistration step and go straight to the mosaicking step. Beware that if the program runs into any unplanned segment breaks for a particular strip during a run with this argument, all remaining segments for that strip will be run through both the coregistration and mosaicking steps.

While it is not recommended, it is possible to turn off filtering during the coregistration step and essentially roll both the coregistration and mosaicking steps into one by providing the `--nofilter-coreg` option. See the "Commentary on script arguments" section below for more details on usage of this option.

**Note:** If you are unsure if you will want filtered and/or unfiltered strips, it is computationally cheaper to create unfiltered strips first. If you are not tight on disk space and can spare generating the two sets of strips in a single run, use the `--save-coreg-step=all` option. Otherwise, you can use the `batch_mask.py` script included in the root directory of the repo (alongside `batch_scenes2strips.py`) to apply the water and/or cloud filter(s) saved in the *bitmask.tif* raster files to mask out the corresponding pixels in the strip (or scene) result rasters and quickly generate a new set of filtered results that way.


### Completion of strip processing and creation of the *.fin* file
It is possible for all data to be masked out in every scene for a particular strip, resulting in no output strip rasters being created. It is also possible for the writing of output strip files to fail due to program interruption/termination. Because the possibility of these cases occurring during batch runs of the scenes2strips program on potentially thousands of strips (which we do at PGC) is very much nonzero, a short and sweet *{strip-pair ID}.fin* file listing the filenames of all scene DEMs worked on for that strip is created in the strip destination directory at the end of processing each strip-pair ID. The purpose of this file's existence is simply to let the user know with confidence that processing of the indicated strip-pair ID ran to completion, and to definitively list which scene DEMs went into making it.


### Error handling
If the scenes2strips process ends abruptly due to any error, an automatic attempt will be made to remove any scene masks and/or strip result data for the strip-pair ID that was being worked on, as such result data would be incomplete and must be reprocessed. If this automatic removal fails, incomplete strips (indicated by the absence of the corresponding *.fin* file from the strip results folder) can be identified and removed using the `--remove-incomplete` argument.


### Notes on batch job submission to scheduler
Use the `--scheduler` script argument to submit to your system's job scheduler the processing of all strip-pair IDs found in the source directory, each as a separate job. Currently, only the PBS and SLURM job schedulers are supported. You may BYOJS(job script) using the `--jobscript` argument, but certain conditions apply that are stated in the "Commentary on script arguments" section below. Otherwise, a default jobscript will be selected from the 'jobscripts' folder in the repo root directory that corresponds to this script and the indicated scheduler type.

**Note:** Provide the `--email` option to have an email sent to you when the last submitted job in the batch either completes or errors out. Just providing `--email` invokes the scheduler's internal mail option for the submitted job, but you can also specify an email address to send an additional email to the provided address using Python standard libraries to send the email (with an error trace in the email’s body if something went wrong). You don’t need to be running the batch with a job scheduler for the latter email to attempt to send.


### Commentary on script arguments

```
usage: batch_scenes2strips.py [-h] [--dst DST]
                              [--meta-trans-dir META_TRANS_DIR]
                              [--skip-browse] [--dem-type {lsf,non-lsf}]
                              [--mask-ver {maskv1,maskv2,rema2a,mask8m,bitmask}]
                              [--noentropy] [--nowater] [--nocloud] [--unf]
                              [--nofilter-coreg]
                              [--save-coreg-step {off,meta,all}]
                              [--rmse-cutoff RMSE_CUTOFF] [--scheduler {pbs}]
                              [--jobscript JOBSCRIPT] [--logdir LOGDIR]
                              [--email [EMAIL]] [--restart]
                              [--remove-incomplete] [--use-old-masks]
                              [--old-org] [--dryrun] [--stripid STRIPID]
                              src res

Filters scene DEMs in a source directory, then mosaics them into strips and saves the results. 
Batch work is done in units of strip-pair IDs, as parsed from scene dem filenames (see --stripid argument for how this is parsed).

positional arguments:
  src                   Path to source directory containing scene DEMs to process. If --dst is not specified, this path should contain the folder 'tif_results'. The range of source scenes worked on may be limited with the --stripid argument.
  res                   Resolution of target DEMs in meters.

optional arguments:
  -h, --help            show this help message and exit
  --dst DST             Path to destination directory for output mosaicked strip data. (default is src.(reverse)replace('tif_results', 'strips')) (default: None)
  --meta-trans-dir META_TRANS_DIR
                        Path to directory of old strip metadata from which translation values will be parsed to skip scene coregistration step. (default: None)
  --skip-browse         TURN OFF building of 10m hillshade *_dem_browse.tif browse images of all output DEM strip segments after they are created inside --dst directory. (default: False)
  --dem-type {lsf,non-lsf}
                        Which version of all scene DEMs to work with. 
                        'lsf': Use the LSF DEM with 'dem_smooth.tif' file suffix. 
                        'non-lsf': Use the non-LSF DEM with 'dem.tif' file suffix. 
                         (default: lsf)
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
  --unf                 Shortcut for setting ['--nowater', '--nocloud'] options to create "unfiltered" strips. 
                        Default for --dst argument becomes (src.(reverse)replace('tif_results', 'strips_unf')). 
                        Can only be used when --mask-ver=bitmask. (default: False)
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
                        **Note:** Due to implementation difficulties, this directory will also become the working directory for the job process. Since relative path inputs are always changed to absolute paths in this script, this should not be an issue. (default: None)
  --email [EMAIL]       Send email to user upon end or abort of the LAST SUBMITTED task. (default: None)
  --restart             Remove any unfinished (no .fin file) output before submitting all unfinished strips. (default: False)
  --remove-incomplete   Only remove unfinished (no .fin file) output, do not build strips. (default: False)
  --use-old-masks       Use existing scene masks instead of deleting and re-filtering. (default: False)
  --old-org             Use old scene and strip results organization in flat directory structure,prior to reorganization into strip pairname folders. (default: False)
  --dryrun              Print actions without executing. (default: False)
  --stripid STRIPID     Run filtering and mosaicking for a single strip with strip-pair ID as parsed from scene DEM filenames using the following regex: '(^[A-Z0-9]{4}_.*?_?[0-9A-F]{16}_.*?_?[0-9A-F]{16}).*$' 
                        A text file containing a list of strip-pair IDs, each on a separate line,may instead be provided for batch processing of select strips. (default: None)
```

* `src` :: In the OSU-PGC processing scheme, this is a path to a `*/tif_results/8m` or `*/tif_results/2m` folder for 8-meter or 2-meter DEMs, respectively. For 50-centimeter processing, it doesn't matter if the name of the lowest folder is "50cm" or "0.5m" or whatever.
* `res` :: NUMERIC INPUT; This value must be provided in METERS! The input resolution value is used to (1) make sure that the selected scenes in the `src` source directory are indicated by their filenames to be of the same resolution, (2) check if strip results already exist in the `--dst` destination directory *and skip processing the strip if any results already exist*, (3) make sure the selected `--mask-ver` filter scheme works with the resolution, and (4) is the resolution included in the filenames of the output strip results files. For more information on how condition (1) is enforced (as )
* `--dst` :: In the OSU-PGC processing scheme, this will be a path to a `*/strips/8m` or `*/strips/2m` folder where the `*` is the same path as given in the commentary for `src`. If this argument isn't provided an attempt is made to determine this path by default (as specified in the `--help` text), so if you're following the OSU-PGC processing scheme you don't ever need to provide this argument. However, if you think you will be creating both filtered and unfiltered versions of strips (through the `--nowater`/`--nocloud` arguments), I recommend making this path `*/strips/2m_filtXXX` where \[`XXX`\] is the bit-code corresponding to filter components (\[cloud, water, edge\], respectively) applied during the final mosaicking step of the scenes2strips process. The bit-code for completely filtered strips is thus `111` while for completely "unfiltered" strips (both `--nowater` and `--nocloud` provided) it is `001`.
* `--meta-trans-dir` :: This option is useful if either (1) you need to reprocess/recreate strips for some reason and you trust that the coregistration of the scenes in the earlier processing is still sound or (2) you were running scenes2strips with `--save-coreg-step` set to either `meta` or `all` when a crash/abort caused processing to end prematurely and you want to avoid redoing the processing of the coregistration step. If provided, the program will attempt to read the translation values for scenes from old strip segment *meta.txt* files in the provided directory and use these values for assembling strips in lieu of running the coregistration step. If it happens that the scenes2strips process decides to break the new strip into segments differently than it broke for the old strip results in the `--meta-trans-dir` directory, the program will continue to create a new segment(s) as if the argument was never provided *for only that particular strip*.
* `--skip-browse` :: By default, 10-meter resolution hillshades are automatically generated for every output strip segment DEM with the filename suffix *dem_browse.tif*. This is done by calling the *gdal_translate* and *gdaldem hillshade* programs, so these must be callable from the current shell environment. *gdal_translate* creates a 10-meter downsampled version of the DEM (with filename suffix *dem_10m.tif*) as a temporary file that is fed to *gdaldem hillshade*. By supplying this argument to `batch_scenes2strips.py`, you may turn off this additional processing step.
* `--dem-type` :: Either of the scene DEM versions `*dem.tif` (the DEM as it came straight out of SETSM) or `*dem_smooth.tif` (in which  `*dem.tif` has been smoothed for noise using the LSF method) can be used along with the other matchtag/orto/meta.txt scene components to create a strip. Which version of strips you should create may depend on the land cover type captured in the DEM and/or the intended application for the DEM results.
* `--mask-ver` :: Improvements to the scenes2strips filtering step are currently focusing solely on the *bitmask* version that creates the *bitmask.tif* scene/strip mask component raster. Only change this option from its default value (`bitmask`) if for some reason you're interested in seeing what the old masks look like.
* `--noentropy` :: As noted in the `--help` text, this argument can only be provided along with `--mask-ver=maskv1`. That mask version is deprecated, so use of this argument is unlikely.
* `--nowater`/`--nocloud` :: These arguments allow for producing unfiltered strip results, which is what really differentiates this version of the scenes2strips process from the earlier MATLAB version. See subsection "Step 2: Scene filtering and creation of the *bitmask.tif* scene component" above for more information.
* `--unf` :: This argument is a shortcut for setting the `--nowater` and `--nocloud` arguments to create "unfiltered" strips.
* `--nofilter-coreg` :: By default, all filters (edge, water, and cloud) are applied during the coregistration step with the assumption that the filters remove only bad data and lead to a better coregistration. If the filters are instead removing more good data than bad data, a better coregistration may be achieved by providing this argument to turn off the offending filters during the coregistration step. (Note that the edge filter must be applied in both the coregistration and mosaicking steps because it is known that DEMs produced by SETSM have a border of bad data around their edges that needs to be cropped off before merging.)
* `--save-coreg-step` :: When the scenes2strips process is split into separate coregistration and mosaicking steps due to providing `--nowater`/`--nocloud`, this option allows for caching the results of the coregistration step. Set this option to `all` if you want the full unfiltered version of strip output in addition to a filtered version. Set it to `meta` if you want to make sure to have a backup of the translation values for `--meta-trans-dir` in case of a crash. Also note that the strip segment *meta.txt* files saved with this option will have different values for the the RMSE component of the "Mosaicking Alignment Statistics" than the output strip meta files saved in the `--dst` directory because the RMSE statistic is properly recalculated for the unfiltered version of the strip during the mosaicking step. For conditions on when this argument applies, see the `--help` text.
* `--rmse-cutoff` :: After the iterative coregistration step is complete, the final RMSE value for the coregistration is reported. If that RMSE value is greater than the value specified by this argument, the scene that failed to register to the strip will become the start of a new strip segment.
* `--scheduler` :: Currently, only the PBS and SLURM job schedulers are supported. If you provide this argument, note that if you do not specify a particular PBS/SLURM jobscript to run with the `--jobscript` argument a default jobscript will be selected from the 'jobscripts' folder in the repo root directory that corresponds to this script and the indicated scheduler type.
* `--jobscript` :: REQUIREMENTS: The jobscript MUST (1) be readable by the provided `--scheduler` job scheduler type, (2) load the Python environment that includes all required packages as specified above under "Python package dependencies" before it (3) executes the main command that runs the script for a single `--stripid` by means of substituting the entire command into the jobscript through the environment variable `$p1`.
* `--logdir` :: If this argument is NOT provided and you are using the default jobscripts from this repo, the default output log file directory for SLURM is the directory where the command to run the script was submitted, while for PBS it is the `$HOME` directory of the user who submitted the command.
* `--email` :: If this option is provided (with or without an email address) and you are using the default jobscripts from this repo, an email will be sent to the email address tied to your user account upon end or abort of ONLY the job that was submitted last in the batch (to avoid spamming you with emails). This relies on the the mail option provided by the selected job scheduler, which is not supported on every system. If an email address is also provided with this option, an additional email will be sent upon end or error of the last submitted task/job using `email.mime.text.MIMEText` with `smtplib` from the Python Standard Library. 
* `--restart` :: See notes for `--remove-incomplete`.
* `--remove-incomplete` :: If the scenes2strips process ends abruptly due to any error, an automatic attempt will be made to remove any strip result data for the strip-pair ID that was being worked on, as such result data would be incomplete and must be reprocessed. If this automatic removal fails, incomplete strips (indicated by the absence of the corresponding *.fin* file from the strip results folder) can be identified and removed using this argument.
* `--use-old-masks` :: Old scene masks being used to create new strips can cause issues when either the other scene DEM components are updated while the old scene mask remains, or the filtering algorithm is updated and an incomplete run of scenes2strips resulted in partial scene mask creation such that two mask versions could exist for the same strip-pair ID. To avoid these issues, any scene mask files that exist before strip creation are deleted and will be recreated during the filtering step.
* `--old-org` :: Prior to July 26, 2019, scripts in this repo assumed the organization of all source and destination scene and strip files was flat within the `*/tif_results/2m` and `*/strips/2m` folders, respectively. After July 26, 2019, both scene and strip files are organized within proper strip-ID pairname folders like `*/tif_results/2m/{strip-pair ID}_2m` for scenes and `*/strips/2m/{strip-pair ID}_2m_lsf` for strips built with LSF-version DEM scenes. This argument must be provided to use the older organizational scheme.
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
(my_root) ehusby@pgc:/mnt/usr/ehusby/scratch/repos/setsm_postprocessing_python$ python batch_scenes2strips.py ~/scratch/data/setsm_results/tif_results/2m/ 2 --dryrun
argument --dst set automatically to: /home/ehusby/scratch/data/setsm_results/strips/2m
Found 3 *dem.tif strip-pair IDs, 3 unfinished
Sleeping 5 seconds before task submission
1, python -u /mnt/usr/ehusby/scratch/repos/setsm_postprocessing_python/batch_scenes2strips.py --dst "/home/ehusby/scratch/data/setsm_results/strips/2m" --mask-ver "bitmask" --save-coreg-step "off" --rmse-cutoff 1.0 --dryrun --stripid "WV01_20170717_102001006264A100_1020010066A25800" "/home/ehusby/scratch/data/setsm_results/tif_results/2m" 2
2, python -u /mnt/usr/ehusby/scratch/repos/setsm_postprocessing_python/batch_scenes2strips.py --dst "/home/ehusby/scratch/data/setsm_results/strips/2m" --mask-ver "bitmask" --save-coreg-step "off" --rmse-cutoff 1.0 --dryrun --stripid "WV03_20160731_1040010020191100_104001001F543C00" "/home/ehusby/scratch/data/setsm_results/tif_results/2m" 2
3, python -u /mnt/usr/ehusby/scratch/repos/setsm_postprocessing_python/batch_scenes2strips.py --dst "/home/ehusby/scratch/data/setsm_results/strips/2m" --mask-ver "bitmask" --save-coreg-step "off" --rmse-cutoff 1.0 --dryrun --stripid "WV03_20161014_10400100231DFD00_104001002363A300" "/home/ehusby/scratch/data/setsm_results/tif_results/2m" 2
```
The program found three sets of scenes that it can attempt to merge together to create (segments of) three distinct strips. Since we did not provide the `--scheduler` argument to specify a job scheduler for processing these strips, the program will simply run each printed command in serial from the parent process. Notice how the most meaningful (non-`None`) script arguments are passed from the parent process to each child command, even default script arguments that you did not specify. This should not be a cause for concern because the child process runs the same script as the parent and would pick up the same default script arguments anyways.

Let's try removing the `--dryrun` option and giving it a go:
<details>
<summary>[click to view output]</summary>
<br/>
  
```
(my_root) ehusby@pgc:/mnt/usr/ehusby/scratch/repos/setsm_postprocessing_python$ python batch_scenes2strips.py ~/scratch/data/setsm_results/tif_results/2m/ 2
argument --dst set automatically to: /home/ehusby/scratch/data/setsm_results/strips/2m
Creating argument --dst directory: /home/ehusby/scratch/data/setsm_results/strips/2m
Found 3 *dem.tif strip-pair IDs, 3 unfinished
Sleeping 5 seconds before task submission
1, python -u /mnt/usr/ehusby/scratch/repos/setsm_postprocessing_python/batch_scenes2strips.py --dst "/home/ehusby/scratch/data/setsm_results/strips/2m" --mask-ver "bitmask" --save-coreg-step "off" --rmse-cutoff 1.0 --stripid "WV01_20170717_102001006264A100_1020010066A25800" "/home/ehusby/scratch/data/setsm_results/tif_results/2m" 2

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
/mnt/usr/ehusby/scratch/repos/setsm_postprocessing_python/lib/filter_scene.py:1163: RuntimeWarning: invalid value encountered in greater
  mask = (mean_slope_array > 1)
/mnt/usr/ehusby/scratch/repos/setsm_postprocessing_python/lib/filter_scene.py:1417: RuntimeWarning: invalid value encountered in less
  stdev_elev_array[stdev_elev_array < 0] = 0
20/80 percentile elevation difference: 986.9, sigma-z threshold: 50.0
/mnt/usr/ehusby/scratch/repos/setsm_postprocessing_python/lib/filter_scene.py:1447: RuntimeWarning: invalid value encountered in greater
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
/mnt/usr/ehusby/scratch/repos/setsm_postprocessing_python/lib/scenes2strips.py:589: RuntimeWarning: invalid value encountered in less_equal
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
Writing output strip segment with DEM: /home/ehusby/scratch/data/setsm_results/strips/2m/WV01_20170717_102001006264A100_1020010066A25800_seg1_2m_dem.tif
                # of verts              perimeter               area
in              54                      122283.38                       543235306.50
out             40                      122283.36                       543231644.50
-----------------------------------------------------
change  -25.93%                 -0.00%                  -0.00%

Saving Geotiff /home/ehusby/scratch/data/setsm_results/strips/2m/WV01_20170717_102001006264A100_1020010066A25800_seg1_2m_dem.tif ... GDAL data type: Float32, NoData value: -9999, Creation Options: BIGTIFF=IF_SAFER COMPRESS=LZW TILED=YES, Projection (Proj4): +proj=utm +zone=39 +south +datum=WGS84 +units=m +no_defs ... creating file ... writing array values ... finishing file ... done!
/mnt/usr/ehusby/scratch/repos/setsm_postprocessing_python/lib/raster_array_tools.py:462: UserWarning: NumPy array data type (bool) does not have equivalent GDAL data type and is not supported, but can be safely promoted to uint8
  "supported, but can be safely promoted to {}".format(dtype_np, promote_dtype(1).dtype))
Saving Geotiff /home/ehusby/scratch/data/setsm_results/strips/2m/WV01_20170717_102001006264A100_1020010066A25800_seg1_2m_matchtag.tif ... GDAL data type: Byte, NoData value: 0, Creation Options: BIGTIFF=IF_SAFER COMPRESS=LZW TILED=YES, Projection (Proj4): +proj=utm +zone=39 +south +datum=WGS84 +units=m +no_defs ... creating file ... writing array values ... finishing file ... done!
/mnt/usr/ehusby/scratch/repos/setsm_postprocessing_python/lib/raster_array_tools.py:669: UserWarning: Input array NumPy data type (uint16) differs from output NumPy data type (int16)
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
Writing output strip segment with DEM: /home/ehusby/scratch/data/setsm_results/strips/2m/WV01_20170717_102001006264A100_1020010066A25800_seg2_2m_dem.tif
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
Writing output strip segment with DEM: /home/ehusby/scratch/data/setsm_results/strips/2m/WV01_20170717_102001006264A100_1020010066A25800_seg3_2m_dem.tif
                # of verts              perimeter               area
in              25                      26046.56                        25047709.00
out             19                      26046.54                        25045397.00
-----------------------------------------------------
change  -24.00%                 -0.00%                  -0.01%

Saving Geotiff /home/ehusby/scratch/data/setsm_results/strips/2m/WV01_20170717_102001006264A100_1020010066A25800_seg3_2m_dem.tif ... GDAL data type: Float32, NoData value: -9999, Creation Options: BIGTIFF=IF_SAFER COMPRESS=LZW TILED=YES, Projection (Proj4): +proj=utm +zone=39 +south +datum=WGS84 +units=m +no_defs ... creating file ... writing array values ... finishing file ... done!
/mnt/usr/ehusby/scratch/repos/setsm_postprocessing_python/lib/raster_array_tools.py:462: UserWarning: NumPy array data type (bool) does not have equivalent GDAL data type and is not supported, but can be safely promoted to uint8
  "supported, but can be safely promoted to {}".format(dtype_np, promote_dtype(1).dtype))
Saving Geotiff /home/ehusby/scratch/data/setsm_results/strips/2m/WV01_20170717_102001006264A100_1020010066A25800_seg3_2m_matchtag.tif ... GDAL data type: Byte, NoData value: 0, Creation Options: BIGTIFF=IF_SAFER COMPRESS=LZW TILED=YES, Projection (Proj4): +proj=utm +zone=39 +south +datum=WGS84 +units=m +no_defs ... creating file ... writing array values ... finishing file ... done!
/mnt/usr/ehusby/scratch/repos/setsm_postprocessing_python/lib/raster_array_tools.py:669: UserWarning: Input array NumPy data type (uint16) differs from output NumPy data type (int16)
  "NumPy data type ({})".format(dtype_in, dtype_out(1).dtype))
Saving Geotiff /home/ehusby/scratch/data/setsm_results/strips/2m/WV01_20170717_102001006264A100_1020010066A25800_seg3_2m_ortho.tif ... GDAL data type: Int16, NoData value: 0, Creation Options: BIGTIFF=IF_SAFER COMPRESS=LZW TILED=YES, Projection (Proj4): +proj=utm +zone=39 +south +datum=WGS84 +units=m +no_defs ... creating file ... writing array values ... finishing file ... done!
Saving Geotiff /home/ehusby/scratch/data/setsm_results/strips/2m/WV01_20170717_102001006264A100_1020010066A25800_seg3_2m_bitmask.tif ... GDAL data type: Byte, NoData value: 0, Creation Options: BIGTIFF=IF_SAFER COMPRESS=LZW TILED=YES, Projection (Proj4): +proj=utm +zone=39 +south +datum=WGS84 +units=m +no_defs ... creating file ... writing array values ... finishing file ... done!

Completed processing for this strip-pair ID
.fin finished indicator file created: /home/ehusby/scratch/data/setsm_results/strips/2m/WV01_20170717_102001006264A100_1020010066A25800_2m.fin

2, python -u /mnt/usr/ehusby/scratch/repos/setsm_postprocessing_python/batch_scenes2strips.py --dst "/home/ehusby/scratch/data/setsm_results/strips/2m" --mask-ver "bitmask" --save-coreg-step "off" --rmse-cutoff 1.0 --stripid "WV03_20160731_1040010020191100_104001001F543C00" "/home/ehusby/scratch/data/setsm_results/tif_results/2m" 2
...
```

</details>
<br/>

Now that strip results exist in the `--dst` directory, rerunning the previous command should yield no new results. We receive the following output:
```
(my_root) ehusby@pgc:/mnt/usr/ehusby/scratch/repos/setsm_postprocessing_python$ python batch_scenes2strips.py ~/scratch/data/setsm_results/tif_results/2m/ 2
argument --dst set automatically to: /home/ehusby/scratch/data/setsm_results/strips/2m
Found 3 *dem.tif strip-pair IDs, 0 unfinished
No unfinished strip DEMs found to process, exiting
```

Let's say that next we would like to create completely *unfiltered* strips (only doing the bare minimum crop to remove bad data around the edges of each scene). Recognize that our previous run has already done the brunt of the work to align the scenes for each strip during the iterative coregistration step. Instead of doing this intensive process over again (which should achieve the same results) we will use the `--meta-trans-dir` argument to pull the translation values from the "Mosaicking Alignment Statistics" section of the *meta.txt* files in the strip results folder of the previous run. We run the following command with a new `--dst` directory and test our processing options with `--dryrun` again:
```
(my_root) ehusby@pgc:/mnt/usr/ehusby/scratch/repos/setsm_postprocessing_python$ python batch_scenes2strips.py ~/scratch/data/setsm_results/tif_results/2m/ 2 --nowater --nocloud --dst ~/scratch/data/setsm_results/strips/2m_filt001 --meta-trans-dir ~/scratch/data/setsm_results/strips/2m/ --dryrun
Found 3 *dem.tif strip-pair IDs, 3 unfinished
Sleeping 5 seconds before task submission
1, python -u /mnt/usr/ehusby/scratch/repos/setsm_postprocessing_python/batch_scenes2strips.py --dst "/home/ehusby/scratch/data/setsm_results/strips/2m_filt001" --meta-trans-dir "/home/ehusby/scratch/data/setsm_results/strips/2m" --mask-ver "bitmask" --nowater --nocloud --save-coreg-step "off" --rmse-cutoff 1.0 --dryrun --stripid "WV01_20170717_102001006264A100_1020010066A25800" "/home/ehusby/scratch/data/setsm_results/tif_results/2m" 2
2, python -u /mnt/usr/ehusby/scratch/repos/setsm_postprocessing_python/batch_scenes2strips.py --dst "/home/ehusby/scratch/data/setsm_results/strips/2m_filt001" --meta-trans-dir "/home/ehusby/scratch/data/setsm_results/strips/2m" --mask-ver "bitmask" --nowater --nocloud --save-coreg-step "off" --rmse-cutoff 1.0 --dryrun --stripid "WV03_20160731_1040010020191100_104001001F543C00" "/home/ehusby/scratch/data/setsm_results/tif_results/2m" 2
3, python -u /mnt/usr/ehusby/scratch/repos/setsm_postprocessing_python/batch_scenes2strips.py --dst "/home/ehusby/scratch/data/setsm_results/strips/2m_filt001" --meta-trans-dir "/home/ehusby/scratch/data/setsm_results/strips/2m" --mask-ver "bitmask" --nowater --nocloud --save-coreg-step "off" --rmse-cutoff 1.0 --dryrun --stripid "WV03_20161014_10400100231DFD00_104001002363A300" "/home/ehusby/scratch/data/setsm_results/tif_results/2m" 2
```

Remove the `--dryrun` option and let's run this:
<details>
<summary>[click to view output]</summary>
<br/>
  
```
(my_root) ehusby@pgc:/mnt/usr/ehusby/scratch/repos/setsm_postprocessing_python$ python batch_scenes2strips.py ~/scratch/data/setsm_results/tif_results/2m/ 2 --nowater --nocloud --dst ~/scratch/data/setsm_results/strips/2m_filt001 --meta-trans-dir ~/scratch/data/setsm_results/strips/2m/
Creating argument --dst directory: /home/ehusby/scratch/data/setsm_results/strips/2m_filt001
Found 3 *dem.tif strip-pair IDs, 3 unfinished
Sleeping 5 seconds before task submission
1, python -u /mnt/usr/ehusby/scratch/repos/setsm_postprocessing_python/batch_scenes2strips.py --dst "/home/ehusby/scratch/data/setsm_results/strips/2m_filt001" --meta-trans-dir "/home/ehusby/scratch/data/setsm_results/strips/2m" --mask-ver "bitmask" --nowater --nocloud --save-coreg-step "off" --rmse-cutoff 1.0 --stripid "WV01_20170717_102001006264A100_1020010066A25800" "/home/ehusby/scratch/data/setsm_results/tif_results/2m" 2

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
/mnt/usr/ehusby/scratch/repos/setsm_postprocessing_python/lib/scenes2strips.py:589: RuntimeWarning: invalid value encountered in less_equal
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
The `--meta-trans-dir` option pulls translation values out of the *meta.txt* files in the specified directory of old strip results. This allows the program to skip the intensive iteration procedure in the coregistration step as indicated by the statement `Holding trans guess, stopping`. However, an RMSE value is still calculated as normal using the provided translation values on the scene rasters as they are mosaicked with *the particular filter options applied in this run*. Since we are creating unfiltered strips in this run using the translation values from our previous run that created the filtered versions (which is the same procedure that is done under the hood when going straight to creating unfiltered strips if the `--meta-trans-dir` and `--nofilter-coreg` arguments are not provided), it is not only reasonable but *expected* that the RMSE values will differ between the filtered and unfiltered versions. The program only warns the user when the RMSE difference is greater than 10^-2 in magnitude and prints a list of all given and newly-calculated RMSE values for the current strip segment up to and including the current scene. In this warning, `'rmse_guess'` and `'rmse'` refer to RMSE values pulled from the *meta.txt* files and the new RMSE values calculated during this run, respectively.
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
Writing output strip segment with DEM: /home/ehusby/scratch/data/setsm_results/strips/2m_filt001/WV01_20170717_102001006264A100_1020010066A25800_seg1_2m_dem.tif
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
Ortho had wv_correct applied, rescaling values to range [0, 2047.0]
Writing output strip segment with DEM: /home/ehusby/scratch/data/setsm_results/strips/2m_filt001/WV01_20170717_102001006264A100_1020010066A25800_seg2_2m_dem.tif
                # of verts              perimeter               area
in              63                      34769.01                        80535885.00
out             43                      34768.93                        80528941.00
-----------------------------------------------------
change  -31.75%                 -0.00%                  -0.01%

Saving Geotiff /home/ehusby/scratch/data/setsm_results/strips/2m_filt001/WV01_20170717_102001006264A100_1020010066A25800_seg2_2m_dem.tif ... GDAL data type: Float32, NoData value: -9999, Creation Options: BIGTIFF=IF_SAFER COMPRESS=LZW TILED=YES, Projection (Proj4): +proj=utm +zone=39 +south +datum=WGS84 +units=m +no_defs ... creating file ... writing array values ... finishing file ... done!
...
```
```
...
Building segment 3
Running s2s with masking filter options: nowater, nocloud
Scene 1 of 1: /home/ehusby/scratch/data/setsm_results/tif_results/2m/WV01_20170717_102001006264A100_1020010066A25800_501591396070_01_P002_501591395050_01_P001_2_dem.tif
Ortho had wv_correct applied, rescaling values to range [0, 2047.0]
Writing output strip segment with DEM: /home/ehusby/scratch/data/setsm_results/strips/2m_filt001/WV01_20170717_102001006264A100_1020010066A25800_seg3_2m_dem.tif
                # of verts              perimeter               area
in              26                      26049.52                        25054109.00
out             19                      26049.50                        25051461.00
-----------------------------------------------------
change  -26.92%                 -0.00%                  -0.01%

Saving Geotiff /home/ehusby/scratch/data/setsm_results/strips/2m_filt001/WV01_20170717_102001006264A100_1020010066A25800_seg3_2m_dem.tif ... GDAL data type: Float32, NoData value: -9999, Creation Options: BIGTIFF=IF_SAFER COMPRESS=LZW TILED=YES, Projection (Proj4): +proj=utm +zone=39 +south +datum=WGS84 +units=m +no_defs ... creating file ... writing array values ... finishing file ... done!
/mnt/usr/ehusby/scratch/repos/setsm_postprocessing_python/lib/raster_array_tools.py:462: UserWarning: NumPy array data type (bool) does not have equivalent GDAL data type and is not supported, but can be safely promoted to uint8
  "supported, but can be safely promoted to {}".format(dtype_np, promote_dtype(1).dtype))
Saving Geotiff /home/ehusby/scratch/data/setsm_results/strips/2m_filt001/WV01_20170717_102001006264A100_1020010066A25800_seg3_2m_matchtag.tif ... GDAL data type: Byte, NoData value: 0, Creation Options: BIGTIFF=IF_SAFER COMPRESS=LZW TILED=YES, Projection (Proj4): +proj=utm +zone=39 +south +datum=WGS84 +units=m +no_defs ... creating file ... writing array values ... finishing file ... done!
/mnt/usr/ehusby/scratch/repos/setsm_postprocessing_python/lib/raster_array_tools.py:669: UserWarning: Input array NumPy data type (uint16) differs from output NumPy data type (int16)
  "NumPy data type ({})".format(dtype_in, dtype_out(1).dtype))
Saving Geotiff /home/ehusby/scratch/data/setsm_results/strips/2m_filt001/WV01_20170717_102001006264A100_1020010066A25800_seg3_2m_ortho.tif ... GDAL data type: Int16, NoData value: 0, Creation Options: BIGTIFF=IF_SAFER COMPRESS=LZW TILED=YES, Projection (Proj4): +proj=utm +zone=39 +south +datum=WGS84 +units=m +no_defs ... creating file ... writing array values ... finishing file ... done!
Saving Geotiff /home/ehusby/scratch/data/setsm_results/strips/2m_filt001/WV01_20170717_102001006264A100_1020010066A25800_seg3_2m_bitmask.tif ... GDAL data type: Byte, NoData value: 0, Creation Options: BIGTIFF=IF_SAFER COMPRESS=LZW TILED=YES, Projection (Proj4): +proj=utm +zone=39 +south +datum=WGS84 +units=m +no_defs ... creating file ... writing array values ... finishing file ... done!

Completed processing for this strip-pair ID
.fin finished indicator file created: /home/ehusby/scratch/data/setsm_results/strips/2m_filt001/WV01_20170717_102001006264A100_1020010066A25800_2m.fin

2, python -u /mnt/usr/ehusby/scratch/repos/setsm_postprocessing_python/batch_scenes2strips.py --dst "/home/ehusby/scratch/data/setsm_results/strips/2m_filt001" --meta-trans-dir "/home/ehusby/scratch/data/setsm_results/strips/2m" --mask-ver "bitmask" --nowater --nocloud --save-coreg-step "off" --rmse-cutoff 1.0 --stripid "WV03_20160731_1040010020191100_104001001F543C00" "/home/ehusby/scratch/data/setsm_results/tif_results/2m" 2
...
```
  
</details>
<br/>

*** It is important to note that the scene filtering step was skipped during this second run of the program because once a particular mask version (such as the *bitmask*) raster file is built alongside each scene in the `src` directory those mask files will remain in the directory past any termination of the program.

It should also be noted that the same two sets of filtered and unfiltered strip results could have been created during a single run of the program by utilizing the `--save-coreg-step` argument like so:
```
(my_root) ehusby@pgc:/mnt/usr/ehusby/scratch/repos/setsm_postprocessing_python$ python batch_scenes2strips.py ~/scratch/data/setsm_results/tif_results/2m/ 2 --nowater --nocloud --save-coreg-step all
argument --dst set automatically to: /home/ehusby/scratch/data/setsm_results/strips/2m
Creating argument --dst directory: /home/ehusby/scratch/data/setsm_results/strips/2m
Found 3 *dem.tif strip-pair IDs, 3 unfinished
Sleeping 5 seconds before task submission
1, python -u /mnt/usr/ehusby/scratch/repos/setsm_postprocessing_python/batch_scenes2strips.py --dst "/home/ehusby/scratch/data/setsm_results/strips/2m" --mask-ver "bitmask" --nowater --nocloud --save-coreg-step "all" --rmse-cutoff 1.0 --stripid "WV01_20170717_102001006264A100_1020010066A25800" "/home/ehusby/scratch/data/setsm_results/tif_results/2m" 2

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

**Note 1:** The `--dst` argument that specifies the strip output directory doesn’t need to be provided; if it is not provided, an attempt will be made to automatically derive a destination directory from the `src` directory path by replacing the last instance of `'tif_results'` in the `src` path with `'strips'`.

**Note 2:** The "_filtXXX" naming convention as a suffix for the `--dst` destination directory in the example command `python batch_scenes2strips.py ~/scratch/data/setsm_results/tif_results/2m/ 2 --nowater --nocloud --dst ~/scratch/data/setsm_results/strips/2m_filt001 --meta-trans-dir ~/scratch/data/setsm_results/strips/2m/ --dryrun` corresponds to the filtering options applied in that run as they may be interpreted from the *bitmask.tif* scene/strip component raster files. See the bonus section "The bitmask.tif raster, explained" below for more information about the bitmask raster specifics.

**Note 3:** If you are creating both filtered and unfiltered versions of strips and lose track of which is which (it is to prevent this that the "_filtXXX" naming convention is advised), this information is stored in the strip *meta.txt* result files. In the "Filtering Applied" section of the text file, the "bit" index and corresponding filter "class" is listed along with a value of "1" if the filter was applied during the "coreg"/"mosaic" coregistration/mosaicking step or a value of "0" if it was not applied.


### The *bitmask.tif* raster, explained
This is a UInt8 bitmask raster in which the three least significant bits (LSB) X-X-X (rightmost being the least significant bit) correspond to the presence of Cloud-Water-Edge components of the mask, respectively. When interpreting the pixel values in base 10, this means 0=good data, 1=edge, 2=water, 4=cloud, with integers in-between and up to 7 (1-1-1 in binary) meaning that pixel is covered by a combination of the three components. A lookup table is provided below.

Using this mask, the water and/or cloud filters as they are computed for each scene/strip during the filtering step of the scenes2strips program can optionally be applied to any (unfiltered) strip (or scene) DEMs after they have been created, using the `batch_mask.py` script. The edge component of the mask is always applied in both the coregistration and mosaicking steps of the scenes2strips program because currently bad data is always present on the edges of the scene DEMs when they come out of SETSM.

| **Bit Index (zero-based, from LSB)** | 3-7 | 2 | 1 | 0 |   |   |
| --- | --- | --- | --- | --- | --- | --- |
| **Bit Indication** | Not used | Cloud | Water | Edge |   |   |
|   | **Bit Value** | **Bit Value** | **Bit Value** | **Bit Value** | **Decimal Value** | **Interpretation** |
|   | 00000 | 0 | 0 | 0 | 0 | &quot;Good data&quot; |
|   | 00000 | 0 | 0 | 1 | 1 | Bad edge data |
|   | 00000 | 0 | 1 | 0 | 2 | Water |
|   | 00000 | 0 | 1 | 1 | 3 | Water and edge |
|   | 00000 | 1 | 0 | 0 | 4 | Cloud |
|   | 00000 | 1 | 0 | 1 | 5 | Cloud and edge |
|   | 00000 | 1 | 1 | 0 | 6 | Cloud and water |
|   | 00000 | 1 | 1 | 1 | 7 | Cloud, water, and edge |

#### Version History

##### 1.0
The first version of the bitmask.

##### 1.1
Improved concave hull algorithm to fix rare bugs. The edge mask now has considerably more definition as the concave hull algorithm is able to erode more of the convex hull.

##### 1.2
Modified the cloud mask function to prevent the pre-dilation mask base from overlapping with the finalized water mask. When the water mask is able to get a good representation of water bodies up to the coast/shore, this prevents the dilated cloud mask from unnecessarily eroding away good coast/shoreline.
