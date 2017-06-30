# Version 2.0; Ryan Shellberg, Erik Husby; Polar Geospatial Center, University of Minnesota; 2017

from __future__ import division
import math
import os
import re

import numpy as np
import shapely.geometry as geometry
from scipy import misc, ndimage, signal, spatial
from shapely.ops import polygonize, unary_union
from skimage import morphology as sk_morphology
from skimage import draw
from skimage.filters.rank import entropy

import raster_array_tools as rat


def generateMasks(matchFile):
    # TODO: Write docstring.

    # Find SETSM version.
    metaFile = matchFile.replace('matchtag.tif', 'meta.txt')
    if not os.path.isfile(metaFile):
        print "No meta file, assuming SETSM version > 2.0"
        setsmVersion = 3
    else:
        setsm_pattern = re.compile("SETSM Version=(.*)")
        meta = open(metaFile, 'r')
        line = meta.readline()
        setsmVersion = None
        while line != '':
            match = re.search(setsm_pattern, line)
            if match:
                setsmVersion = float(match.group(1).strip())
                break
            line = meta.readline()
        meta.close()
        if setsmVersion is None:
            setsmVersion = 2.03082016
            print "WARNING: Missing SETSM Version number in '{}'".format(metaFile)
            print "--> Using settings for default 'SETSM Version={}'".format(setsmVersion)

    matchtag, res = rat.oneBandImageToArray_res(matchFile)
    matchtag = matchtag.astype(np.bool)

    # TODO: Check that values in the ortho image are interpreted correctly
    # -t    throughout the entropy mask derivation process. Ian's note in 'entropyMask.m':
    # -t    % There is some sort of scaling difference between the wv_correct and non
    # -t    % wv_correct - applied othos that screws up the conversion to uint8 in
    # -t    % entropyfilt (which uses im2uint8) unless I convert to uint8 first. So
    # -t    % need to check if wvc applied.
    orthoFile = matchFile.replace('matchtag.tif', 'ortho.tif')
    ortho = rat.oneBandImageToArray(orthoFile)

    ###########################################################
    #### THIS PART IS LIKELY TO REQUIRE CONSTANT UPDATING!! ###
    ###########################################################

    # Set edgemask filtering parameters based on SETSM version and image resolution.
    if setsmVersion < 2.01292016:       # This comparison is ridiculously stupid, I know.
                                        # But what can ya do?
        n = int(math.floor(21*2/res))   # data density kernel [was "kernel_size"]
        Pmin = 0.8                      # data density threshold for masking [was "map_threshold"]
        Amin = int(2000/res)            # minimum data cluster area [was "cluster_area_min"]
        cf = 0.5                        # boundary curvature factor (0 = convex hull, 1 = point boundary)
        crop = n
    else:
        n = int(math.floor(101*2/res))
        Pmin = 0.99
        Amin = int(2000/res)
        cf = 0.5
        crop = n

    print "Step 1: Derive water mask using entropy filter"
    M0 = getEntropyMask(ortho, res)

    print "Step 2: Deriving edgemask"
    edgemask = getEdgeMask(matchtag, n, Pmin, Amin, cf, crop, M0)
    rat.saveArrayAsTiff(edgemask, matchFile.replace('matchtag.tif', 'edgemask.tif'),
                        like_rasterFile=matchFile)

    # Set datamask filtering parameters based on SETSM version and image resolution.
    if setsmVersion <= 2.0:
        n = int(math.floor(21*2/res))
        Pmin = 0.3
        Amin = 1000
        Amax = 10000
    else:
        n = int(math.floor(101*2/res))
        Pmin = 0.90
        Amin = 1000
        Amax = 1000

    ###########################################################
    ###########################################################

    print "Step 3: Deriving datamask"
    matchtag[~edgemask] = 0
    datamask = getDataDensityMask(matchtag, n, Pmin, Amin, Amax)
    rat.saveArrayAsTiff(datamask.astype(np.uint8), matchFile.replace('matchtag.tif', 'datamask.tif'),
                        like_rasterFile=matchFile)

    del matchtag, edgemask, datamask


def getEntropyMask(ortho, res):
    """
    From Ian's 'entropyMask.m':

    function M = entropyMask(orthoFile)
    % entropyMask classify areas of low entropy in an image such as water
    %
    % M = entropyMask(orthoFile) returns the low entropy classification mask
    % from the geotif image in orthoFile. Also checks whether wvc was applied
    % from the metafile.
    %
    % Ian Howat,ihowat@gmail.com, Ohio State
    % 13-Apr-2017 10:41:41
    """

    # Parameters
    pres = 8        # Resample to this resolution for processing for speed and smooth.
    Jthresh = 0.2   # Minimum entropy threshold. 0.2 seems to be good for water.
    minPix = 1000   # Clusters of mask and void pixels in the resampled image
                    # less than this will be removed.

    # TODO: See prior TODO in main()
    # -t    before "orthoFile = matchFile.replace('matchtag.tif', 'ortho.tif')".

    sz = ortho.shape
    bg = (ortho == 0)  # Image background mask.

    # Resize ortho to pres.
    if res != pres:
        ortho = rat.my_imresize(ortho, res/pres, 'cubic')

    # Subtraction image
    or_subtraction = (ndimage.maximum_filter1d(ortho, 5, axis=0)
                      - ndimage.minimum_filter1d(ortho, 5, axis=0))

    # Entropy image
    J = entropy(or_subtraction, np.ones((5,5)))

    M = (J < Jthresh)

    M =  sk_morphology.remove_small_objects(M,  min_size=minPix, connectivity=2)
    M = ~sk_morphology.remove_small_objects(~M, min_size=minPix, connectivity=2)

    # Resize ortho to 8m.
    if res != pres:
        M = rat.my_imresize(M, sz, 'nearest')

    M[np.where(bg)] = False

    return M


def getDataDensityMap(array, k):
    """
    Given a NumPy 2D boolean array, returns an array of the same size
    with each node describing the fraction of nodes containing ones
    within a [k x k]-size kernel (or "window") of the input array.
    """
    conv = signal.fftconvolve(array, np.ones((k, k)), mode='same')
    density_map = conv / float(k*k)

    return density_map


def getEdgeMask(matchtag, kernel_size, threshold_Pmin, cluster_Amin, curvature_factor, crop, M0):
    # TODO: Write docstring.

    print "Computing data density map"
    density_map = getDataDensityMap(matchtag, kernel_size)

    density_map = np.logical_or((density_map >= threshold_Pmin), M0)
    if ~np.any(density_map):
        print "No points exceed minimum density threshold. Returning."
        return density_map

    # Fill interior holes since we're just looking for edges here.
    density_map = ndimage.morphology.binary_fill_holes(density_map)
    # Get rid of small, isolated clusters of data.
    density_map = sk_morphology.remove_small_objects(density_map,
                                                     min_size=cluster_Amin, connectivity=2)

    if ~np.any(density_map):
        print "No clusters exceed minimum cluster area. Returning."
        return density_map

    # Find data coverage boundaries.
    density_boundaries = (density_map != ndimage.binary_erosion(density_map))
    boundary_points = np.argwhere(density_boundaries)
    del density_map, density_boundaries

    # ##################################################################
    # # Concave hull method, using Ken Clarkson's Hull (ANSI C program)
    # print "Computing concave hull"
    #
    # # Downsample boundary to < 10000 points, since Hull cannot handle any more than that.
    # downsample_factor = len(boundary_points) // 10000 + 1
    # bpoints_sample = boundary_points[::downsample_factor]
    #
    # # Write boundary points into temporary file.
    # bpoints_txt = open(EM_TEMPFILE_BPOINTS, 'w')
    # for p in bpoints_sample:
    #     bpoints_txt.write((str(p)[1:-1]).lstrip() + '\n')
    # bpoints_txt.close()
    #
    # # -A argument to first call of clarkson-hull is supposed to:
    # # "compute the alpha shape of the input, finding the smallest alpha
    # #  so that the sites are all contained in the alpha-shape."
    # # This value of alpha is printed on the command line after being computed,
    # # so our objective is to capture it.
    # command = "clarkson-hull -A -i{} -oF{}".format(EM_TEMPFILE_BPOINTS, EM_TEMPFILE_CCHULL)
    # alpha_pattern = re.compile("alpha=(.*)")
    # alpha = -1
    # for line in run_command(command):
    #     print line
    #     match = re.search(alpha_pattern, line)
    #     if match:
    #         alpha = float(match.group(1).strip())
    #
    # if alpha == -1:
    #     raise MaskingError("Initial call to clarkson-hull failed to create alpha shape for edgemask")
    #
    # # -aa argument specifies the alpha value to use in the second call to clarkson-hull.
    # # After testing, a value 100 times the alpha value reported from the -A run creates
    # # a similar concave hull to MATLAB's boundary function with curvature factor of 0.5.
    # command = "clarkson-hull -aa{} -i{} -oF{}".format(alpha*100, EM_TEMPFILE_BPOINTS, EM_TEMPFILE_CCHULL)
    # subprocess.check_call(command.split())
    #
    # hull_edges = np.loadtxt(EM_TEMPFILE_CCHULL+'-alf', dtype=int, skiprows=1)
    # hull_vertices = rat.connectEdges(hull_edges)
    # hull_edgepoints = bpoints_sample[hull_vertices]
    # row_coords, column_coords = draw.polygon(bpoints_sample[hull_vertices, 0],
    #                                          bpoints_sample[hull_vertices, 1])
    # ##################################################################

    # #############################
    # # Concave hull method (old)
    # print "Computing concave hull"
    # hull = alpha_shape(boundary_points, alpha=.007)
    # edge_coords_r, edge_coords_c = hull.exterior.coords.xy
    # row_coords, column_coords = draw.polygon(edge_coords_r,
    #                                          edge_coords_c)
    # #############################

    #############################
    # Convex hull method
    print "Computing convex hull"
    hull = spatial.ConvexHull(boundary_points)
    row_coords, column_coords = draw.polygon(hull.points[hull.vertices, 0],
                                             hull.points[hull.vertices, 1])
    # hull_points = boundary_points[hull.vertices]  # For testing purposes.
    # print hull_points
    #############################

    hull_array = np.zeros(matchtag.shape).astype(np.bool)
    hull_array[row_coords, column_coords] = 1
    del hull

    print "Eroding image"
    edgemask = ndimage.binary_erosion(hull_array, structure=np.ones((crop, crop)))

    return edgemask


def getDataDensityMask(matchtag, kernel_size, threshold_Pmin, cluster_Amin, cluster_Amax):
    # TODO: Write docstring.

    density_map = getDataDensityMap(matchtag, kernel_size)
    density_map = (density_map >= threshold_Pmin)

    if ~np.any(density_map):
        print "No points exceed minimum density threshold. Returning."
        return density_map

    # Remove small data clusters.
    density_mask = sk_morphology.remove_small_objects(density_map,
                                                      min_size=cluster_Amin, connectivity=2)
    # Remove small data gaps.
    density_mask = ~sk_morphology.remove_small_objects(~density_mask,
                                                       min_size=cluster_Amax, connectivity=2)

    return density_mask


# The following functions (alpha_shape and add_edge)
# are based on KEVIN DWYER's code found at the following URL:
# http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python/
def alpha_shape(coords, alpha):
    """
    Computes the alpha shape (concave hull) of a set of points.
    @param coords: array of coords
    @param alpha: alpha value to influence the
        gooey-ness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    tri = spatial.Delaunay(coords)
    edges = set()
    edge_points = []
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
        # Semi-perimeter of triangle
        s = (a + b + c)/2.0
        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)
        # Here's the radius filter.
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)

    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    union = unary_union(triangles)

    return union


def add_edge(edges, edge_points, coords, i, j):
    """
    Add a line between the i-th and j-th points,
    if not in the list already.
    """
    if (i, j) in edges or (j, i) in edges:
        return
    edges.add((i, j))
    edge_points.append(coords[[i, j]])



#########################
### Testing Functions ###
#########################


def test_DDM(matchFile):

    if not matchFile.endswith('.tif'):
        matchFile += '.tif'
    if not os.path.isfile(matchFile):
        matchFile_temp = os.path.join(rat.TESTDIR, matchFile)
        if os.path.isfile(matchFile_temp):
            matchFile = matchFile_temp
        else:
            raise rat.InvalidArgumentError("No such matchFile: '{}'".format(matchFile))

    generateMasks(matchFile)
