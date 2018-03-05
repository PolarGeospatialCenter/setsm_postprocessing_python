import cProfile
import numpy as np

from skimage.morphology import convex_hull_image

from testing import test


function = 'convex_hull_image(image, offset_coordinates=True)'

chull_type = 'latest'

imageFname_in = 'image.tif'
imageFname_out = 'chull_{}.tif'.format(chull_type)
imageFname_debug = 'chull_{}_inspection.tif'.format(chull_type)

image = test.readImage(imageFname_in)

print("{} [{}]:\n".format(function, chull_type))
cProfile.run('image_chull = {}'.format(function))
test.saveImage(image_chull, imageFname_out)

debug_mask = np.zeros(image.shape, dtype=np.int8)
debug_mask[image_chull] = 1
debug_mask[image] += 2
test.saveImage(debug_mask, imageFname_debug)
