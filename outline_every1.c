/*
Modified by Erik Husby, Summer 2016, for use at the
Polar Geospatial Center, University of Minnesota.
 - assume data has already been copied to a larger mask
   to avoid the call to bitmask_draw
 - commented out blocks that would return NULL
Original code found at URL: http://stackoverflow.com/questions/14110904/numpy-binary-raster-image-to-polygon-transformation
*/

/*
Modifioitu pygame.mask.Mask.outline
Input: data, rows, cols, starty, startx
*/

PyObject *plist, *value;
int y, x, firsty, firstx, secy, secx, curry, currx, nexty, nextx, n;
int a[14], b[14];
a[0] = a[1] = a[7] = a[8] = a[9] = b[1] = b[2] = b[3] = b[9] = b[10] = b[11]= 1;
a[2] = a[6] = a[10] = b[4] = b[0] = b[12] = b[8] = 0;
a[3] = a[4] = a[5] = a[11] = a[12] = a[13] = b[5] = b[6] = b[7] = b[13] = -1;

plist = NULL;
plist = PyList_New (0);
/* En ymmärrä mihin tätä tarvii 
if (!plist) {
    return NULL;
}*/

n = firsty = firstx = secy = y = 0;

/* if(!PyArg_ParseTuple(args, "|i", &every)) {
    return NULL;
}*/

/* by copying to a new, larger mask, we avoid having to check if we are at
   a border pixel every time 
bitmask_draw(m, x, 1, 1);*/

if (starty > 0) {
    firsty = starty;
    firstx = startx;
}
else {
    /* find the first set pixel in the mask */
    for (x = 1; x < cols-1; x++) {
        for (y = 1; y < rows-1; y++) {
            if (data(y, x)) {
                 firsty = y;
                 firstx = x;
                 value = Py_BuildValue("(ii)", y, x);
                 PyList_Append(plist, value);
                 Py_DECREF(value);
                 break;
            }
        }
        if (data(y, x))
            break;
    }
}


/* if the start node is not set, return (an empty list) */
if (data(firsty, firstx)) {

    /* if a start node was used, add it to the list and set vars y, x */
    if (!y) {
        value = Py_BuildValue("(ii)", firsty, firstx);
        PyList_Append(plist, value);
        Py_DECREF(value);
        y = firsty;
        x = firstx;
    }

    /* covers the mask having zero pixels or only the final pixel
    Pikseleitä on ainakin kymmenen */
    if (!((y == rows-1) && (x == cols-1))) {

        /* check just the first pixel for neighbors */
        for (n = 0; n < 8; n++) {
            if (data(y+a[n], x+b[n])) {
                curry = secy = y+a[n];
                currx = secx = x+b[n];
                value = Py_BuildValue("(ii)", secy, secx);
                PyList_Append(plist, value);
                Py_DECREF(value);
                break;
            }
        }

        /* if there are no neighbors, return
        Pikseleitä on ainakin kymmenen */
        if (secy) {

            /* the outline tracing loop */
            for (;;) {
                /* look around the pixel, it has to have a neighbor */
                for (n = (n + 6) & 7;; n++) {
                    if (data(curry+a[n], currx+b[n])) {
                        nexty = curry+a[n];
                        nextx = currx+b[n];
                        if (currx == firstx && curry == firsty) {
                            if (pass_start) {
                                if (secy == nexty && secx == nextx)
                                    break;
                            }
                            else break;
                        }
                        value = Py_BuildValue("(ii)", nexty, nextx);
                        PyList_Append(plist, value);
                        Py_DECREF(value);
                        break;
                    }
                }
                if (currx == firstx && curry == firsty) {
                    if (pass_start) {
                        /* if we are back at the first pixel, and the next one will be the
                           second one we visited, we are done */
                        if (secy == nexty && secx == nextx)
                            break;
                    }
                    else break;
                }

                currx = nextx;
                curry = nexty;
            }
        }
    }
}

return_val = plist;