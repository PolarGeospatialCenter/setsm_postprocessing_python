
# Source file: DecimatePoly.m, "Decimate Polygon" package (MATLAB)
# Source author: Anton Semechko (a.semechko@gmail.com)
# Source date: Jan. 2011
# Source URL: https://www.mathworks.com/matlabcentral/fileexchange/34639-decimate-polygon

# Translator: Erik Husby (husby036@umn.edu)
# Translation date: Feb. 2018

# Source file header:
"""
function [C_out,i_rem]=DecimatePoly(C,opt)
% Reduce the complexity of a 2D simple (i.e. non-self intersecting), closed
% piecewise linear contour by specifying boundary offset tolerance.
% IMPORTANT: This function may not preserve the topology of the original
% polygon.
%
% INPUT ARGUMENTS:
%   - C     : N-by-2 array of polygon co-ordinates, such that the first,
%             C(1,:), and last, C(end,:), points are the same.
%   - opt   : opt can be specified in one of two ways:
%             ----------------------APPROACH #1 (default) -----------------
%             - opt : opt=[B_tol 1], where B_tol is the maximum acceptible
%                     offset from the original boundary, B_tol must be
%                     expressed in the same lenth units as the co-ords in
%                     C. Default setting is B_tol=Emin/2, where Emin is the
%                     length of the shortest edge.
%             ----------------------APPROACH #2----------------------------
%              - opt : opt=[P_tol 2], where P_tol is the fraction of the
%                      total number of polygon's vertices to be retained.
%                      Accordingly, P_tol must be a real number on the
%                      interval (0,1).
%
% OUTPUT:
%   - C_out : M-by-2 array of polygon coordinates.
%   - i_rem : N-by-1 logical array used to indicate which vertices were
%             removed during decimation.
%
% ALGORITHM:
% 1) For every vertex compute the boundary offset error.
% 2) Rank all vertics according to the error score from step 1.
% 3) Remove the vertex with the lowest error.
% 4) Recompute and accumulate the errors for the two neighbours adjacent to
%    the deleted vertex and go back to step 2.
% 5) Repeat step 2 to 4 until no more vertices can be removed or the number
%    of vertices has reached the desired number.
%
% AUTHOR: Anton Semechko (a.semechko@gmail.com)
% DATE: Jan.2011
%
"""


from numbers import Number
from sys import stdout
import numpy as np


class InvalidArgumentError(Exception):
    def __init__(self, msg=""):
        super(Exception, self).__init__(msg)


def DecimatePoly(C, B_tol=None, P_tol=None):
    """
    Reduce the complexity of a 2D simple (i.e. non-self intersecting), closed
    piecewise linear contour by specifying boundary offset tolerance.
    IMPORTANT: This function may not preserve the topology of the original
    polygon.

    Parameters
    ----------
    C : ndarray (N,2)
        N-by-2 array of polygon coordinates, such that the first,
        `C[0,:]`, and last, `C[-1,:]`, points are the same.
    B_tol : None, or float > 0
        Maximum acceptable offset from the original boundary.
        Must be expressed in the same length units as the coords in `C`.
        If None, the default setting is `B_tol=Emin/2`, where `Emin` is
        the length of the shortest edge.
    P_tol : None, or 0 < float < 1
        Fraction of the total number of polygon's vertices to be retained.

    Returns
    -------
    (C_out, i_rem) tuple
    C_out : ndarray (N,2)
        M-by-2 array of polygon coordinates.
        The first, `C_out[0,:]`, and last, `C_out[-1,:]`, points are the same.
    i_rem : ndarray (N,)
        N-by-1 boolean array used to indicate which of the corresponding
        vertices of `C` were removed during decimation.

    Notes
    -----
    Only one decimation criterion, `B_tol` or `P_tol`, may be provided (one
    must be left `None`). If neither are provided, the `B_tol` criterion is
    used with its default setting.

    ALGORITHM:
    1) For every vertex compute the boundary offset error.
    2) Rank all vertics according to the error score from step 1.
    3) Remove the vertex with the lowest error.
    4) Recompute and accumulate the errors for the two neighbours adjacent to
       the deleted vertex and go back to step 2.
    5) Repeat step 2 to 4 until no more vertices can be removed or the number
       of vertices has reached the desired number.

    This function is part of a translation to Python by Erik Husby
    (husby036@umn.edu) of MATLAB code originally written by Anton Semechko and
    posted on the MathWorks File Exchange, last updated 23 Jan 2012. [1]
    See translation note in "Compute the distance offset errors" section for
    one slight difference in functionality between source and translation.

    References
    ----------
    .. [1] https://www.mathworks.com/matlabcentral/fileexchange/34639-decimate-polygon

    """
    # Check the input args
    C = CheckInputArgs(C, B_tol, P_tol)

    N = C.shape[0]
    i_rem = np.zeros(N, dtype=np.bool)
    if N <= 4:
        C_out = C
        return C_out, i_rem

    # Tolerance parameter, perimeter and area of the input polygon
    Po, Emin = PolyPerim(C)
    if B_tol is None:
        if P_tol is not None:
            B_tol = P_tol
        else:
            B_tol = Emin / 2
    Ao = PolyArea(C)
    No = N - 1

    Nmin = 3
    if P_tol is not None:
        Nmin = int(round((N-1) * P_tol))
        if (N-1) == Nmin:
            C_out = np.empty((0, 2), dtype=np.float64)
            return C_out, i_rem
        if Nmin < 3:
            Nmin = 3

    # Remove the (repeating) end-point
    C = np.delete(C, -1, 0)
    N = N - 1


    # Compute the distance offset errors --------------------------------------
    D31 = np.roll(C, -1, axis=0) - np.roll(C, 1, axis=0)
    D21 = C - np.roll(C, 1, axis=0)
    dE_new2 = np.sum(np.square(D31), axis=1)  # length^2 of potential new edges

    # Find the closest point to the current vertex on the new edge
    t = np.sum(D21*D31, axis=1) / dE_new2
    # TRANSLATION NOTE:
    # The following clip of t behaves differently than in the source.
    # See comments posted by "Erik Husby, 9 Feb 2018" on MathWorks
    # File Exchange (reference [1] in docstring).
    t = np.clip(t, 0, 1)
    t = np.reshape(t, (t.size, 1))
    V = np.roll(C, 1, axis=0) + t*D31

    # Evaluate the distance^2
    Err_D2 = np.sum(np.square(V - C), axis=1)

    # Initialize distance error accumulation array
    DEAA = np.zeros(N, dtype=np.float64)


    # Begin decimation --------------------------------------------------------
    idx_ret = np.arange(N)  # keep track of retained vertices
    while True:

        # Find the vertices whose removal will satisfy the decimation criterion
        idx_i = Err_D2 < B_tol
        if P_tol is not None and not np.any(idx_i) and N > Nmin:
            B_tol = B_tol * np.sqrt(1.5)
            continue

        idx_i = np.where(idx_i)[0]
        if idx_i.size == 0 or N == Nmin:
            break
        N = N - 1

        # Vertex with the smallest net error
        i_min = np.argmin(Err_D2[idx_i])
        idx_i = idx_i[i_min]


        # Update the distance error accumulation array
        DEAA[idx_i] = DEAA[idx_i] + np.sqrt(Err_D2[idx_i])

        i1 = idx_i - 1
        if i1 < 0:
            i1 = N-1
        i3 = idx_i + 1
        if i3 >= N:
            i3 = 0

        DEAA[i1] = DEAA[idx_i]
        DEAA[i3] = DEAA[idx_i]

        # Recompute the errors for the vertices neighbouring the vertex marked
        # for deletion
        i1_1 = i1 - 1
        if i1_1 < 0:
            i1_1 = N-1
        i1_3 = i3

        i3_1 = i1
        i3_3 = i3 + 1
        if i3_3 >= N:
            i3_3 = 0

        err_D1 = RecomputeErrors(C[[i1_1, i1, i1_3], :])
        err_D3 = RecomputeErrors(C[[i3_1, i3, i3_3], :])

        # Update the errors
        Err_D2[i1] = np.square(np.sqrt(err_D1) + DEAA[i1])
        Err_D2[i3] = np.square(np.sqrt(err_D3) + DEAA[i3])

        # Remove the vertex
        C = np.delete(C, idx_i, 0)
        idx_ret = np.delete(idx_ret, idx_i)
        DEAA = np.delete(DEAA, idx_i)

        Err_D2 = np.delete(Err_D2, idx_i)

    C = np.vstack((C, C[0]))
    C_out = C

    i_rem[idx_ret] = True
    i_rem = ~i_rem
    i_rem[-1] = i_rem[0]

    # Perimeter and area of the simplified polygon
    P, _ = PolyPerim(C)
    A = PolyArea(C)

    # Performance summary
    stdout.write('\t\t# of verts\t\tperimeter\t\tarea\n')
    stdout.write('in\t\t%-5u\t\t\t%-.2f\t\t\t%-.2f\n' % (No, Po, Ao))
    stdout.write('out\t\t%-5u\t\t\t%-.2f\t\t\t%-.2f\n' % (N, P, A))
    stdout.write('-----------------------------------------------------\n')
    stdout.write('change\t%-5.2f%%\t\t\t%-5.2f%%\t\t\t%-5.2f%%\n\n' % ((N-No)/No*100, (P-Po)/Po*100, (A-Ao)/Ao*100))

    return C_out, i_rem


# ==========================================================================
def RecomputeErrors(V):
    """
    Recompute the distance offset error for a small subset of polygonal
    vertices.

    Parameters
    ----------
    V : ndarray (3,2)
        Array of triangle vertices, where V[1,:] is the vertex marked for
        removal.

    """
    # Compute the distance offset error.
    D31 = V[2, :] - V[0, :]
    D21 = V[1, :] - V[0, :]
    dE_new2 = np.sum(np.square(D31))  # length^2 of potential new edge

    # Find the closest point to the current vertex on the new edge.
    t = np.clip(np.sum(D21*D31) / dE_new2, 0, 1)
    p = V[0, :] + t*D31

    # Evaluate the distance^2.
    err_D2 = np.sum(np.square(p - V[1, :]))

    return err_D2


# ==========================================================================
def PolyPerim(C):
    # Polygon perimeter.

    dE = C[1:, :] - C[:-1, :]
    dE = np.sqrt(np.sum(np.square(dE), axis=1))
    P = np.sum(dE)
    Emin = np.min(dE)
    return P, Emin


# ==========================================================================
def PolyArea(C):
    # Polygon area.

    dx = C[1:, 0] - C[:-1, 0]
    dy = C[1:, 1] + C[:-1, 1]
    A = abs(np.sum(dx*dy) / 2)
    return A


# ==========================================================================
def CheckInputArgs(C, B_tol, P_tol):

    if type(C) != np.ndarray:
        raise InvalidArgumentError("`C` must be a N-by-2 array of polygon vertices")

    siz = C.shape
    if len(siz) != 2 or siz[1] != 2 or siz[0] < 3 or not np.issubdtype(C.dtype, np.number):
        raise InvalidArgumentError("`C` must be a N-by-2 array of polygon vertices")

    if C.dtype != np.float64:
        C = C.astype(np.float64)  # avoid accidental truncation

    if not np.array_equal(C[0], C[-1]):
        raise InvalidArgumentError("First and last points in `C` must be the same")

    if B_tol is not None and P_tol is not None:
        raise InvalidArgumentError("Only one decimation criterion (`B_tol` or `P_tol`) may be specified")

    if B_tol and (not isinstance(B_tol, Number) or B_tol <= 0):
        raise InvalidArgumentError("`B_tol` must be a number greater than zero")

    if P_tol and (not isinstance(B_tol, Number) or not (0 < P_tol < 1)):
        raise InvalidArgumentError("`P_tol` must be on the interval (0,1)")

    return C
