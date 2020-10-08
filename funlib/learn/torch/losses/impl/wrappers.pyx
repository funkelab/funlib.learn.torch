import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t


cdef extern from "um_loss.h":
    double c_um_loss_gradient(
        size_t numNodes,
        const double* mst,
        const int64_t* gtSeg,
        double alpha,
        double* gradients,
        double* ratioPos,
        double* ratioNeg,
        double& totalNumPairsPos,
        double& totalNumPairsNeg);
    void c_prune_mst(
        size_t numNodes,
        size_t numComponents,
        const double* mst,
        const int64_t* labels,
        const int64_t* components,
        double* filtered_mst);

def um_loss(
    np.ndarray[double, ndim=2] mst,
    np.ndarray[int64_t, ndim=1] gt_seg,
    double alpha):

    cdef size_t num_points = gt_seg.shape[0]
    cdef size_t num_edges = mst.shape[0]

    assert num_points == num_edges + 1, (
        "Number of edges %d in MST is unequal number of points %d in "
        "segmentation minus one." % (num_edges, num_points))

    assert mst.shape[1] == 3, "mst not given as rows of [u, v, dist]"

    # the C++ part assumes contiguous memory, make sure we have it (and do 
    # nothing, if we do)
    if not mst.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous mst arrray (avoid this by "
              "passing C_CONTIGUOUS arrays)")
        mst = np.ascontiguousarray(mst)
    if not gt_seg.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous gt_seg arrray (avoid this by "
              "passing C_CONTIGUOUS arrays)")
        gt_seg = np.ascontiguousarray(gt_seg)

    # prepare output arrays
    cdef np.ndarray[double, ndim=1] gradients = np.zeros(
            (num_edges,),
            dtype=np.float64)
    cdef np.ndarray[double, ndim=1] ratio_neg = np.zeros(
            (num_edges,),
            dtype=np.float64)
    cdef np.ndarray[double, ndim=1] ratio_pos = np.zeros(
            (num_edges,),
            dtype=np.float64)

    cdef double num_pairs_pos;
    cdef double num_pairs_neg;

    cdef double loss = c_um_loss_gradient(
        num_points,
        &mst[0, 0],
        &gt_seg[0],
        alpha,
        &gradients[0],
        &ratio_pos[0],
        &ratio_neg[0],
        num_pairs_pos,
        num_pairs_neg)

    return (
        loss,
        gradients,
        ratio_pos,
        ratio_neg,
        num_pairs_pos,
        num_pairs_neg)


def prune_mst(
        np.ndarray[double, ndim=2] mst,
        np.ndarray[int64_t, ndim=1] labels,
        np.ndarray[int64_t, ndim=1] components):
    '''Filter edges of an MST such that only edges connecting differently
    labeled components remain. The edges will form a spanning tree between the
    components.'''

    cdef size_t num_edges = mst.shape[0]
    cdef size_t num_points = labels.shape[0]
    cdef size_t num_components = components.shape[0]

    assert num_points == num_edges + 1, (
        "Number of edges %d in MST is unequal number of points %d in "
        "segmentation minus one." % (num_edges, num_points))

    assert mst.shape[1] == 3, "mst not given as rows of [u, v, dist]"

    # the C++ part assumes contiguous memory, make sure we have it (and do 
    # nothing, if we do)
    if not mst.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous mst arrray (avoid this by "
              "passing C_CONTIGUOUS arrays)")
        mst = np.ascontiguousarray(mst)
    if not labels.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous labels arrray (avoid this by "
              "passing C_CONTIGUOUS arrays)")
        labels = np.ascontiguousarray(labels)
    if not components.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous components arrray (avoid this by "
              "passing C_CONTIGUOUS arrays)")
        components = np.ascontiguousarray(components)

    # prepare output arrays
    cdef np.ndarray[double, ndim=2] filtered_mst = np.zeros(
            (num_components - 1, 3),
            dtype=np.float64)

    c_prune_mst(
        num_points,
        num_components,
        &mst[0, 0],
        &labels[0],
        &components[0],
        &filtered_mst[0, 0])

    return filtered_mst
