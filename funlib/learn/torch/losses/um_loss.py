from .impl import um_loss, prune_mst

import numpy as np
import mlpack as mlp
import torch

import logging
from typing import Optional, Union, List

logger = logging.getLogger(__name__)


def compute_emst(embedding: np.ndarray) -> np.ndarray:

    if embedding.shape[0] <= 1:
        logger.warn("can't compute EMST for %d points", embedding.shape[0])
        return np.zeros((0, 3), dtype=np.float64)

    return mlp.emst(embedding)["output"]


def get_unconstrained_emst(embedding: np.ndarray) -> np.ndarray:

    emst = compute_emst(embedding.astype(np.float64))

    d_min = np.min(emst[:, 2])
    d_max = np.max(emst[:, 2])
    logger.info("min/max ultrametric: %f/%f", d_min, d_max)

    return emst


def get_constrained_emst(embedding: np.ndarray, labels: np.ndarray) -> np.ndarray:

    if embedding.shape[0] <= 1:
        logger.warn("can't compute EMST for %d points", embedding.shape[0])
        return np.zeros((0, 3), dtype=np.float64)

    embedding = embedding.astype(np.float64)
    components = np.unique(labels)
    if len(components) <= 1:
        logger.warn("can't compute constrained EMST for 1 or fewer components")
        return compute_emst(embedding)
    num_points = embedding.shape[0]
    indices = np.arange(num_points)

    component_emsts = []

    # grow inside each component first
    for component in components:

        mask = labels == component
        masked_embedding = embedding[mask]
        masked_indices = indices[mask]

        component_emst = compute_emst(masked_embedding)

        # fix indices
        component_indices = component_emst[:, 0:2].astype(np.int32)
        component_emst[:, 0:2] = masked_indices[component_indices]

        component_emsts.append(component_emst)

    # grow on complete embedding
    complete_emst = compute_emst(embedding)

    # prune emst to only connect components
    pruned_emst = prune_mst(complete_emst, labels, components)

    emst = np.concatenate(component_emsts + [pruned_emst])

    assert emst.shape[0] == num_points - 1

    d_min = np.min(emst[:, 2])
    d_max = np.max(emst[:, 2])
    logger.info("min/max ultrametric: %f/%f", d_min, d_max)

    return emst


def get_emst(
    embedding: torch.Tensor, constrain_to: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute the EMST for the given embedding.

    Args:

        embedding (Tensor, shape ``(n, k)``):

            A k-dimensional feature embedding of n points.

        constrain_to (Tensor, shape ``(n, k)``, optional):

            Labels for the points in ``embedding``. If given, the EMST will be
            constrained to first grow inside each component, then connect the
            components to each other.

    Returns:

        A tensor ``(n-1, 3)`` representing the EMST as rows of ``(u, v,
        distance)``, where ``u`` and ``v`` are indices of points in
        ``embedding``.
    """
    embedding = embedding.numpy()
    if constrain_to is not None:
        constrain_to = constrain_to.numpy()
        emst = get_constrained_emst(embedding, constrain_to)
    else:
        emst = compute_emst(embedding)
    return torch.from_numpy(emst)


def get_um_loss(
    mst: torch.Tensor, dist: torch.Tensor, gt_seg: torch.Tensor, alpha: torch.Tensor
):
    """Compute the ultra-metric loss given an MST and segmentation.

    Args:

        mst (Tensor, shape ``(n-1, 3)``): u, v indices and distance of edges of
            the MST spanning n nodes.

        dist (Tensor, shape ``(n-1)``): The distances of the edges. This
            argument will be ignored, it is used only to communicate to
            tensorflow that there is a dependency on distances. The distances
            actually used are the ones in parameter ``mst``.

        gt_seg (Tensor, arbitrary shape): The label of each node. Will be
            flattened. The indices in mst should be valid indices into this
            array.

        alpha (Tensor, single float): The margin value of the quadrupel loss.

    Returns:

        A tuple of tensors::

            (loss, ratio_pos, ratio_neg, num_pairs_pos, num_pairs_neg)

        ``ratio_pos`` and ``ratio_neg`` are tensors of shape ``(n-1,)``,
        corresponding to the edges in the MST. ``ratio_pos`` and ``ratio_neg``
        are the ratio of positive and negative pairs that share an edge, of the
        total number of positive and negative pairs.
    """

    if mst.shape[0] == 0:
        return (
            torch.tensor(0, dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
            torch.tensor(0, dtype=torch.int32),
            torch.tensor(0, dtype=torch.int32),
        )

    # We don't use 'dist' here, it is already contained in the mst. It is
    # passed here just so that tensorflow knows there is dependecy to the
    # ouput.
    (loss, _, ratio_pos, ratio_neg, num_pairs_pos, num_pairs_neg) = um_loss(
        mst.numpy(), gt_seg.numpy(), float(alpha)
    )

    return (
        torch.tensor(loss, dtype=torch.float32),
        torch.from_numpy(ratio_pos).float(),
        torch.from_numpy(ratio_neg).float(),
        torch.tensor(num_pairs_pos, dtype=torch.int32),
        torch.tensor(num_pairs_neg, dtype=torch.int32),
    )


def ultrametric_loss(
    embedding: torch.Tensor,
    gt_seg: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    alpha: float = 0.1,
    add_coordinates: bool = True,
    coordinate_scale: Union[float, List[float]] = 1.0,
    balance: bool = True,
    quadrupel_loss: bool = False,
    constrained_emst: bool = False,
    name=None,
):
    """Returns a tensorflow op to compute the ultra-metric loss on pairs of
    embedding points::

        L = 1/P * sum_p d(p)^2 + 1/N * sum_n max(0, alpha - d(n))^2

    where ``p`` and ``n`` are pairs of points with same and different labels in
    ``gt_seg``, respectively, and ``d(.)`` the ultrametric distance between the
    points. ``P`` and ``N`` are the total number of positive and negative
    pairs.

    There are two special labels for ``gt_seg``: 0 for background, and -1 for
    foreground with an unknown label (e.g., for overlapping objects, where it
    doesn't matter to which object the point gets assigned).

    A pair is positive if both points have the same label >= 1, and negative if
    both points have a different label >= 1.

    In addition to that, pairs that have one point with a label of 0
    (background) and one point with a foreground label (-1 or >=1) are
    considered negative pairs. This is to express that we know that labelled
    areas are different from background.

    All other pairs are ignored, i.e., (-1, -1), (-1, >=1), and (0, 0). For
    those pairs, we don't know whether they are part of the same object.

    Args:

        embedding (Tensor, shape ``(k, d, h, w)``):

            A k-dimensional feature embedding of points in 3D.

        gt_seg (Tensor, shape ``(d, h, w)``):

            The ground-truth labels of the points.

        mask (optional, Tensor, shape ``(d, h, w)``):

            If given, consider only points that are not zero in the mask.

        alpha (optional, float):

            The margin term of the quadrupel loss.

        add_coordinates (optional, bool):

            If ``True``, add the ``(z, y, x)`` coordinates of the points to the
            embedding.

        coordinate_scale (optional, ``float`` or ``tuple`` of ``float``):

            How to scale the coordinates, if used to augment the embedding.

        balance (optional, ``bool``):

            If ``true`` (the default), the total loss is the sum of positive
            pair losses and negative pair losses; each divided by the number of
            positive and negative pairs, respectively. This puts equal emphasis
            on positive and negative pairs, independent of the number of
            positive and negative pairs.

            If ``false``, the total loss is the sum of positive pair losses and
            negative pair losses, divided by the total number of pairs. This
            puts more emphasis on the set of pairs (positive or negative) that
            occur more frequently::

                L = 1/(P + N) * (sum_p d(p)^2 + sum_n max(0, alpha - d(n))^2)

        quadrupel_loss (optional, ``bool``):

            If ``true``, compute the loss on all quadrupels of points, instead
            of pairs of points::

                L = 1/(P*N) * sum_p sum_n max(0, d(p) - d(n) + alpha)^2

        constrained_emst (optional, ``bool``):

            If set to ``true``, compute the EMST such that it first grows
            inside each component of the same label in ``gt_seg``, then between
            the components. This results in a loss that is an upper bound of L.

        name (optional, ``string``):

            An optional name for the operator.

    Returns:

        A tuple ``(loss, emst, edges_u, edges_v, dist)``, where ``loss`` is a
        scalar, ``emst`` a tensor holding the MST edges as pairs of nodes,
        ``edges_u`` and ``edges_v`` the respective embeddings of each edges,
        and ``dist`` the length of the edges.
    """

    # We get the embedding as a tensor of shape (k, d, h, w).
    k, depth, height, width = embedding.shape

    # 1. Augmented by spatial coordinates, if requested.

    if add_coordinates:

        try:
            scale = tuple(coordinate_scale)
        except TypeError:
            scale = (coordinate_scale,) * 3

        coordinates = torch.meshgrid(
            np.arange(0, depth * scale[0], scale[0]),
            np.arange(0, height * scale[1], scale[1]),
            np.arange(0, width * scale[2], scale[2]),
            indexing="ij",
        )
        for i in range(len(coordinates)):
            coordinates[i] = coordinates[i].float32()
        embedding = torch.cat([embedding, coordinates], 0)

        max_scale = float(scale.max())
        min_scale = float(scale.min())
        min_d = min_scale
        max_d = np.sqrt(max_scale ** 2 + k)

        if (max_d - min_d) < alpha:
            logger.warn(
                "Your alpha is too big: min and max ultrametric between any "
                "pair of points is %f and %f (this assumes your embedding is "
                "in [0, 1], if it is not, you might ignore this warning)",
                min_d,
                max_d,
            )

    # 2. Transpose into tensor (d*h*w, k), i.e., one embedding vector per node.

    embedding = embedding.permute(1, 2, 3, 0)
    embedding = embedding.view(depth * width * height, -1)
    gt_seg = gt_seg.view(depth * width * height)

    if mask is not None:
        # TODO: find alternative of masked_select that does not require excessive reshaping
        mask = mask.view(depth * width * height, 1)
        embedding = torch.masked_select(embedding, mask).view(-1, embedding.shape[-1])
        gt_seg = torch.masked_select(gt_seg.view(-1, 1), mask).view(-1)

    # 3. Get the EMST on the embedding vectors.

    if constrained_emst:
        emst = get_emst(embedding, constrain_to=gt_seg)
    else:
        emst = get_emst(embedding)

    # 4. Compute the lengths of EMST edges

    edges_u = embedding[emst[:, 0].to(torch.int64)]
    edges_v = embedding[emst[:, 1].to(torch.int64)]
    dist_squared = ((edges_u - edges_v) ** 2).sum(1)
    dist = dist_squared ** 0.5

    # 5. Compute the UM loss

    if quadrupel_loss:
        loss, *_ = get_um_loss(emst, dist, gt_seg, alpha)
        # TODO: Assign gradients and test their assignment
    else:

        # we need the um_loss just to get the ratio_pos, ratio_neg, and the
        # total number of positive and negative pairs
        _, ratio_pos, ratio_neg, num_pairs_pos, num_pairs_neg = get_um_loss(
            emst, dist, gt_seg, alpha
        )

        loss_pos = dist_squared * ratio_pos
        loss_neg = (torch.max(torch.Tensor([0.0]), alpha - dist)) ** 2 * ratio_neg

        if balance:

            # the ratios returned by get_um_loss are already class balanced,
            # there is nothing more to do than to add the losses up
            loss = loss_pos.sum() + loss_neg.sum()

        else:

            # denormalize the ratios, add them up, and divide by the total
            # number of pairs
            sum_pos = loss_pos.sum() * num_pairs_pos
            sum_neg = loss_neg.sum() * num_pairs_neg
            num_pairs = num_pairs_pos + num_pairs_neg

            loss = (sum_pos + sum_neg) / num_pairs

    return (float(loss), emst, edges_u, edges_v, dist.numpy())
