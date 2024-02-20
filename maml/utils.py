import torch

from collections import OrderedDict
from torchmeta.modules import MetaModule
from scipy import ndimage
def _get_coutour_sample(y_true):
    """
    y_true: Bx2xHxW

    """
    # print('y true shape:', y_true.shape)
    # positive_mask = torch.unsqueeze(y_true[..., 1], dim=3)
    positive_mask= y_true.clone()         ### GT
    #positive_mask = (y_true1).unsqueeze(1)

    metrix_label_group = torch.unsqueeze(torch.tensor([1, 0, 1, 0]), dim=1)
    coutour_group = torch.zeros_like(positive_mask)

    # print('positive mask shape:', positive_mask.shape)
    #
    for i in range(positive_mask.shape[0]):

        slice_i = positive_mask[i, 0]
        slice_i_np = slice_i.cpu().numpy()
        # print('slice i:', slice_i.shape)
        # print()

        if metrix_label_group[i] == 1:
            # generate coutour mask
            # print('slice i:', slice_i.dtype)
            erosion = torch.tensor(ndimage.binary_erosion(slice_i_np, iterations=1).astype(slice_i_np.dtype))
            erosion = erosion.to('cuda')

            # print('slice_i[..., 0], erosion:', (slice_i[..., 0]).dtype, erosion.dtype)
            slice_i = slice_i
            sample = slice_i - erosion

            # print('class 1 shape:', sample.shape)
            # print()3
        #  0, 0_1, 0_2
        elif metrix_label_group[i] == 0:
            # generate background mask
            dilation = torch.tensor(ndimage.binary_dilation(slice_i_np, iterations=5).astype(slice_i_np.dtype))
            dilation = dilation.to('cuda')
            sample = dilation - slice_i

            # change to neg sameple
            # easy neg  --> psuedo labels
            # hard neg--> outputs from other networks, what we use are Unet, d2e2, segnet


            # print('class 0 shape: ', sample.shape)
            # print()
        coutour_group[i] = sample.unsqueeze(0)
    # print('metrix_label_group:', metrix_label_group)
    return coutour_group, metrix_label_group
def _get_compactness_cost(y_pred, y_true):
    """
    y_pred: BxCxHxW
    """

    y_pred = y_pred[:,0,:,:]
    # y_true = y_pred[:,0,:,:]

    x = y_pred[:, 1:, :] - y_pred[:, :-1, :]  # horizontal and vertical directions
    y = y_pred[:, :, 1:] - y_pred[:, :, :-1]

    delta_x = x[:, :, 1:] ** 2
    delta_y = y[:, 1:, :] ** 2

    delta_u = torch.abs(delta_x + delta_y)

    epsilon = 0.00000001  # A parameter to avoid division by zero
    w = 0.01
    length = w * torch.sum(torch.sqrt(delta_u + epsilon), [1, 2])

    area = torch.sum(y_pred, [1, 2])

    compactness_loss = torch.sum(length ** 2 / (area * 4 * 3.1415926))

    return compactness_loss, torch.sum(length), torch.sum(area), delta_u

def pairwise_distance_torch(embeddings, device):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
      embeddings: 2-D Tensor of size [number of data, feature dimension].
    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """

    # pairwise distance matrix with precise embeddings
    precise_embeddings = embeddings.to('cuda')
    # print('precise_embeddings:', precise_embeddings.shape)
    precise_embeddings = precise_embeddings.reshape((precise_embeddings.shape[0], 1))
    # print('new precise_embeddings:', precise_embeddings.shape)

    c1 = torch.pow(precise_embeddings, 2).sum(axis=-1)
    c2 = torch.pow(precise_embeddings.transpose(0, 1), 2).sum(axis=0)
    c3 = precise_embeddings @ precise_embeddings.transpose(0, 1)

    c1 = c1.reshape((c1.shape[0], 1))
    c2 = c2.reshape((1, c2.shape[0]))
    c12 = c1 + c2
    pairwise_distances_squared = c12 - 2.0 * c3

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.max(pairwise_distances_squared, torch.tensor([0.]).to(device))
    # Get the mask where the zero distances are at.
    error_mask = pairwise_distances_squared.clone()
    error_mask[error_mask > 0.0] = 1.
    error_mask[error_mask <= 0.0] = 0.

    pairwise_distances = torch.mul(pairwise_distances_squared, error_mask)

    # Explicitly set diagonals to zero.
    mask_offdiagonals = torch.ones((pairwise_distances.shape[0], pairwise_distances.shape[1])) - torch.diag(torch.ones(pairwise_distances.shape[0]))
    pairwise_distances = torch.mul(pairwise_distances.to('cuda'), mask_offdiagonals.to('cuda'))
    return pairwise_distances

def TripletSemiHardLoss(y_true, y_pred, device, margin=1.0):
    """Computes the triplet loss_functions with semi-hard negative mining.
       The loss_functions encourages the positive distances (between a pair of embeddings
       with the same labels) to be smaller than the minimum negative distance
       among which are at least greater than the positive distance plus the
       margin constant (called semi-hard negative) in the mini-batch.
       If no such negative exists, uses the largest negative distance instead.
       See: https://arxiv.org/abs/1503.03832.
       We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
       [batch_size] of multi-class integer labels. And embeddings `y_pred` must be
       2-D float `Tensor` of l2 normalized embedding vectors.
       Args:
         margin: Float, margin term in the loss_functions definition. Default value is 1.0.
         name: Optional name for the op.
       """

    labels, embeddings = y_true, y_pred

    # Reshape label tensor to [batch_size, 1].
    lshape = labels.shape
    labels = torch.reshape(labels, [lshape[0], 1])

    pdist_matrix = pairwise_distance_torch(embeddings, device)

    # Build pairwise binary adjacency matrix.
    adjacency = torch.eq(labels, labels.transpose(0, 1))
    # Invert so we can select negatives only.
    adjacency_not = adjacency.logical_not()
    # print('adjacency_not:', adjacency_not.shape)  # 4, 4

    batch_size = labels.shape[0]

    # print('batch size:', batch_size)    # 4
    # print('pdist_matrix shape:', pdist_matrix.shape)   # embedding generate (2,2)

    # Compute the mask.
    pdist_matrix_tile = pdist_matrix.repeat(batch_size, 1)
    # print('pdist_matrix_tile:', pdist_matrix_tile.shape)  #  8, 2
    adjacency_not_tile = adjacency_not.repeat(batch_size, 1)
    # print('adjacency_not_tile:', adjacency_not_tile.shape)  #  16, 4

    transpose_reshape = pdist_matrix.transpose(0, 1).reshape(-1, 1)
    # print('transpose_reshape:', transpose_reshape.shape)  # 4, 1
    greater = pdist_matrix_tile > transpose_reshape

    # print('greater:', greater)
    # print('adjacency_not_tile:', adjacency_not_tile)
    adjacency_not_tile = adjacency_not_tile.to('cuda')
    mask = adjacency_not_tile & greater

    # final mask
    mask_step = mask.to('cuda')
    mask_step = mask_step.sum(axis=1)
    mask_step = mask_step > 0.0
    mask_final = mask_step.reshape(batch_size, batch_size)
    mask_final = mask_final.transpose(0, 1)

    adjacency_not = adjacency_not.to('cuda')
    mask = mask.to('cuda')

    # negatives_outside: smallest D_an where D_an > D_ap.
    axis_maximums = torch.max(pdist_matrix_tile, dim=1, keepdim=True)
    masked_minimums = torch.min(torch.mul(pdist_matrix_tile - axis_maximums[0], mask), dim=1, keepdim=True)[0] + axis_maximums[0]
    negatives_outside = masked_minimums.reshape([batch_size, batch_size])
    negatives_outside = negatives_outside.transpose(0, 1)

    # negatives_inside: largest D_an.
    axis_minimums = torch.min(pdist_matrix, dim=1, keepdim=True)
    masked_maximums = torch.max(torch.mul(pdist_matrix - axis_minimums[0], adjacency_not), dim=1, keepdim=True)[0] + axis_minimums[0]
    negatives_inside = masked_maximums.repeat(1, batch_size)

    semi_hard_negatives = torch.where(mask_final, negatives_outside, negatives_inside)

    loss_mat = margin + pdist_matrix - semi_hard_negatives

    mask_positives = adjacency.to('cuda').float() - torch.diag(torch.ones(batch_size)).to('cuda')
    num_positives = mask_positives.sum()

    triplet_loss = (torch.max(torch.mul(loss_mat, mask_positives), torch.tensor([0.]).to('cuda'))).sum() / num_positives
    triplet_loss = triplet_loss.to('cuda')

    return triplet_loss

def _get_boundary_cost(y_pred, y_true):
    """
    y_pred: B* c * H * W
    """
    """
    lenth term
    """
    y_pred = y_pred[:,0,:,:]
    y_true = y_pred[:,0,:,:]

    x = y_pred[:, 1:, :] - y_pred[:, :-1, :]  # horizontal and vertical directions
    y = y_pred[:, :, 1:] - y_pred[:, :, :-1]

    delta_x = x[:, :, 1:] ** 2
    delta_y = y[:, 1:, :] ** 2

    delta_u = torch.abs(delta_x + delta_y)

    epsilon = 0.00000001  # A parameter to avoid division by zero
    w = 0.01
    length = w * torch.sum(torch.sqrt(delta_u + epsilon), [1, 2])

    area = torch.sum(y_pred, [1, 2])

    compactness_loss = torch.sum(length ** 2 / (area * 4 * 3.1415926))

    return compactness_loss, torch.sum(length), torch.sum(area), delta_u

def compute_accuracy(logits, targets):
    """Compute the accuracy"""
    with torch.no_grad():
        _, predictions = torch.max(logits, dim=1)
        accuracy = torch.mean(predictions.eq(targets).float())
    return accuracy.item()

def compute_IoU(logits, targets):
    with torch.no_grad():
        predictions = (logits > 0.5).float()
        targets = targets.to(predictions.device)
        intersection = torch.sum(predictions * targets)
        union = torch.sum(predictions) + torch.sum(targets) - intersection
        # print('union:', union)
        iou = intersection / (union + 1e-7)
        # print('iou:', iou.item())
    return iou.item()

def dice(logits, targets):
    with torch.no_grad():
        predictions = (logits > 0.5).float()
        targets = targets.to(predictions.device)
        intersection = torch.sum(predictions * targets)
        union = torch.sum(predictions) + torch.sum(targets)
        dice = 2 * intersection / (union + 1e-7)
    return dice.item()

def dice_loss(logits, targets):
    loss = 1 - dice(logits, targets)
    return loss

def compute_loss(logits, targets):
    """Compute the cross entropy loss"""
    # loss = torch.nn.functional.cross_entropy(logits, targets)
    """ Compute the dice loss"""
    loss = 1 - dice_loss(logits, targets)
    """ Compute the IoU loss"""
    # loss = 1 - compute_IoU(logits, targets)
    return loss



def tensors_to_device(tensors, device=torch.device('cpu')):
    """Place a collection of tensors in a specific device"""
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, (list, tuple)):
        return type(tensors)(tensors_to_device(tensor, device=device)
            for tensor in tensors)
    elif isinstance(tensors, (dict, OrderedDict)):
        return type(tensors)([(name, tensors_to_device(tensor, device=device))
            for (name, tensor) in tensors.items()])
    else:
        raise NotImplementedError()

class ToTensor1D(object):
    """Convert a `numpy.ndarray` to tensor. Unlike `ToTensor` from torchvision,
    this converts numpy arrays regardless of the number of dimensions.

    Converts automatically the array to `float32`.
    """
    def __call__(self, array):
        return torch.from_numpy(array.astype('float32'))

    def __repr__(self):
        return self.__class__.__name__ + '()'