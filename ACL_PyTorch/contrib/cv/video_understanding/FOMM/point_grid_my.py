import torch
import torch.nn.functional as F


def bilinear_grid_sample(im, grid, align_corners=False, cuda=True):
    """Given an input and a flow-field grid, computes the output using input
    values and pixel locations from grid. Supported only bilinear interpolation
    method to sample the input pixels.

    Args:
        im (torch.Tensor): Input feature map, shape (N, C, H, W)
        grid (torch.Tensor): Point coordinates, shape (N, Hg, Wg, 2)
        align_corners {bool}: If set to True, the extrema (-1 and 1) are
            considered as referring to the center points of the input’s
            corner pixels. If set to False, they are instead considered as
            referring to the corner points of the input’s corner pixels,
            making the sampling more resolution agnostic.

    Returns:
        torch.Tensor: A tensor with sampled points, shape (N, C, Hg, Wg)
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.is_available():
    #     print("CUDA IS AVAILABLE!")
    # else:
    #     print("CUDA IS NOT AVAILABLE ... ")
    n, c, h, w = im.shape
    gn, gh, gw, _ = grid.shape
    assert n == gn

    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]

    if align_corners:
        x = ((x + 1) / 2) * (w - 1)
        y = ((y + 1) / 2) * (h - 1)
    else:
        x = ((x + 1) * w - 1) / 2
        y = ((y + 1) * h - 1) / 2

    x = x.view(n, -1)
    y = y.view(n, -1)

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1

    wa = ((x1 - x) * (y1 - y)).unsqueeze(1)
    wb = ((x1 - x) * (y - y0)).unsqueeze(1)
    wc = ((x - x0) * (y1 - y)).unsqueeze(1)
    wd = ((x - x0) * (y - y0)).unsqueeze(1)

    # Apply default for grid_sample function zero padding
    im_padded = F.pad(im, pad=[1, 1, 1, 1], mode='constant', value=0)
    padded_h = h + 2
    padded_w = w + 2
    # save points positions after padding

    x0, x1, y0, y1 = (x0 + 1).to(device), (x1 + 1).to(device), (y0 + 1).to(device), (y1 + 1).to(device)


    zero = torch.tensor(0).to(device)
    w_1 = torch.tensor(padded_w - 1).to(device)
    h_1 = torch.tensor(padded_h - 1).to(device)
    # Clip coordinates to padded image size
    x0 = torch.where(x0 < 0, zero, x0)
    x0 = torch.where(x0 > padded_w - 1, w_1, x0)
    x1 = torch.where(x1 < 0, zero, x1)
    x1 = torch.where(x1 > padded_w - 1, w_1, x1)
    y0 = torch.where(y0 < 0, zero, y0)
    y0 = torch.where(y0 > padded_h - 1, h_1, y0)
    y1 = torch.where(y1 < 0, zero, y1)
    y1 = torch.where(y1 > padded_h - 1, h_1, y1)

    im_padded = im_padded.view(n, c, -1)

    x0_y0 = (x0 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x0_y1 = (x0 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y0 = (x1 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y1 = (x1 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)

    Ia = torch.gather(im_padded, 2, x0_y0)
    Ib = torch.gather(im_padded, 2, x0_y1)
    Ic = torch.gather(im_padded, 2, x1_y0)
    Id = torch.gather(im_padded, 2, x1_y1)

    return (Ia * wa + Ib * wb + Ic * wc + Id * wd).reshape(n, c, gh, gw)