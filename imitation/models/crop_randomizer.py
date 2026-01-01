import torch
import numpy as np

# crop randomizer from robomimic: https://github.com/ARISE-Initiative/robomimic/blob/0ca7ce74cf8f20be32029657ef9320db033d93e9/robomimic/models/obs_core.py#L489

class CropRandomizer(torch.nn.Module):
    """
    Randomly sample crops at input, and then average across crop features at output.
    """
    def __init__(
        self,
        input_shape,
        crop_height,
        crop_width,
        num_crops=1,
        pos_enc=False,
    ):
        """
        Args:
            input_shape (tuple, list): shape of input (not including batch dimension)
            crop_height (int): crop height
            crop_width (int): crop width
            num_crops (int): number of random crops to take
            pos_enc (bool): if True, add 2 channels to the output to encode the spatial
                location of the cropped pixels in the source image
        """
        super().__init__()
        assert len(input_shape) == 3 # (C, H, W)
        assert crop_height < input_shape[1]
        assert crop_width < input_shape[2]

        self.input_shape = input_shape
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.num_crops = num_crops
        self.pos_enc = pos_enc

    def output_shape_in(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. Corresponds to
        the @forward_in operation, where raw inputs (usually observation modalities)
        are passed in.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        # outputs are shape (C, CH, CW), or maybe C + 2 if using position encoding, because
        # the number of crops are reshaped into the batch dimension, increasing the batch
        # size from B to B * N
        out_c = self.input_shape[0] + 2 if self.pos_enc else self.input_shape[0]
        return [out_c, self.crop_height, self.crop_width]

    def output_shape_out(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. Corresponds to
        the @forward_out operation, where processed inputs (usually encoded observation
        modalities) are passed in.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        # since the forward_out operation splits [B * N, ...] -> [B, N, ...]
        # and then pools to result in [B, ...], only the batch dimension changes,
        # and so the other dimensions retain their shape.
        return list(input_shape)

    def forward_in(self, inputs):
        """
        Samples N random crops for each input in the batch, and then reshapes
        inputs to [B * N, ...].
        """
        assert len(inputs.shape) >= 3 # must have at least (C, H, W) dimensions
        out, _ = sample_random_image_crops(
            images=inputs,
            crop_height=self.crop_height,
            crop_width=self.crop_width,
            num_crops=self.num_crops,
            pos_enc=self.pos_enc,
        )
        # [B, N, ...] -> [B * N, ...]
        return out.view(-1, *out.shape[2:])

    # def _forward_in_eval(self, inputs):
    #     """
    #     Do center crops during eval
    #     """
    #     assert len(inputs.shape) >= 3 # must have at least (C, H, W) dimensions
    #     inputs = inputs.permute(*range(inputs.dim()-3), inputs.dim()-2, inputs.dim()-1, inputs.dim()-3)
    #     out = ObsUtils.center_crop(inputs, self.crop_height, self.crop_width)
    #     out = out.permute(*range(out.dim()-3), out.dim()-1, out.dim()-3, out.dim()-2)
    #     return out

    def forward_out(self, inputs):
        """
        Splits the outputs from shape [B * N, ...] -> [B, N, ...] and then average across N
        to result in shape [B, ...] to make sure the network output is consistent with
        what would have happened if there were no randomization.
        """
        batch_size = inputs.shape[0] // self.num_crops
        out = inputs.view(batch_size, self.num_crops, *inputs.shape[1:])
        return out.mean(dim=1)

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = header + "(input_shape={}, crop_size=[{}, {}], num_crops={})".format(
            self.input_shape, self.crop_height, self.crop_width, self.num_crops)
        return msg

def sample_random_image_crops(images, crop_height, crop_width, num_crops, pos_enc=False):
    """
    For each image, randomly sample @num_crops crops of size (@crop_height, @crop_width), from
    @images.

    Args:
        images (torch.Tensor): batch of images of shape [..., C, H, W]
        crop_height (int): height of crop to take
        crop_width (int): width of crop to take
        num_crops (n): number of crops to sample
        pos_enc (bool): if True, also add 2 channels to the outputs that gives a spatial 
            encoding of the original source pixel locations.

    Returns:
        crops (torch.Tensor): crops of shape (..., @num_crops, C, @crop_height, @crop_width) 
            if @pos_enc is False, otherwise (..., @num_crops, C + 2, @crop_height, @crop_width)
        crop_inds (torch.Tensor): sampled crop indices of shape (..., N, 2)
    """
    device = images.device
    batch_shape = images.shape[:-3]
    image_c, image_h, image_w = images.shape[-3:]
    
    # Add positional encoding if requested
    source_im = images
    if pos_enc:
        h, w = image_h, image_w
        pos_y, pos_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        pos_y = pos_y.float() / float(h)
        pos_x = pos_x.float() / float(w)
        position_enc = torch.stack((pos_y, pos_x))  # [2, H, W]
        
        # Expand to match batch dimensions
        position_enc = position_enc.view((1,) * len(batch_shape) + position_enc.shape)
        position_enc = position_enc.expand(*batch_shape, -1, -1, -1)
        source_im = torch.cat((source_im, position_enc), dim=-3)

    # Sample crop locations
    max_sample_h = image_h - crop_height
    max_sample_w = image_w - crop_width
    
    crop_inds_h = torch.randint(0, max_sample_h, (*batch_shape, num_crops), device=device)
    crop_inds_w = torch.randint(0, max_sample_w, (*batch_shape, num_crops), device=device)
    crop_inds = torch.stack((crop_inds_h, crop_inds_w), dim=-1)  # [..., N, 2]

    # Create crops using torch operations
    batch_size = torch.prod(torch.tensor(batch_shape)) if batch_shape else 1
    flat_images = source_im.view(batch_size, *source_im.shape[-3:])  # [B, C, H, W]
    flat_crop_inds = crop_inds.view(batch_size, num_crops, 2)  # [B, N, 2]
    
    crops = []
    for b in range(batch_size):
        img = flat_images[b]  # [C, H, W]
        img_crops = []
        for n in range(num_crops):
            h_start, w_start = flat_crop_inds[b, n]
            crop = img[:, h_start:h_start+crop_height, w_start:w_start+crop_width]
            img_crops.append(crop)
        crops.append(torch.stack(img_crops))  # [N, C, CH, CW]
    
    crops = torch.stack(crops)  # [B, N, C, CH, CW]
    crops = crops.view(*batch_shape, num_crops, *crops.shape[-3:])
    
    return crops, crop_inds