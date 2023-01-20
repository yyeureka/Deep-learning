import torch
import torch.nn as nn

class ContentLoss(nn.Module):
    def forward(self, content_weight, content_current, content_original):
        """
            Compute the content loss for style transfer.

            Inputs:
            - content_weight: Scalar giving the weighting for the content loss.
            - content_current: features of the current image; this is a PyTorch Tensor of shape
              (1, C_l, H_l, W_l).
            - content_target: features of the content image, Tensor with shape (1, C_l, H_l, W_l).

            Returns:
            - scalar content loss
            """

        ##############################################################################
        # TODO: Implement content loss function                                      #
        # Use torch tensor math function or else you will run into issues later      #
        # where the computational graph is broken and appropriate gradients cannot   #
        # be computed.                                                               #
        ##############################################################################

        diff = content_current - content_original
        loss = content_weight * torch.sum(torch.pow(diff, 2))

        return loss
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################

