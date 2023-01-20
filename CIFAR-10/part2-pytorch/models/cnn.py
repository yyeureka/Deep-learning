"""
Vanilla CNN model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import torch
import torch.nn as nn


class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        #############################################################################
        # TODO: Initialize the Vanilla CNN                                          #
        #       Conv: 7x7 kernel, stride 1 and padding 0                            #
        #       Max Pooling: 2x2 kernel, stride 2                                   #
        #############################################################################

        self.conv = nn.Conv2d(3, 32, kernel_size=(7, 7), stride=(1, 1), padding=0)
        self.activation = nn.ReLU(inplace=True)
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.num_classes = 10
        self.fc = nn.Linear(5408, self.num_classes)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################

        outs = self.conv(x)
        outs = self.activation(outs)
        outs = self.maxPool(outs)
        outs = self.fc(outs.view(outs.shape[0], -1))

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return outs
