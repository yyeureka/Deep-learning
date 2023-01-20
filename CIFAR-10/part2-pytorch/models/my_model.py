"""
MyModel model.  (c) 2021 Georgia Tech

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
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################

        # TODO: You could try adding batch normalization and dropout layers

        self.conv1 = nn.Conv2d(3, 64, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.pool2 = nn.MaxPool2d(3, 2)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.conv4 = nn.Conv2d(256, 64, 1)
        self.fc1 = nn.Linear(1024, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 10)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################

        outs = self.pool1(F.relu(self.conv1(x)))
        outs = self.pool2(F.relu(self.conv2(outs)))
        outs = F.relu(self.conv3(outs))
        outs = F.relu(self.conv4(outs))
        outs = outs.view(outs.shape[0], -1)
        outs = F.relu(self.fc1(outs))
        outs = F.relu(self.fc2(outs))
        outs = F.softmax(self.fc3(outs))

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs
