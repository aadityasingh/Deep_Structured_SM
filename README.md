
# Deep Structured Similarity Matching

## Overview

This repository is based off of the code from the Pehlevan group: https://github.com/Pehlevan-Group/Deep_Structured_SM. Thanks for the open sourced code from your paper.

In this project, I take their work further by doing some more experimentation. I also modified (fixed) the convergence criterion for the neural dynamics simulation. I also vectorized the neural dynamics so multiple inputs can be run concurrently. This leads to great speedups when fitting a linear classifier since one doesn't have to iterate through each image in MNIST. 

To see a description of my results, check out the PDF in the repo.
