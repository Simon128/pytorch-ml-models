# Oriented R-CNN for Object Detection

[Paper](https://arxiv.org/abs/2108.05699)

## Requirements

## Components

### FPN backbone

This component is not provided by this module as it also not introduced by the publication.
The model expects the backbone to accept a batch of images (B, C, H, W) and return a
dictionary of feature maps, where each entry in the dictionary represents an FPN level and
holds a value of shape (B, C', H', W') with C' representing the feature channels and H'/W' representing
the feature map spatial size. The expected behaviour is covered by [unit tests](./tests/backbone.py).

### Oriented RPN

The oriented RPN expects an input corresponding to the output format of the FPN backbone. It uses
normal convolutions to produce rotated region proposals in the midpoint offset representation.

### Rotated RoIAlign

### Oriented R-CNN Head
