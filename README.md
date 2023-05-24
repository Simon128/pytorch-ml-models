# Pytorch C&P Models

```diff
IN CONSTRUCTION
```

Pytorch Copy and Paste Models offers a variety of isolated machine learning models ready for copy and paste to
accelerate benchmark and scientific developments.

## Motivation

The motivation behind this repository is to fill a use case niche between
approaches like [OpenMMLab](https://github.com/open-mmlab) and development from scratch.
While the latter can be unnecessarily time consuming, the former might dictate a pipeline and model structure that
conflicts your project requirements.
Therefore, the idea of this repository is to offer a variety of model implementations, that can simply be
"copy and pasted" for further modifications or to be inserted into a pytorch based ML pipeline.
Every model is implemented in its own isolated module without shared functionality between different models.
This heavily violates the DRY programming principle but allows to extract only the necessary implementations
for your ML project without any unwanted additional code. These modules do not include a training/test pipeline
on purpose and only offer the required components as presented in their corresponding publication for your usage.

## Design Principle

The implementation of each model follows 3 simple principles:

1. **Minimalism**: Only the functionality presented in the model's publication is implemented, no extra features.
2. **Test Driven**: For each model implementation a 'tests' module will be included, which will cover the model's functionality with tests. This way, you will be able to quickly verify your changes to the model after copy and pasting it.
3. **No secondary dependencies**: Each model implementation should only use primary dependencies like numpy and pytorch.

```diff
- THE THIRD PRINCIPLE WILL BE VIOLATED AT THE MOMENT. It is planned to follow this principle in the future.
```
