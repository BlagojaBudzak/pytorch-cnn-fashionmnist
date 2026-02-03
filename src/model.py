import torch
from torch import nn
class FashionMNISTCNN(nn.Module):
  """
  Replicates the TinyVGG architecture.

  This model uses two convolutional blocks with MaxPool2d layers to extract
  spatial features, followed by a linear classifier..
  """
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
    super().__init__()

    # Block 1: Input size (Batch, 1, 28, 28) -> Output size (Batch, hidden_units, 14, 14)
    self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2) # Reduces spatial dimensions by half
    )

    # Block 2: Input (Batch, hidden_units, 14, 14) -> Output (Batch, hidden_units, 7, 7)
    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    # Classifier: Flattens the 7x7 feature maps into a single vector
    self.classifier = nn.Sequential(
        nn.Flatten(),
        # Calculation: hidden_units * 7 * 7 (from two 2x2 pooling layers on 28x28 input)
        nn.Linear(in_features=hidden_units*7*7,
                  out_features=output_shape)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.classifier(self.conv_block_2(self.conv_block_1(x)))