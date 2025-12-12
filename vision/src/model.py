import torch
import torch.nn as nn
import torchvision.models as models

class LegoEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128, pretrained=True):
        super(LegoEmbeddingNet, self).__init__()
        
        # specific weights param creation
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        
        # Load MobileNetV3 Small (Efficient & Fast)
        self.backbone = models.mobilenet_v3_small(weights=weights)
        
        # Remove the last classifier layer
        # MobileNetV3 classifier structure: Sequential(Linear, Hardswish, Dropout, Linear)
        # We want to replace the final Linear layer to output 'embedding_dim'
        
        # Get the input features of the last linear layer
        last_channel = self.backbone.classifier[3].in_features
        
        # Replace the classifier
        self.backbone.classifier[3] = nn.Linear(last_channel, embedding_dim)
        
    def forward(self, x):
        # x shape: (N, 3, 224, 224)
        x = self.backbone(x)
        # Normalize embeddings to unit length (critical for metric learning!)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x

if __name__ == "__main__":
    # Test instantiation
    model = LegoEmbeddingNet()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Should be [1, 128]
