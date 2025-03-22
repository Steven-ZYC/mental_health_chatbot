import torch
import torch.nn as nn
from torchvision import models, transforms
import json
from typing import Dict, Optional

class ModelLoader:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Optional[nn.Module] = None
        self.class_labels: Optional[Dict[str, int]] = None
        self.transform = self._get_transform()
    
    def load_model(self, model_path: str, class_labels_path: str) -> None:
        """Load the emotion detection model and class labels."""
        # Load class labels
        with open(class_labels_path, 'r') as f:
            self.class_labels = json.load(f)
        
        # Initialize and load model
        self.model = EmotionResNet(num_classes=len(self.class_labels)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def _get_transform(self) -> transforms.Compose:
        """Get image transformation pipeline."""
        return transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.225])
        ])

class EmotionResNet(nn.Module):
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.base_model = models.resnet18(weights=None)
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)