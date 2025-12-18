import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as T
import gc
from model import CSRNet, CrowdRiskClassifier

class DensityEstimationCounter:
    def __init__(self, model_path="assets/partB_model_best.pth.tar"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Initialize CSRNet without loading VGG weights since we load checkpoint
        self.model = CSRNet(load_vgg_weights=False).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        print(f"✅ CSRNet initialized on {self.device}")

    def preprocess_frame(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = T.to_tensor(img)
        img_tensor = T.normalize(img_tensor, mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        return img_tensor.unsqueeze(0).to(self.device)

    def count_people(self, frame):
        try:
            input_tensor = self.preprocess_frame(frame)
            with torch.no_grad():
                density_map = self.model(input_tensor)
            people_count = torch.sum(density_map).item()
            return int(people_count), density_map
        except Exception as e:
            print(f"⚠️ CSRNet Error: {e}")
            return 0, None

class CrowdMindProcessor:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.counter = DensityEstimationCounter()
        self.has_risk_model = False
        self.risk_labels = {0: "Low", 1: "Medium", 2: "High"}

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        try:
            self.risk_model = CrowdRiskClassifier(num_classes=3).to(self.device)
            self.risk_model.load_state_dict(
                torch.load("assets/crowd_risk_dl.pth", map_location=self.device)
            )
            self.risk_model.eval()
            print(f"✅ Risk Model loaded on {self.device}")
            self.has_risk_model = True
        except Exception as e:
            print(f"⚠️ Risk Model Error: {e}")

    def assess_risk(self, frame, people_count):
        if people_count < 50:
            return 0, 0.99
            
        if self.has_risk_model:
            try:
                preprocessed_frame = self.transform(frame).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.risk_model(preprocessed_frame)
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    predicted_class = torch.argmax(probs, dim=1).item()
                    predicted_prob = probs[0, predicted_class].item()
                return predicted_class, predicted_prob
            except Exception as e:
                print(f"⚠️ Inference Error: {e}")
                return 0, 0.5
        else:
            # Simple heuristic fallback
            if people_count < 250: return 1, 0.75
            return 2, 0.90

    def process_frame(self, frame):
        people_count, density_map = self.counter.count_people(frame)
        risk_class, risk_prob = self.assess_risk(frame, people_count)
        
        # Memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return frame, people_count, risk_class, int(risk_prob * 100), density_map
