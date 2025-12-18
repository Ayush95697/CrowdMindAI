import cv2
import numpy as np
import time
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import functional as T
import warnings
from PIL import Image
from model import CrowdRiskClassifier, CSRNet
import gc

warnings.filterwarnings("ignore", category=UserWarning)


# ======================================================================
# ====== DensityEstimationCounter CLASS ======
# ======================================================================
class DensityEstimationCounter:
    def __init__(self, model_path="assets/partB_model_best.pth.tar"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"✅ DensityEstimationCounter using device: {self.device}")

        # CORRECT CSRNet ARCHITECTURE TO MATCH THE CHECKPOINT
        class CSRNet(nn.Module):
            def __init__(self):
                super(CSRNet, self).__init__()
                self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
                self.backend_feat = [512, 512, 512, 256, 128, 64]
                self.frontend = self.make_layers(self.frontend_feat)
                self.backend = self.make_layers(self.backend_feat, in_channels=512, dilation=True)
                self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

            def make_layers(self, cfg, in_channels=3, batch_norm=False, dilation=False):
                if dilation:
                    d_rate = 2
                else:
                    d_rate = 1
                layers = []
                for v in cfg:
                    if v == 'M':
                        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                    else:
                        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
                        if batch_norm:
                            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                        else:
                            layers += [conv2d, nn.ReLU(inplace=True)]
                        in_channels = v
                return nn.Sequential(*layers)

            def forward(self, x):
                x = self.frontend(x)
                x = self.backend(x)
                x = self.output_layer(x)
                return x

        self.model = CSRNet().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        print("✅ Initialized and loaded weights for CSRNet Model")

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
            return int(people_count)
        except Exception as e:
            print(f"⚠️ CSRNet model failed to count people. Using fallback count. Error: {e}")
            return 0


# ======================================================================
# ====== CrowdMindProcessor CLASS ======
# ======================================================================
# ====== CrowdMindProcessor CLASS ======
# ======================================================================
class CrowdMindProcessor:
    def __init__(self):
        # (Rest of the __init__ method remains the same)
        self.counter = DensityEstimationCounter()
        self.method = "DL"
        self.has_risk_model = False

        self.risk_labels = {
            0: "Low",
            1: "Medium",
            2: "High"
        }

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        try:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.risk_model = CrowdRiskClassifier(num_classes=3).to(self.device)
            self.risk_model.load_state_dict(
                torch.load("assets/crowd_risk_dl.pth", map_location=self.device)
            )
            self.risk_model.eval()
            print(f"✅ Loaded Deep Learning risk model onto device: {self.device}")
            self.has_risk_model = True
        except Exception as e:
            print(f"⚠️ Could not load DL risk model. Error: {e}")

    def assess_risk(self, frame, people_count):
        # (This method remains the same)
        if people_count < 50:
            return 0, 0.99

        if self.has_risk_model:
            try:
                preprocessed_frame = self.transform(frame).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    outputs = torch.nn.functional.softmax(self.risk_model(preprocessed_frame), dim=1)
                    predicted_class = torch.argmax(outputs, dim=1).item()
                    predicted_prob = outputs[0, predicted_class].item()

                return predicted_class, predicted_prob
            except Exception as e:
                print(f"⚠️ DL model failed to assess risk. Using fallback risk. Error: {e}")
                return 0, 0.5
        else:
            if people_count < 50:
                return 0, 0.99
            elif people_count < 250:
                return 1, 0.75
            else:
                return 2, 0.90

    def process_frame(self, frame):
        # The key change is here:
        # 1. We get both the people count and the density map.
        people_count, density_map = self.counter.count_people(frame)

        # 2. We assess the risk based on the frame and people count.
        risk_class, risk_prob = self.assess_risk(frame, people_count)
        risk_percent = int(risk_prob * 100)

        # 3. We return the original frame without applying any overlay.
        # This ensures the video feed is clean.
        result_frame = frame

        del frame
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # 4. We return the clean frame and all the calculated data for the Streamlit UI.
        return result_frame, people_count, risk_class, risk_percent, density_map


# Ensure your DensityEstimationCounter.count_people also returns the density_map.
class DensityEstimationCounter:
    def count_people(self, frame):
        try:
            input_tensor = self.preprocess_frame(frame)
            with torch.no_grad():
                density_map = self.model(input_tensor)
            people_count = torch.sum(density_map).item()
            # The change is to return both the count AND the density map
            return int(people_count), density_map
        except Exception as e:
            print(f"⚠️ CSRNet model failed to count people. Using fallback count. Error: {e}")
            return 0, None