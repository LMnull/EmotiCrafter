from pathlib import Path
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPRegressor1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=512, out_features=1, bias=True),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)


class va_predictor(torch.nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_path).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.device = device
        self.ar = CLIPRegressor1().to(device)
        self.vr = CLIPRegressor1().to(device)
        # ar.load_state_dict(torch.load('.../arousal1_CLIP_lr=0.001_loss=MSELoss_sc=test_cuda-1.pth'))
        self.ar.load_state_dict(torch.load('./arousal1_CLIP_lr=0.001_loss=MSELoss_sc=test_cuda-1.pth',
                                           map_location=torch.device('cuda:0')))
        self.vr.load_state_dict(torch.load('./valence1_CLIP_lr=0.0001_loss=MSELoss_sc=test_cuda.pth'))
        self.eval()

    def forward(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            return self.vr(image_features) * 6 - 3, self.ar(image_features) * 6 - 3


if __name__ == '__main__':
    current_dir = Path(__file__).parent
    model_path = current_dir.parent / 'model' / 'clip-vit-base-patch32'
    image_path = current_dir.parent / 'results' / 'V-A (3.0,0.0) | An oil painting shows an astronaut.png'
    device = 'cuda'
    va_evaluator = va_predictor(model_path, device)
    pil_image = Image.open(image_path)
    evaluate_result = va_evaluator(pil_image)
    print(evaluate_result)
