from pathlib import Path

import torch
import clip
import numpy as np
from PIL import Image
from typing import Union, List, Optional
import torch.nn.functional as F


class CLIPScore:
    """
    CLIPScore: 图像-文本匹配度评估指标
    基于论文: "CLIPScore: A Reference-free Evaluation Metric for Image Captioning"
    """

    def __init__(self, model_name: str = "ViT-L/14", device: Optional[str] = None):
        """
        初始化CLIPScore

        Args:
            model_name: CLIP模型名称 ('ViT-B/32', 'ViT-B/16', 'ViT-L/14')
            device: 运行设备 ('cuda' 或 'cpu')
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 加载CLIP模型
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

    def compute_image_features(self, images: Union[str, List[str], Image.Image, List[Image.Image]]):
        """
        计算图像特征
        """
        if isinstance(images, (str, Image.Image)):
            images = [images]

        image_tensors = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            # 预处理图像
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            image_tensors.append(img_tensor)

        image_tensors = torch.cat(image_tensors, dim=0)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensors)
            # 归一化特征
            image_features = F.normalize(image_features, dim=-1)

        return image_features

    def compute_text_features(self, texts: Union[str, List[str]]):
        """
        计算文本特征
        """
        if isinstance(texts, str):
            texts = [texts]

        # tokenize文本
        text_tokens = clip.tokenize(texts).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            # 归一化特征
            text_features = F.normalize(text_features, dim=-1)

        return text_features

    def score(self,
              images: Union[str, List[str], Image.Image, List[Image.Image]],
              texts: Union[str, List[str]],
              reduction: str = 'mean'):
        """
        计算CLIPScore

        Args:
            images: 图像路径或PIL Image对象
            texts: 文本描述
            reduction: 聚合方式 ('mean', 'none', 'sum')

        Returns:
            scores: CLIP分数 (范围0-100)
        """
        # 计算特征
        image_features = self.compute_image_features(images)
        text_features = self.compute_text_features(texts)

        # 计算余弦相似度（点积，因为已经归一化）
        similarity = image_features @ text_features.T

        # 转换为CLIPScore（论文中使用的尺度）
        scores = similarity * 100

        # 处理维度匹配问题
        if similarity.shape[0] == 1 and similarity.shape[1] > 1:
            scores = scores.squeeze(0)  # 一个图像对多个文本
        elif similarity.shape[0] > 1 and similarity.shape[1] == 1:
            scores = scores.squeeze(1)  # 多个图像对一个文本

        # 根据reduction参数聚合
        if reduction == 'mean':
            return scores.mean().item()
        elif reduction == 'sum':
            return scores.sum().item()
        else:
            if isinstance(scores, torch.Tensor):
                return scores.cpu().numpy()
            return scores

if __name__ == '__main__':
    model_path = '/mnt/d/studyWork/model/clip-vit-large-patch14'
    device = 'cuda:0'
    current_dir = Path(__file__).parent
    image_path = current_dir.parent / 'results' / 'V-A (-2.0,-2.0) | A person is standing over the ocean.png'
    text = 'an boy plays football'
    clip_score = CLIPScore()
    result = clip_score.score(image_path, text)
    print(result)