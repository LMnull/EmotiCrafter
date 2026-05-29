from pathlib import Path

import torch
import clip
import numpy as np
from PIL import Image
from typing import Union, List, Tuple, Optional
import torch.nn.functional as F

# todo 数据输出格式修改为 “均值 ± 标准差”

class CLIPIQA:
    """
    CLIP-IQA: 基于CLIP的无参考图像质量评估
    基于论文: "CLIP-IQA: Towards Blind Image Quality Assessment with CLIP"
    """

    def __init__(self,
                 model_name: str = "ViT-L/14",
                 device: Optional[str] = None,
                 use_soft_prompts: bool = True):
        """
        初始化CLIP-IQA

        Args:
            model_name: CLIP模型名称
            device: 运行设备
            use_soft_prompts: 是否使用软提示词增强
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 加载CLIP模型
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        self.use_soft_prompts = use_soft_prompts

        # 定义质量相关的提示词对
        self.quality_prompts = {
            'good': [
                "good quality photo",
                "high quality image",
                "clear image",
                "sharp photo",
                "perfect quality",
                "excellent image"
            ],
            'bad': [
                "bad quality photo",
                "low quality image",
                "blurry image",
                "noisy photo",
                "poor quality",
                "terrible image"
            ]
        }

        # 更细粒度的质量维度
        self.dimension_prompts = {
            'sharpness': {
                'good': ["sharp image", "clear details", "well-defined edges"],
                'bad': ["blurry image", "fuzzy details", "unclear edges"]
            },
            'noise': {
                'good': ["clean image", "noise-free photo", "smooth areas"],
                'bad': ["noisy image", "grainy photo", "speckled image"]
            },
            'contrast': {
                'good': ["good contrast", "vibrant image", "well-exposed"],
                'bad': ["poor contrast", "washed out", "too dark or bright"]
            },
            'color': {
                'good': ["natural colors", "vivid colors", "accurate color"],
                'bad': ["color distortion", "unnatural colors", "color cast"]
            }
        }

    def encode_text_prompts(self, prompts: List[str]) -> torch.Tensor:
        """
        编码文本提示词
        """
        text_tokens = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
        return text_features

    def encode_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        编码单张图像
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = F.normalize(image_features, dim=-1)

        return image_features

    def compute_quality_score(self,
                              image: Union[str, Image.Image, torch.Tensor]) -> float:
        """
        计算图像质量分数

        Args:
            image: 输入图像
            method: 计算方法
                  'pair_comparison': 比较好坏提示词对的相似度
                  'direct_regression': 直接回归到质量分数
                  'ensemble': 集成多种方法
        """
        image_features = self.encode_image(image)
        return self._direct_regression_score(image_features)

    def _direct_regression_score(self, image_features: torch.Tensor) -> float:
        """
        直接回归到质量分数
        """
        # 使用质量锚点
        quality_levels = [
            "excellent quality",
            "good quality",
            "fair quality",
            "poor quality",
            "bad quality"
        ]

        # 质量分数映射
        quality_scores = [100, 75, 50, 25, 0]

        # 编码质量锚点
        anchor_features = self.encode_text_prompts(quality_levels)

        # 计算与各锚点的相似度
        similarities = image_features @ anchor_features.T

        # 加权平均得到质量分数
        weights = F.softmax(similarities * 10, dim=-1)
        weighted_score = (weights * torch.tensor(quality_scores).to(self.device)).sum()

        return weighted_score.item() / 100


if __name__ == '__main__':
    current_dir = Path(__file__).parent
    image_path = current_dir.parent / 'results' / 'V-A (-2.0,-2.0) | A person is standing over the ocean.png'
    clip_iqa = CLIPIQA()
    result = clip_iqa.compute_quality_score(image_path)
    print(result)
