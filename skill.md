# 迭代式指标提升 Skill

## 目标

使用本 Skill 对当前仓库中的 Python 实验代码进行闭环优化：
先分析代码，提出一个明确创新点，修改相关 `.py` 文件，执行训练、推理或评估代码，
记录每一次指标，并与前一个最好方法对比；如果没有超过，则分析原因并继续提出新的创新点；
如果超过，则记录创新点、参数和新的指标 log。

## 当前项目背景

本仓库是一个基于 SDXL prompt feature 的情绪控制生成项目，核心文件如下：

- `preprocess.py`：从 neutral/emotional prompt 和 valence/arousal 标签构建
  `data/data-cache.pt`。
- `model.py`：定义 `EmotionInjectionTransformer`，包含 arousal、valence 条件注入结构。
- `train.py`：训练模型并打印 `train_loss`、`val_loss`，可选打印 `val_loss_weight`。
- `inference.py`：使用指定 prompt、arousal、valence 和 checkpoint 生成单张图片。
- `inference5x5.py`：对验证 prompt 生成 5x5 valence/arousal 网格图片。
- `metrics/va_evaluate.py`：评估生成图片的 VA 控制效果，输出
  `valence_abs_error` 和 `arousal_abs_error`。
- `metrics/clip_score.py`：评估图片与文本 prompt 的一致性，输出 `CLIPScore`。
- `metrics/clip_iqa.py`：评估无参考图片质量，输出 `CLIP-IQA`。

## 指标方向

修改代码之前，必须先确定主指标和优化方向：

- 训练类指标：越低越好，例如 `val_loss`、`val_loss_weight`。
- VA 控制指标：越低越好，例如 `VA valence_abs_error`、`VA arousal_abs_error`。
- 图文一致性和图片质量指标：越高越好，例如 `CLIPScore`、`CLIP-IQA`。

如果任务没有额外指定，默认优先级如下：

1. VA 控制误差，因为项目核心目标是控制 valence/arousal。
2. CLIPScore，避免生成图片偏离原 prompt。
3. CLIP-IQA，避免图片质量下降。
4. 训练验证损失，作为生成评估成本较高时的早期代理指标。

## 必须遵守的迭代流程

每一轮实验都按以下顺序执行。

### 1. 分析代码和基线

1. 先阅读相关 Python 文件，再提出修改方案。
2. 明确当前基线方法：
   - `model.py` 中的模型结构和情绪条件注入路径；
   - `train.py` 中的 loss、权重、优化器和超参数；
   - `inference.py` 或 `inference5x5.py` 中的推理参数；
   - `metrics/` 下的指标脚本和 log 路径。
3. 如果没有现成基线指标，先运行基线实验或收集已有 log。
4. 按本文档的实验记录模板写入基线结果。

### 2. 提出一个创新点

每一轮只提出一个主要创新点，且创新点必须能在 Python 中独立实现和独立评估。

适合本项目的创新点示例：

- 调整 arousal、valence token 融合到 `EmotionInjectionTransformer` 的方式；
- 为情绪特征注入增加 residual gate；
- 将 loss 目标从直接预测 emotional prompt feature 改为预测 residual delta；
- 增加 density-aware 或 VA-magnitude-aware 的 loss weighting；
- 调整 `scale_factor`、learning rate、batch size、weight decay 或 scheduler；
- 调整推理参数，例如 `guidance_scale`、`num_inference_steps`、seed 策略；
- 改进日志，让每次运行同时保存参数、checkpoint 路径和指标。

不要在同一轮混合多个无关想法。若指标提升，必须能清楚判断是哪一个创新点有效。

### 3. 修改 Python 文件

1. 只修改该创新点必需的 `.py` 文件。
2. 改动保持小而可回滚。
3. 对可调参数增加命令行参数，不要把实验值硬编码进代码。
4. 程序运行时必须打印本次实际使用的关键参数。
5. 除非创新点需要改变默认方法，否则保留原有默认值。

### 4. 执行代码

先用最小但可信的评估运行实验，结果有希望后再扩大规模。

训练示例：

```bash
python train.py \
  --data_cache_path ./data/data-cache.pt \
  --save_dir ./checkpoints/experiment_name \
  --batch_size 1024 \
  --lr 1e-4 \
  --epochs 200 \
  --scale_factor 1.0
```

生成 5x5 验证图片示例：

```bash
python inference5x5.py \
  --prompt_json val_prompt.json \
  --ckpt_path ./checkpoints/experiment_name/best_model.pth \
  --output_dir ./results/experiment_name \
  --seed 0 \
  --device cuda \
  --overwrite
```

VA 指标评估示例：

```bash
python metrics/va_evaluate.py \
  --image_dir ./results/experiment_name \
  --log_path ./logs/experiment_name_va.log \
  --batch_size 16
```

CLIPScore 指标评估示例：

```bash
python metrics/clip_score.py \
  --image_dir ./results/experiment_name \
  --log_path ./logs/experiment_name_clip.log \
  --batch_size 16
```

CLIP-IQA 指标评估示例：

```bash
python metrics/clip_iqa.py \
  --image_dir ./results/experiment_name \
  --log_path ./logs/experiment_name_iqa.log \
  --batch_size 16
```

### 5. 记录每一次指标

每次运行结束后都追加记录，不要跳过失败或变差的实验。

```markdown
## Experiment <编号>: <短名称>

- Date:
- Git commit:
- Changed files:
- Innovation:
- Parameters:
  - train:
  - inference:
  - metrics:
- Commands:
  - `<command>`
- Metrics:
  - train_loss:
  - val_loss:
  - val_loss_weight:
  - VA pred_valence:
  - VA pred_arousal:
  - VA valence_abs_error:
  - VA arousal_abs_error:
  - CLIPScore:
  - CLIP-IQA:
- Previous best:
- Comparison:
  - improved: yes/no
  - metric delta:
- Result:
  - accepted/rejected
- Notes:
```

### 6. 判断是否超过前面方法

使用预先确定的主指标和前一个最好方法比较。

- 若指标越低越好：`new_metric < best_metric` 才算提升。
- 若指标越高越好：`new_metric > best_metric` 才算提升。
- 若指标包含均值和标准差，先比较均值，同时记录标准差。
- 如果主指标提升但副指标下降，必须记录 trade-off，不能只报好结果。

### 7. 如果没有超过

当新指标没有超过 previous best 时：

1. 将本轮实验标记为 rejected。
2. 基于代码和 log 分析可能原因，例如：
   - 优化不稳定；
   - 过拟合或欠拟合；
   - 情绪条件信号太弱或太强；
   - loss weighting 与目标指标不一致；
   - 训练参数和推理参数不匹配；
   - prompt 或图片数量过少导致指标噪声较大。
3. 根据失败原因提出下一轮新的创新点。
4. 再次修改相关 `.py` 文件，只实现新想法。
5. 继续运行并记录下一轮实验。

### 8. 如果超过了

当新指标超过 previous best 时：

1. 将本轮实验标记为 accepted。
2. 记录精确创新点和全部关键参数。
3. 打印清晰 log，包含新指标和 previous best。
4. 保存或标明 winning checkpoint 与输出图片目录。
5. 保留成功的 Python 改动，并说明为什么该改动可能带来提升。

成功时使用如下 log 格式：

```text
[NEW BEST]
innovation: <创新点简述>
parameters: <key=value 参数列表>
primary_metric: <指标名>
previous_best: <旧值>
new_metric: <新值>
delta: <提升幅度>
checkpoint: <checkpoint 路径>
outputs: <输出目录>
log: <log 路径>
```

## 决策规则

- 每轮只改变一个主要想法。
- 优先依赖可量化指标，而不是只看主观图片效果。
- 所有日志都要保留，包括失败实验。
- 只有主指标超过 previous best，才能宣布成功。
- 如果指标波动明显，接受新方法前要换 seed 重跑验证。
- 如果代码运行失败，先记录错误，再做最小修复，并用同一创新点重跑。

## 最终总结

循环结束后，总结以下内容：

1. baseline 指标；
2. 每一个尝试过的创新点；
3. accepted 和 rejected 的实验；
4. 最优参数；
5. 最终指标 log；
6. 修改过的 Python 文件；
7. 剩余风险和下一步可尝试方向。
