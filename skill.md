# Iterative Metric Improvement Skill

## Purpose

Use this skill to run a closed-loop improvement process for the Python experiment code in this
repository. The loop is:

1. analyze the current code;
2. propose one focused innovation;
3. modify the relevant `.py` files;
4. execute the code;
5. record every metric;
6. compare the new result against the previous best method;
7. if the result is not better, analyze why and start another iteration;
8. if the result is better, record the innovation, parameters, and final log.

## Repository Context

This repository implements emotion-conditioned SDXL prompt feature generation.

- `preprocess.py`: builds `data/data-cache.pt` from neutral/emotional prompts and
  valence/arousal labels.
- `model.py`: defines `EmotionInjectionTransformer` and the arousal/valence conditioning path.
- `train.py`: trains the model and prints `train_loss`, `val_loss`, and optional
  `val_loss_weight`.
- `inference.py`: generates one image from a prompt, arousal value, valence value, and
  checkpoint.
- `inference5x5.py`: generates a 5x5 valence/arousal grid for validation prompts.
- `metrics/va_evaluate.py`: evaluates generated images with VA predictors and logs
  `valence_abs_error` and `arousal_abs_error`.
- `metrics/clip_score.py`: evaluates image-text alignment with `CLIPScore`.
- `metrics/clip_iqa.py`: evaluates no-reference image quality with `CLIP-IQA`.

## Metric Direction

Before changing code, define the primary metric and its direction.

- Training metrics are better when lower: `val_loss`, `val_loss_weight`.
- VA control metrics are better when lower: `VA valence_abs_error`,
  `VA arousal_abs_error`.
- Image-text and quality metrics are better when higher: `CLIPScore`, `CLIP-IQA`.

Unless the task specifies another priority, use this metric priority:

1. VA control error, because the core task is controllable valence/arousal generation.
2. CLIPScore, to avoid reducing prompt-image alignment.
3. CLIP-IQA, to avoid reducing image quality.
4. Validation loss, as an early proxy when image generation is expensive.

## Required Iteration Loop

Follow this sequence for every experiment.

### 1. Analyze the Code and Baseline

1. Read the relevant Python files before proposing a change.
2. Identify the current baseline method:
   - model architecture and conditioning path in `model.py`;
   - loss, weighting, optimizer, and hyperparameters in `train.py`;
   - inference parameters in `inference.py` or `inference5x5.py`;
   - metric scripts and log paths under `metrics/`.
3. If no baseline metric is available, run the baseline experiment or collect the existing log.
4. Record the baseline with the experiment template below.

### 2. Propose One Innovation

Propose exactly one main innovation per iteration. The innovation must be specific enough to
implement in Python and evaluate independently.

Innovation examples for this repository:

- change how arousal/valence tokens are fused into `EmotionInjectionTransformer`;
- add a residual gate for emotion feature injection;
- change the loss target from direct emotional prompt feature prediction to residual delta
  prediction;
- add density-aware or VA-magnitude-aware loss weighting;
- tune `scale_factor`, learning rate, batch size, weight decay, or scheduler behavior;
- tune inference parameters such as `guidance_scale`, `num_inference_steps`, or seed policy;
- improve logging so each run stores parameters, checkpoint path, output path, and metrics
  together.

Do not mix unrelated ideas in one iteration. If the metric improves, the winning idea must be
easy to identify.

### 3. Modify Python Files

1. Edit only the `.py` files required by the innovation.
2. Keep the change small and reversible.
3. Add command-line arguments for tunable parameters instead of hard-coding experiment values.
4. Print the effective parameters at runtime.
5. Preserve existing defaults unless the innovation intentionally changes the default method.

### 4. Execute the Code

Run the smallest reliable evaluation first. Scale up only if the result is promising.

Training command template:

```bash
python train.py \
  --data_cache_path ./data/data-cache.pt \
  --save_dir ./checkpoints/experiment_name \
  --batch_size 1024 \
  --lr 1e-4 \
  --epochs 200 \
  --scale_factor 1.0
```

5x5 validation generation command template:

```bash
python inference5x5.py \
  --prompt_json val_prompt.json \
  --ckpt_path ./checkpoints/experiment_name/best_model.pth \
  --output_dir ./results/experiment_name \
  --seed 0 \
  --device cuda \
  --overwrite
```

VA evaluation command template:

```bash
python metrics/va_evaluate.py \
  --image_dir ./results/experiment_name \
  --log_path ./logs/experiment_name_va.log \
  --batch_size 16
```

CLIPScore evaluation command template:

```bash
python metrics/clip_score.py \
  --image_dir ./results/experiment_name \
  --log_path ./logs/experiment_name_clip.log \
  --batch_size 16
```

CLIP-IQA evaluation command template:

```bash
python metrics/clip_iqa.py \
  --image_dir ./results/experiment_name \
  --log_path ./logs/experiment_name_iqa.log \
  --batch_size 16
```

### 5. Record Every Metric

Append a log entry after every run. Do not skip failed runs or worse runs.

```markdown
## Experiment <number>: <short_name>

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

### 6. Compare Against the Previous Best Method

Use the preselected primary metric for the decision.

- If lower is better, the new method improves only when `new_metric < best_metric`.
- If higher is better, the new method improves only when `new_metric > best_metric`.
- If the metric includes mean and standard deviation, compare the mean first and record both.
- If the primary metric improves but secondary metrics regress, record the trade-off explicitly.

### 7. If the New Method Does Not Improve

When the new metric does not beat the previous best:

1. Mark the experiment as `rejected`.
2. Analyze likely causes using the code and logs, such as:
   - optimization instability;
   - overfitting or underfitting;
   - emotion conditioning signal is too weak or too strong;
   - loss weighting does not match the target metric;
   - training parameters and inference parameters are mismatched;
   - too few prompts or images created noisy metrics.
3. Propose the next innovation based on the failure analysis.
4. Modify only the `.py` files needed for the new idea.
5. Run the next experiment and record the metrics again.

### 8. If the New Method Improves

When the new metric beats the previous best:

1. Mark the experiment as `accepted`.
2. Record the exact innovation and all key parameters.
3. Print a clear log that includes the new metric and the previous best.
4. Save or identify the winning checkpoint and output image directory.
5. Keep the successful Python changes and explain why the change likely improved the metric.

Use this success log format:

```text
[NEW BEST]
innovation: <short description>
parameters: <key=value list>
primary_metric: <metric name>
previous_best: <old value>
new_metric: <new value>
delta: <improvement amount>
checkpoint: <checkpoint path>
outputs: <output directory>
log: <log path>
```

## Decision Rules

- Change only one main idea per iteration.
- Prefer measurable metrics over subjective visual-only judgments.
- Keep all logs, including failed experiments.
- Do not declare success unless the selected primary metric beats the previous best.
- If metrics are noisy, rerun the same configuration with another seed before accepting it.
- If the code fails, log the error, make the smallest necessary fix, and rerun the same idea.

## Final Report Template

At the end of the loop, summarize:

1. baseline metric;
2. every attempted innovation;
3. accepted and rejected experiments;
4. best parameters;
5. final metric log;
6. modified Python files;
7. remaining risks and next ideas.
