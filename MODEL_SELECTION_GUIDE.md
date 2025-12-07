# Model Selection Guide / 模型选择指南

## 什么是EMA模型？/ What is EMA Model?

### 中文说明

**EMA (Exponential Moving Average，指数移动平均)** 是一种提高模型性能的技术。

#### 普通模型 vs EMA模型的区别：

1. **普通模型**：
   - 训练过程中直接更新的模型参数
   - 参数会随着每个batch的训练而波动
   - 可能受到训练数据噪声的影响

2. **EMA模型**：
   - 对模型参数进行指数移动平均
   - 参数更新公式：`EMA_param = decay × EMA_param + (1 - decay) × current_param`
   - 在本项目中，decay = 0.999
   - 更加平滑和稳定，减少了训练过程中的参数波动
   - **通常具有更好的泛化能力和测试性能**

#### 为什么EMA模型更好？

- **平滑性**：EMA模型平均了训练过程中的参数变化，减少了噪声影响
- **稳定性**：不会因为某个batch的异常数据导致参数剧烈变化
- **泛化能力**：由于参数更平滑，在新数据上的表现通常更好
- **已被验证**：在许多深度学习任务中，EMA模型都被证明优于普通模型

#### 推荐使用：

**对于推理/测试，强烈推荐使用EMA模型（best_model_ema.pth）**

---

### English Explanation

**EMA (Exponential Moving Average)** is a technique to improve model performance.

#### Difference between Regular Model and EMA Model:

1. **Regular Model**:
   - Model parameters updated directly during training
   - Parameters fluctuate with each training batch
   - May be affected by training data noise

2. **EMA Model**:
   - Applies exponential moving average to model parameters
   - Update formula: `EMA_param = decay × EMA_param + (1 - decay) × current_param`
   - In this project, decay = 0.999
   - More smooth and stable, reduces parameter fluctuations during training
   - **Usually has better generalization and test performance**

#### Why is EMA Model Better?

- **Smoothness**: EMA model averages parameter changes during training, reducing noise impact
- **Stability**: Won't have drastic parameter changes due to abnormal data in a single batch
- **Generalization**: Due to smoother parameters, usually performs better on new data
- **Proven**: In many deep learning tasks, EMA models have been shown to outperform regular models

#### Recommendation:

**For inference/testing, strongly recommend using EMA model (best_model_ema.pth)**

---

## 如何使用最佳模型 / How to Use Best Model

### 训练过程 / During Training

训练脚本会自动保存两类模型：

The training script automatically saves two types of models:

1. **定期保存的模型** / Regular Checkpoints:
   - `epoch_0.pth`, `epoch_20.pth`, `epoch_40.pth`, ... (每20个epoch保存一次)
   - `epoch_0_ema.pth`, `epoch_20_ema.pth`, `epoch_40_ema.pth`, ...

2. **最佳模型** / Best Models (NEW!):
   - `best_model.pth` - 训练损失最低的普通模型
   - `best_model_ema.pth` - 训练损失最低的EMA模型 (**推荐使用**)

### 推理/测试 / Inference/Testing

#### 方法1：自动加载最佳EMA模型（推荐）/ Method 1: Auto-load Best EMA Model (Recommended)

```bash
python test.py --dataset Chesapeake --model_path best --save_path ./results --gpu 0
```

这会自动加载 `best_model_ema.pth`（最佳EMA模型）

This will automatically load `best_model_ema.pth` (best EMA model)

#### 方法2：自动加载最佳普通模型 / Method 2: Auto-load Best Regular Model

```bash
python test.py --dataset Chesapeake --model_path best_regular --save_path ./results --gpu 0
```

这会自动加载 `best_model.pth`（最佳普通模型）

This will automatically load `best_model.pth` (best regular model)

#### 方法3：手动指定模型路径 / Method 3: Manually Specify Model Path

如果模型在训练目录中（例如 experiments/），可以这样加载：

If models are in training directory (e.g., experiments/), you can load them like:

```bash
python test.py --dataset Chesapeake --model_path experiments/best_model_ema.pth --save_path ./results --gpu 0
```

或使用特定epoch的模型 / Or use a specific epoch model:

```bash
python test.py --dataset Chesapeake --model_path experiments/epoch_99_ema.pth --save_path ./results --gpu 0
```

**注意** / **Note**: 使用 `best` 或 `best_regular` 会在当前工作目录查找模型文件。如果模型在其他目录，请使用完整路径。

Using `best` or `best_regular` looks for model files in the current working directory. Use full path if models are in a different directory.

---

## 训练日志示例 / Training Log Example

训练过程中，你会看到如下日志：

During training, you will see logs like:

```
[19:17:40. 705] Epoch : 89, CE-branch1 : 1.920986, MCE-branch2: 1.380549, loss: 1.650767, lr: 7.419852e-04
[19:24:46. 273] Epoch : 90, CE-branch1 : 1.898868, MCE-branch2: 1.372761, loss: 1.635814, lr: 6.963277e-04
[19:24:46. 500] save best model to experiments/best_model.pth (epoch 90, loss: 1.635814)
[19:24:46. 800] save best EMA model to experiments/best_model_ema.pth (epoch 90, loss: 1.635814)
```

当训练完成时，会显示最佳模型信息：

When training completes, it shows the best model info:

```
[20:28:39. 556] Training completed! Best model was at epoch 90 with loss 1.635814
[20:28:39. 556] Best model saved at: experiments/best_model.pth
[20:28:39. 556] Best EMA model (recommended) saved at: experiments/best_model_ema.pth
```

---

## 总结 / Summary

### 对于训练 / For Training:
- 训练脚本会自动保存最佳模型，无需手动干预
- The training script automatically saves the best model, no manual intervention needed

### 对于推理 / For Inference:
- **推荐**：使用 `--model_path best` 自动加载最佳EMA模型
- **Recommended**: Use `--model_path best` to auto-load the best EMA model
- EMA模型通常比普通模型性能更好
- EMA models usually perform better than regular models

### 文件位置 / File Locations:
- 所有模型文件保存在你指定的 `--savepath` 目录下
- All model files are saved in the `--savepath` directory you specified
- 例如：`experiments/best_model_ema.pth`
- Example: `experiments/best_model_ema.pth`
