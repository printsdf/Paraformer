# 更新摘要 / Summary of Changes

## 你的问题 / Your Questions

根据你的训练日志，你提出了以下问题：
Based on your training logs, you asked:

1. 现在应该怎么做？如何直接使用最好的模型？
   What should I do now? How to directly use the best model?

2. 普通模型和EMA模型有什么区别？
   What's the difference between regular and EMA models?

## 解决方案 / Solution

### ✅ 1. 自动保存最佳模型 / Automatic Best Model Saving

训练脚本现在会自动追踪并保存最佳模型：
The training script now automatically tracks and saves the best models:

- **每个epoch后检查**：如果当前epoch的loss比之前所有epoch都低，自动保存
- **After each epoch**: If current loss is lower than all previous epochs, automatically save
- **保存两个版本**：`best_model.pth` (普通) 和 `best_model_ema.pth` (EMA)
- **Saves two versions**: `best_model.pth` (regular) and `best_model_ema.pth` (EMA)

根据你的训练日志，epoch 92的loss最低(1.616945)，系统会自动保存那个时刻的模型。
Based on your logs, epoch 92 has the lowest loss (1.616945), so the system will automatically save models from that epoch.

### ✅ 2. 简化推理命令 / Simplified Inference Command

现在你可以直接使用最佳模型进行推理：
Now you can directly use the best model for inference:

```bash
# 推荐：使用最佳EMA模型 (性能通常更好)
# Recommended: Use best EMA model (usually better performance)
python test.py --dataset Chesapeake --model_path experiments/best_model_ema.pth --save_path ./results --gpu 0
```

或者如果你在模型目录内：
Or if you're inside the model directory:

```bash
cd experiments
python ../test.py --dataset Chesapeake --model_path best --save_path ../results --gpu 0
```

### ✅ 3. EMA vs 普通模型的区别 / EMA vs Regular Model Difference

#### 普通模型 / Regular Model
- 直接训练更新的模型参数
- Direct training parameter updates
- 参数会随训练波动
- Parameters fluctuate during training

#### EMA模型 (推荐使用!) / EMA Model (Recommended!)
- **指数移动平均**：`EMA参数 = 0.999 × 旧EMA参数 + 0.001 × 当前参数`
- **Exponential Moving Average**: `EMA_param = 0.999 × old_EMA + 0.001 × current`
- **更平滑**：减少训练噪声的影响
- **Smoother**: Reduces impact of training noise
- **更稳定**：不会因个别batch的异常数据波动
- **More stable**: Won't fluctuate due to occasional bad batches
- **泛化能力更强**：在新数据上表现通常更好
- **Better generalization**: Usually performs better on new data

**推荐**：**始终使用EMA模型进行推理/测试**
**Recommendation**: **Always use EMA model for inference/testing**

## 新增文件 / New Files

### 1. `MODEL_SELECTION_GUIDE.md`
详细的双语指南，解释：
Detailed bilingual guide explaining:
- EMA模型的原理和优势
- EMA model principles and advantages
- 如何使用最佳模型
- How to use best models
- 训练日志示例
- Training log examples

### 2. `losses_custom.py`
训练所需的损失函数：
Loss functions needed for training:
- ComboLoss（组合损失：CE + 标签平滑 + Dice）
- ComboLoss (Combined: CE + Label Smoothing + Dice)
- 伪标签生成函数
- Pseudo label generation

### 3. `.gitignore`
标准Python项目忽略文件
Standard Python project ignore file

## 训练日志变化 / Training Log Changes

现在训练时你会看到：
Now during training you will see:

```
[时间] Epoch : 90, CE-branch1 : 1.898868, MCE-branch2: 1.372761, loss: 1.635814, lr: 6.963277e-04
[时间] save best model to experiments/best_model.pth (epoch 90, loss: 1.635814)
[时间] save best EMA model to experiments/best_model_ema.pth (epoch 90, loss: 1.635814)
```

训练结束时：
At training completion:

```
Training completed! Best model was at epoch 92 with loss 1.616945
Best model saved at: experiments/best_model.pth
Best EMA model (recommended) saved at: experiments/best_model_ema.pth
```

## 快速开始 / Quick Start

### 训练 / Training
```bash
python train.py --dataset Chesapeake --batch_size 10 --max_epochs 100 --savepath experiments --gpu 0
```

### 推理 / Inference
```bash
# 使用最佳EMA模型 (推荐)
# Use best EMA model (recommended)
python test.py --dataset Chesapeake --model_path experiments/best_model_ema.pth --save_path ./results --gpu 0
```

## 总结 / Summary

现在你不需要手动查找哪个epoch的模型最好了，系统会自动：
Now you don't need to manually find which epoch has the best model, the system will automatically:

1. ✅ 追踪每个epoch的训练loss
1. ✅ Track training loss for each epoch
2. ✅ 保存loss最低的模型（两个版本）
2. ✅ Save models with lowest loss (both versions)
3. ✅ 在训练结束时告诉你最佳模型在哪个epoch
3. ✅ Tell you at training end which epoch had the best model
4. ✅ 让你可以直接使用最佳模型进行推理
4. ✅ Allow you to directly use the best model for inference

**最重要的**：使用 `best_model_ema.pth` 进行推理，它通常比普通模型性能更好！
**Most important**: Use `best_model_ema.pth` for inference, it usually performs better than regular models!

---

如有任何问题，请查看 `MODEL_SELECTION_GUIDE.md` 获取详细说明。
For any questions, please see `MODEL_SELECTION_GUIDE.md` for detailed instructions.
