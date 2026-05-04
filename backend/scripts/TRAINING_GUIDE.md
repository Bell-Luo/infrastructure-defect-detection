# YOLO11m 训练与评估指南

## 📋 目录

1. [环境准备](#环境准备)
2. [模型训练](#模型训练)
3. [模型评估](#模型评估)
4. [场景化错误分析](#场景化错误分析)
5. [结果解读](#结果解读)

---

## 环境准备

### 1. 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

### 2. 数据集配置

确保 `data.yaml` 配置正确：

```yaml
path: E:/crack-seg
train: train/images
val: valid/images
test: test/images

names:
  0: crack      # 裂缝
  1: spalling   # 剥落
  2: seepage    # 渗水
  3: damage     # 破损
  4: corrosion  # 锈蚀
```

---

## 模型训练

### 快速开始

```bash
cd backend
python scripts/train_yolo11m.py --data data.yaml --epochs 100 --batch 16
```

### 完整参数

```bash
python scripts/train_yolo11m.py \
    --data data.yaml \
    --model_size m \           # 模型大小: n/s/m/l/x
    --epochs 100 \             # 训练轮次
    --batch 16 \               # 批大小 (根据显存调整)
    --imgsz 640 \             # 输入图片尺寸
    --device 0 \              # GPU设备编号 (0=cuda:0, cpu=CPU)
    --project runs \           # 项目目录
    --name crack_detection \   # 实验名称
    --patience 50 \            # 早停耐心值
    --copy_to_backend         # 训练完成后复制模型到backend
```

### 参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--model_size` | YOLO11模型大小 | m (平衡精度和速度) |
| `--epochs` | 训练轮次 | 100-300 |
| `--batch` | 批大小 | 16-32 (看显存) |
| `--imgsz` | 输入图片尺寸 | 640 (标准) / 1280 (高精度) |
| `--device` | 训练设备 | 0 (GPU) / cpu (CPU) |

### 显存不足时的解决方案

```bash
# 方案1: 减小批大小
--batch 8

# 方案2: 使用更小的模型
--model_size s

# 方案3: 使用CPU训练（不推荐，太慢）
--device cpu
```

### 训练输出

训练完成后，模型保存在：
- 最佳模型: `runs/crack_detection/weights/best.pt`
- 最后模型: `runs/crack_detection/weights/last.pt`

训练过程会自动记录：
- Loss曲线
- 评价指标曲线 (Precision, Recall, mAP50, mAP50-95)
- 验证集上的预测结果可视化

---

## 模型评估

### 基本评估

```bash
python scripts/evaluate_model.py \
    --model runs/crack_detection/weights/best.pt \
    --data data.yaml \
    --split val
```

### 完整评估 + 错误分析

```bash
python scripts/evaluate_model.py \
    --model runs/crack_detection/weights/best.pt \
    --data data.yaml \
    --split val \
    --conf 0.25 \
    --iou 0.5 \
    --analyze_errors \
    --output my_evaluation
```

### 输出文件

评估完成后会生成：

1. **JSON报告**: `evaluation_results/my_evaluation.json`
   - 完整的评估指标
   - 每类别AP50
   - 错误分析详情

2. **CSV指标**: `evaluation_results/my_evaluation_metrics.csv`
   - mAP50, mAP50-95
   - Precision, Recall
   - 每类别AP50

3. **CSV错误**: `evaluation_results/my_evaluation_errors.csv`
   - 误检(False Positives)
   - 漏检(False Negatives)
   - 小目标问题

### 评估指标说明

| 指标 | 说明 | 理想值 |
|------|------|--------|
| **Precision** | 精确率，预测正确的比例 | 越高越好 |
| **Recall** | 召回率，检测出所有目标的比例 | 越高越好 |
| **mAP50** | IoU>0.5时的平均精度 | 0.8+ |
| **mAP50-95** | IoU从0.5到0.95的平均精度 | 0.6+ |

---

## 场景化错误分析

针对特定场景的深度分析：

### 分析场景

1. **细小裂缝** - 缺陷面积占图片比例 < 1%
2. **弱纹理渗水** - 边缘模糊、纹理不明显的渗水区域
3. **复杂背景剥落** - 背景纹理复杂，与剥落区域对比度低

### 执行分析

```bash
python scripts/analyze_scenarios.py \
    --model runs/crack_detection/weights/best.pt \
    --data data.yaml \
    --split val \
    --conf 0.25 \
    --output scenario_analysis
```

### 输出内容

1. **JSON报告**: `error_analysis_results/scenario_analysis.json`
   - 各场景的错误列表
   - 统计信息

2. **改进建议**: 自动生成的优化建议

---

## 结果解读

### 如何理解评估结果

#### Precision (精确率)
- 定义: TP / (TP + FP)
- 含义: 预测为正的样本中，真正为正的比例
- **低Precision意味着有很多误检**

#### Recall (召回率)
- 定义: TP / (TP + FN)
- 含义: 实际为正的样本中，被正确预测的比例
- **低Recall意味着有很多漏检**

#### mAP50
- 定义: IoU阈值=0.5时的平均精度
- 含义: 综合考虑Precision和Recall的整体性能
- **mAP50 > 0.8 表示模型性能良好**

### 常见问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| Recall低 | 漏检多 | 提高输入分辨率，降低置信度阈值 |
| Precision低 | 误检多 | 提高置信度阈值，添加负样本 |
| 小目标漏检 | 模型不敏感 | 使用更大的输入尺寸，添加小目标专用层 |
| 弱纹理漏检 | 特征不明显 | 使用注意力机制，增加纹理增强 |
| 复杂背景误检 | 干扰多 | 使用更深backbone，增加数据增强 |

---

## 完整工作流程

```bash
# 1. 环境准备
pip install -r requirements.txt

# 2. 训练模型
python scripts/train_yolo11m.py \
    --data data.yaml \
    --model_size m \
    --epochs 100 \
    --batch 16 \
    --copy_to_backend

# 3. 评估模型
python scripts/evaluate_model.py \
    --model best.pt \
    --data data.yaml \
    --split val \
    --analyze_errors

# 4. 场景化分析
python scripts/analyze_scenarios.py \
    --model best.pt \
    --data data.yaml \
    --split val

# 5. 根据分析结果优化后，重新训练或调整参数
```

---

## 注意事项

1. **GPU显存**: YOLO11m 需要至少 6GB 显存，建议使用 8GB+
2. **训练时间**: 100 epochs 在 RTX 3060 上大约需要 2-4 小时
3. **数据增强**: 训练脚本已内置丰富的数据增强，如需调整可修改代码
4. **早停**: 设置 patience=50，如果50个epoch没有提升会自动停止

---

## 模型部署

训练完成后，将 `best.pt` 复制到 `backend/` 目录：

```bash
copy runs\crack_detection\weights\best.pt backend\best.pt
```

然后启动后端服务即可使用训练好的模型：

```bash
cd backend
python main.py
```
