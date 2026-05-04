# 基础设施外观缺陷智能检测系统

## 📋 项目介绍
基于 YOLO11 的基础设施外观缺陷检测系统，支持图片和视频检测。

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/你的用户名/infrastructure-defect-detection.git
cd infrastructure-defect-detection
```

### 2. 配置数据集
```bash
cd backend
cp data_example.yaml data.yaml
# 编辑 data.yaml，把 path 改成你的数据集路径
```

### 3. 安装依赖
```bash
# 后端
cd backend
pip install -r requirements.txt

# 前端（新终端）
cd ../frontend
npm install
```

### 4. 训练模型
```bash
cd backend
# 快速训练（适合CPU）
python scripts/quick_train.py

# 正式训练（适合GPU）
python scripts/train_yolo11m.py --data data.yaml --epochs 100 --batch 16 --device 0 --copy_to_backend
```

### 5. 启动服务
```bash
# 后端（终端1）
cd backend
python main.py

# 前端（终端2）
cd frontend
npm run dev
```

### 6. 访问网页
打开浏览器访问：http://localhost:5173

---

## 📁 项目结构
```
infrastructure-defect-detection/
├── backend/              # 后端（FastAPI + YOLO）
│   ├── main.py          # 主服务
│   ├── data_example.yaml # 数据集配置示例
│   ├── requirements.txt  # 依赖包
│   └── scripts/         # 训练脚本
├── frontend/            # 前端（React）
│   └── src/
│       └── App.jsx     # 主页面
└── README.md           # 说明文档
```

---

## 🎯 功能说明

### 系统健康检查
检查后端服务和模型加载状态

### 模型训练信息
显示训练指标、样本数、类别等

### 图片缺陷检测
上传图片 → 检测 → 显示结果 → 导出报告

### 视频缺陷检测
上传视频 → 逐帧检测 → 显示结果 → 导出报告

### 数据集信息
查看数据集统计

### 误检漏检分析
分析模型表现和问题

---

## 💡 提示

- 如果没有数据集，后端会自动进入模拟模式
- `best.pt` 是训练好的模型，每次训练会覆盖
- 训练需要耐心，时间取决于数据量和设备性能

---

## 📊 依赖

- Python 3.9+
- Node.js 16+
- Ultralytics YOLO
- FastAPI
- React + Vite
