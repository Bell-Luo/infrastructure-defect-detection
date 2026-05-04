from ultralytics import YOLO
import os
import sys
from pathlib import Path
from datetime import datetime
import json
import csv

def log_info(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] INFO: {message}", flush=True)

def log_error(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] ERROR: {message}", flush=True)

def log_warning(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] WARNING: {message}", flush=True)


class YOLOTrainer:
    """YOLO模型训练器"""

    def __init__(self, data_yaml: str, model_size: str = "m"):
        """
        初始化训练器

        参数:
            data_yaml: 数据集配置文件路径
            model_size: 模型大小 (n/s/m/l/x)
        """
        self.data_yaml = data_yaml
        self.model_size = model_size
        self.model = None
        self.results = None
        self.best_model_path = None
        self.last_model_path = None

        # 创建输出目录
        self.output_dir = Path("training_results")
        self.output_dir.mkdir(exist_ok=True)

        log_info(f"训练器初始化完成")
        log_info(f"数据集配置: {self.data_yaml}")
        log_info(f"模型大小: YOLO11{self.model_size}")

    def load_model(self):
        """加载预训练模型"""
        log_info("正在加载预训练模型...")
        try:
            # 使用YOLO11m预训练权重
            self.model = YOLO(f"yolo11{self.model_size}.pt")
            log_info(f"成功加载预训练模型: yolo11{self.model_size}.pt")
            return True
        except Exception as e:
            log_error(f"加载模型失败: {e}")
            return False

    def train(
        self,
        epochs: int = 100,
        batch_size: int = 8,
        image_size: int = 640,
        device: str = "cpu",
        project_name: str = "runs",
        run_name: str = None,
        patience: int = 50,
        save_period: int = 10
    ):
        """
        训练模型

        参数:
            epochs: 训练轮次
            batch_size: 批大小
            image_size: 输入图片尺寸
            device: 训练设备 ("0"=GPU 0, "cpu"=CPU)
            project_name: 项目名称
            run_name: 本次运行名称
            patience: 早停耐心值
            save_period: 保存周期
        """
        if self.model is None:
            if not self.load_model():
                return None

        if run_name is None:
            run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        log_info("=" * 60)
        log_info("开始YOLO11m迁移学习训练")
        log_info("=" * 60)
        log_info(f"数据集配置: {self.data_yaml}")
        log_info(f"训练轮次: {epochs}")
        log_info(f"批大小: {batch_size}")
        log_info(f"图片尺寸: {image_size}")
        log_info(f"训练设备: {device}")
        log_info(f"运行名称: {run_name}")
        log_info("=" * 60)

        try:
            # 开始训练
            self.results = self.model.train(
                data=self.data_yaml,
                epochs=epochs,
                batch=batch_size,
                imgsz=image_size,
                device=device,
                project=project_name,
                name=run_name,
                exist_ok=True,
                optimizer="AdamW",
                lr0=0.001,
                lrf=0.01,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=3.0,
                warmup_momentum=0.8,
                warmup_bias_lr=0.1,
                box=7.5,
                cls=0.5,
                dfl=1.5,
                close_mosaic=10,
                amp=True,
                patience=patience,
                save=True,
                save_period=save_period,
                cache=True,
                workers=8,
                verbose=True,
                pretrained=True,
                val=True,
                plots=True
            )

            # 记录最佳模型路径
            self.best_model_path = f"{project_name}/{run_name}/weights/best.pt"
            self.last_model_path = f"{project_name}/{run_name}/weights/last.pt"

            log_info("=" * 60)
            log_info("训练完成！")
            log_info(f"最佳模型: {self.best_model_path}")
            log_info(f"最后模型: {self.last_model_path}")
            log_info("=" * 60)

            # 保存训练结果摘要
            self.save_training_summary(run_name)

            return self.results

        except Exception as e:
            log_error(f"训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def save_training_summary(self, run_name: str):
        """保存训练摘要"""
        summary = {
            "run_name": run_name,
            "data_yaml": self.data_yaml,
            "model_size": self.model_size,
            "best_model_path": self.best_model_path,
            "last_model_path": self.last_model_path,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        summary_path = self.output_dir / f"{run_name}_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        log_info(f"训练摘要已保存: {summary_path}")

    def evaluate(self, data_yaml: str = None, model_path: str = None):
        """
        在验证集上评估模型

        参数:
            data_yaml: 数据集配置文件路径
            model_path: 模型路径（默认使用最佳模型）
        """
        if model_path is None:
            model_path = self.best_model_path

        if not os.path.exists(model_path):
            log_error(f"模型文件不存在: {model_path}")
            return None

        log_info("=" * 60)
        log_info("开始模型评估")
        log_info(f"模型路径: {model_path}")
        log_info("=" * 60)

        try:
            # 加载模型
            eval_model = YOLO(model_path)

            # 在验证集上评估
            metrics = eval_model.val(
                data=data_yaml or self.data_yaml,
                split='val',
                verbose=True,
                plots=True
            )

            log_info("=" * 60)
            log_info("评估结果:")
            log_info(f"mAP50: {metrics.box.map50:.4f}")
            log_info(f"mAP50-95: {metrics.box.map:.4f}")
            log_info(f"Precision: {metrics.box.mp:.4f}")
            log_info(f"Recall: {metrics.box.mr:.4f}")
            log_info("=" * 60)

            return metrics

        except Exception as e:
            log_error(f"评估失败: {str(e)}")
            return None

    def export_model(self, format: str = "onnx", model_path: str = None):
        """
        导出模型

        参数:
            format: 导出格式 (onnx/torchscript/tflite/etc.)
            model_path: 模型路径
        """
        if model_path is None:
            model_path = self.best_model_path

        if not os.path.exists(model_path):
            log_error(f"模型文件不存在: {model_path}")
            return None

        log_info(f"正在导出模型为{format}格式...")

        try:
            export_model = YOLO(model_path)
            export_path = export_model.export(format=format)
            log_info(f"模型已导出: {export_path}")
            return export_path
        except Exception as e:
            log_error(f"导出失败: {str(e)}")
            return None


def analyze_training_results(project_name: str, run_name: str):
    """
    分析训练结果，读取results.csv并返回Loss曲线和评价指标

    返回:
        dict: 包含loss_curve, metrics等
    """
    import csv
    results_csv = f"{project_name}/{run_name}/results.csv"

    if not os.path.exists(results_csv):
        log_warning(f"未找到训练结果文件: {results_csv}")
        return None

    try:
        with open(results_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if len(rows) < 2:
            log_warning("训练结果数据不足")
            return None

        last_row = rows[-1]

        analysis = {
            "epochs": [],
            "loss_curve": {
                "box_loss": [],
                "cls_loss": [],
                "dfl_loss": [],
                "val_box_loss": [],
                "val_cls_loss": [],
                "val_dfl_loss": []
            },
            "metrics": {
                "precision": float(last_row.get('metrics/precision(B)', 0)),
                "recall": float(last_row.get('metrics/recall(B)', 0)),
                "mAP50": float(last_row.get('metrics/mAP50(B)', 0)),
                "mAP50_95": float(last_row.get('metrics/mAP50-95(B)', 0))
            },
            "final_epoch": int(float(last_row.get('epoch', 0))) if 'epoch' in last_row else 0,
            "total_epochs": len(rows)
        }

        for row in rows:
            epoch = row.get('epoch', '')
            analysis["epochs"].append(float(epoch) if epoch else 0)
            analysis["loss_curve"]["box_loss"].append(float(row.get('train/box_loss', 0)) if row.get('train/box_loss') else None)
            analysis["loss_curve"]["cls_loss"].append(float(row.get('train/cls_loss', 0)) if row.get('train/cls_loss') else None)
            analysis["loss_curve"]["dfl_loss"].append(float(row.get('train/dfl_loss', 0)) if row.get('train/dfl_loss') else None)
            analysis["loss_curve"]["val_box_loss"].append(float(row.get('val/box_loss', 0)) if row.get('val/box_loss') else None)
            analysis["loss_curve"]["val_cls_loss"].append(float(row.get('val/cls_loss', 0)) if row.get('val/cls_loss') else None)
            analysis["loss_curve"]["val_dfl_loss"].append(float(row.get('val/dfl_loss', 0)) if row.get('val/dfl_loss') else None)

        return analysis

    except Exception as e:
        log_error(f"分析训练结果失败: {e}")
        return None


def analyze_detection_issues(model, data_yaml: str, split: str = 'val'):
    """
    分析误检和漏检原因

    针对以下场景分析:
    - 细小裂缝 (Small Cracks)
    - 弱纹理渗水 (Weak Texture Leakage)
    - 复杂背景剥落 (Complex Background Spalling)

    返回:
        dict: 误检漏检分析报告
    """
    if model is None:
        return None

    log_info("开始误检漏检分析...")

    try:
        from ultralytics import YOLO
        eval_model = YOLO(model.best_model_path if hasattr(model, 'best_model_path') else model)

        metrics = eval_model.val(
            data=data_yaml,
            split=split,
            verbose=True,
            plots=False,
            save_json=True
        )

        analysis = {
            "overall_metrics": {
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
                "mAP50": float(metrics.box.map50),
                "mAP50_95": float(metrics.box.map)
            },
            "per_class_metrics": [],
            "issues": []
        }

        if hasattr(metrics.box, 'ap50'):
            per_class_ap50 = metrics.box.ap50
            class_names = eval_model.names

            for i, ap in enumerate(per_class_ap50):
                class_name = class_names[i] if i < len(class_names) else f"class_{i}"
                analysis["per_class_metrics"].append({
                    "class_id": i,
                    "class_name": class_name,
                    "AP50": float(ap) if ap is not None else 0.0
                })

        low_recall_threshold = 0.5
        low_precision_threshold = 0.5

        for pm in analysis["per_class_metrics"]:
            ap50 = pm["AP50"]
            if ap50 < low_recall_threshold * 100:
                issue_type = "漏检风险"
                if "裂缝" in pm["class_name"] or "crack" in pm["class_name"].lower():
                    issue_desc = "细小裂缝检测困难，可能存在漏检"
                elif "渗水" in pm["class_name"] or "leak" in pm["class_name"].lower():
                    issue_desc = "弱纹理渗水检测困难，边缘模糊导致漏检"
                elif "剥落" in pm["class_name"] or "spall" in pm["class_name"].lower():
                    issue_desc = "复杂背景剥落检测困难，与背景混淆导致漏检"
                else:
                    issue_desc = f"{pm['class_name']}类检测性能较差"

                analysis["issues"].append({
                    "type": issue_type,
                    "class_name": pm["class_name"],
                    "description": issue_desc,
                    "severity": "high" if ap50 < 30 else "medium"
                })

            if analysis["overall_metrics"]["precision"] < low_precision_threshold:
                analysis["issues"].append({
                    "type": "误检风险",
                    "class_name": pm["class_name"],
                    "description": "整体误检率较高，可能存在背景误识别",
                    "severity": "medium"
                })

        log_info("误检漏检分析完成")
        return analysis

    except Exception as e:
        log_error(f"误检漏检分析失败: {e}")
        return None


def find_data_yaml(base_path: str = None):
    """自动查找数据集配置文件"""
    if base_path is None:
        base_path = os.getcwd()

    possible_paths = [
        os.path.join(base_path, "data.yaml"),
        os.path.join(base_path, "..", "data.yaml"),
        "E:/crack-seg/data.yaml",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            log_info(f"找到数据集配置: {path}")
            return path

    log_error("未找到数据集配置文件 data.yaml")
    return None


def copy_model_to_backend(model_path: str, dest_path: str = "best.pt"):
    """将训练好的模型复制到backend目录"""
    import shutil

    src = Path(model_path)
    backend_dir = Path(__file__).parent.parent
    dst = backend_dir / "best.pt"

    if src.exists():
        shutil.copy(src, dst)
        log_info(f"模型已复制到: {dst.absolute()}")
        return str(dst.absolute())
    else:
        log_error(f"源模型文件不存在: {src}")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLO11m迁移学习训练脚本")
    parser.add_argument("--data", type=str, default=None, help="数据集配置文件路径")
    parser.add_argument("--model_size", type=str, default="m", choices=["n", "s", "m", "l", "x"], help="模型大小")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮次")
    parser.add_argument("--batch", type=int, default=16, help="批大小")
    parser.add_argument("--imgsz", type=int, default=640, help="图片尺寸")
    parser.add_argument("--device", type=str, default="cpu", help="训练设备 (0=cuda, cpu=cpu)")
    parser.add_argument("--project", type=str, default="runs", help="项目名称")
    parser.add_argument("--name", type=str, default=None, help="任务名称")
    parser.add_argument("--patience", type=int, default=50, help="早停耐心值")
    parser.add_argument("--copy_to_backend", action="store_true", help="训练完成后复制模型到backend目录")
    parser.add_argument("--evaluate_only", action="store_true", help="仅运行评估")
    parser.add_argument("--export", type=str, default=None, help="导出格式 (onnx/torchscript)")

    args = parser.parse_args()

    # 查找数据集配置
    data_yaml = args.data or find_data_yaml()

    if data_yaml is None:
        log_error("请提供数据集配置文件路径 --data")
        exit(1)

    # 创建训练器
    trainer = YOLOTrainer(data_yaml=data_yaml, model_size=args.model_size)

    if args.evaluate_only:
        # 仅运行评估
        metrics = trainer.evaluate()
        if metrics:
            print(f"\n最终评估结果:")
            print(f"  mAP50: {metrics.box.map50:.4f}")
            print(f"  mAP50-95: {metrics.box.map:.4f}")
            print(f"  Precision: {metrics.box.mp:.4f}")
            print(f"  Recall: {metrics.box.mr:.4f}")
    else:
        # 执行训练
        results = trainer.train(
            epochs=args.epochs,
            batch_size=args.batch,
            image_size=args.imgsz,
            device=args.device,
            project_name=args.project,
            run_name=args.name,
            patience=args.patience
        )

        if results is not None:
            # 如果需要，复制模型到backend目录
            if args.copy_to_backend:
                copy_model_to_backend(trainer.best_model_path)

            # 如果需要，导出模型
            if args.export:
                trainer.export_model(format=args.export)
