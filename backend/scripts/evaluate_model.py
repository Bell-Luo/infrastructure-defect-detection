from ultralytics import YOLO
import os
import sys
from pathlib import Path
from datetime import datetime
import json
import csv
import cv2
import numpy as np
from collections import defaultdict

def log_info(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] INFO: {message}", flush=True)

def log_error(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] ERROR: {message}", flush=True)

def log_warning(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] WARNING: {message}", flush=True)


class ModelEvaluator:
    """模型评估器"""

    def __init__(self, model_path: str, data_yaml: str):
        """
        初始化评估器

        参数:
            model_path: 模型文件路径
            data_yaml: 数据集配置文件路径
        """
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.model = None
        self.class_names = []

        # 评估结果存储
        self.results = {
            "model_path": model_path,
            "data_yaml": data_yaml,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": {},
            "per_class_metrics": {},
            "error_analysis": {
                "false_positives": [],
                "false_negatives": [],
                "small_objects": [],
                "weak_texture": [],
                "complex_background": []
            }
        }

        # 创建输出目录
        self.output_dir = Path("evaluation_results")
        self.output_dir.mkdir(exist_ok=True)

        log_info(f"评估器初始化完成")
        log_info(f"模型路径: {model_path}")
        log_info(f"数据集配置: {data_yaml}")

    def load_model(self):
        """加载模型"""
        if not os.path.exists(self.model_path):
            log_error(f"模型文件不存在: {self.model_path}")
            return False

        try:
            self.model = YOLO(self.model_path)
            # 获取类别名称
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
                if isinstance(self.class_names, dict):
                    self.class_names = [self.class_names[i] for i in sorted(self.class_names.keys())]
            log_info(f"成功加载模型，类别: {self.class_names}")
            return True
        except Exception as e:
            log_error(f"加载模型失败: {e}")
            return False

    def evaluate(self, split: str = 'val', conf_threshold: float = 0.25, iou_threshold: float = 0.5):
        """
        在指定数据集上评估模型

        参数:
            split: 数据集划分 (train/val/test)
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
        """
        if self.model is None:
            if not self.load_model():
                return None

        log_info("=" * 60)
        log_info(f"开始模型评估 (数据集: {split})")
        log_info(f"置信度阈值: {conf_threshold}")
        log_info(f"IOU阈值: {iou_threshold}")
        log_info("=" * 60)

        try:
            # 运行验证
            metrics = self.model.val(
                data=self.data_yaml,
                split=split,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=True,
                plots=True,
                save_json=True
            )

            # 记录指标
            self.results["metrics"] = {
                "mAP50": float(metrics.box.map50) if hasattr(metrics.box, 'map50') else 0.0,
                "mAP50-95": float(metrics.box.map) if hasattr(metrics.box, 'map') else 0.0,
                "Precision": float(metrics.box.mp) if hasattr(metrics.box, 'mp') else 0.0,
                "Recall": float(metrics.box.mr) if hasattr(metrics.box, 'mr') else 0.0,
                "conf_threshold": conf_threshold,
                "iou_threshold": iou_threshold
            }

            # 记录每类别指标
            if hasattr(metrics.box, 'ap50'):
                ap50_per_class = metrics.box.ap50
                if isinstance(ap50_per_class, np.ndarray):
                    for i, ap in enumerate(ap50_per_class):
                        class_name = self.class_names[i] if i < len(self.class_names) else f"class_{i}"
                        self.results["per_class_metrics"][class_name] = {
                            "AP50": float(ap) if not np.isnan(ap) else 0.0
                        }

            log_info("=" * 60)
            log_info("评估结果:")
            log_info(f"  mAP50: {self.results['metrics']['mAP50']:.4f}")
            log_info(f"  mAP50-95: {self.results['metrics']['mAP50-95']:.4f}")
            log_info(f"  Precision: {self.results['metrics']['Precision']:.4f}")
            log_info(f"  Recall: {self.results['metrics']['Recall']:.4f}")
            log_info("=" * 60)

            return metrics

        except Exception as e:
            log_error(f"评估失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def analyze_errors(self, split: str = 'val', conf_threshold: float = 0.25):
        """
        分析误检和漏检

        参数:
            split: 数据集划分
            conf_threshold: 置信度阈值
        """
        if self.model is None:
            if not self.load_model():
                return None

        log_info("=" * 60)
        log_info("开始错误分析")
        log_info("=" * 60)

        # 获取数据集路径
        import yaml
        with open(self.data_yaml, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)

        dataset_path = Path(data_config.get('path', '.'))
        images_dir = dataset_path / split / 'images'
        labels_dir = dataset_path / split / 'labels'

        if not images_dir.exists():
            log_error(f"图片目录不存在: {images_dir}")
            return None

        # 遍历所有图片
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        log_info(f"分析图片数量: {len(image_files)}")

        for img_path in image_files:
            try:
                # 读取图片和标签
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                h, w = img.shape[:2]
                label_path = labels_dir / (img_path.stem + ".txt")

                # 获取真实标注
                gt_boxes = []
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cls_id = int(parts[0])
                                x, y, bw, bh = map(float, parts[1:5])
                                # 转换为像素坐标
                                x1 = int((x - bw/2) * w)
                                y1 = int((y - bh/2) * h)
                                x2 = int((x + bw/2) * w)
                                y2 = int((y + bh/2) * h)
                                gt_boxes.append({
                                    'class_id': cls_id,
                                    'bbox': [x1, y1, x2, y2],
                                    'area': (x2-x1) * (y2-y1)
                                })

                # 获取预测结果
                results = self.model.predict(str(img_path), conf=conf_threshold, verbose=False)
                pred_boxes = []
                for r in results:
                    if r.boxes is not None:
                        for box in r.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            cls_id = int(box.cls[0].item())
                            conf = float(box.conf[0].item())
                            area = (x2-x1) * (y2-y1)
                            pred_boxes.append({
                                'class_id': cls_id,
                                'bbox': [x1, y1, x2, y2],
                                'confidence': conf,
                                'area': area
                            })

                # 分析错误类型
                self._analyze_image_errors(img_path.name, gt_boxes, pred_boxes, w, h)

            except Exception as e:
                log_warning(f"处理图片失败 {img_path.name}: {str(e)}")
                continue

        # 统计错误类型
        self._summarize_errors()

        log_info("错误分析完成")
        return self.results["error_analysis"]

    def _analyze_image_errors(self, img_name, gt_boxes, pred_boxes, img_w, img_h):
        """分析单张图片的错误"""
        iou_threshold = 0.5

        # 匹配预测框和真实框
        matched_gt = set()
        matched_pred = set()

        for p_idx, pred in enumerate(pred_boxes):
            px1, py1, px2, py2 = map(int, pred['bbox'])

            best_iou = 0
            best_gt_idx = -1

            for g_idx, gt in enumerate(gt_boxes):
                if g_idx in matched_gt:
                    continue
                if gt['class_id'] != pred['class_id']:
                    continue

                gx1, gy1, gx2, gy2 = gt['bbox']

                # 计算IOU
                inter_x1 = max(px1, gx1)
                inter_y1 = max(py1, gy1)
                inter_x2 = min(px2, gx2)
                inter_y2 = min(py2, gy2)

                if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                    iou = 0
                else:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    pred_area = (px2 - px1) * (py2 - py1)
                    gt_area = (gx2 - gx1) * (gy2 - gy1)
                    union_area = pred_area + gt_area - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0

                if iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = g_idx
                    break

            if best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                matched_pred.add(p_idx)
            else:
                # 误检（预测框没有匹配的真实框）
                area = pred['area']
                img_area = img_w * img_h
                area_ratio = area / img_area

                error_entry = {
                    'image': img_name,
                    'class_id': pred['class_id'],
                    'class_name': self.class_names[pred['class_id']] if pred['class_id'] < len(self.class_names) else 'unknown',
                    'confidence': pred['confidence'],
                    'bbox': [int(x) for x in pred['bbox']],
                    'area_ratio': area_ratio
                }

                # 细分错误类型
                if area_ratio < 0.01:  # 小目标
                    self.results["error_analysis"]["small_objects"].append(error_entry)
                else:
                    self.results["error_analysis"]["false_positives"].append(error_entry)

        # 漏检（真实框没有匹配的预测框）
        for g_idx, gt in enumerate(gt_boxes):
            if g_idx in matched_gt:
                continue

            area = gt['area']
            img_area = img_w * img_h
            area_ratio = area / img_area

            error_entry = {
                'image': img_name,
                'class_id': gt['class_id'],
                'class_name': self.class_names[gt['class_id']] if gt['class_id'] < len(self.class_names) else 'unknown',
                'bbox': gt['bbox'],
                'area_ratio': area_ratio
            }

            if area_ratio < 0.01:
                self.results["error_analysis"]["small_objects"].append(error_entry)
            else:
                self.results["error_analysis"]["false_negatives"].append(error_entry)

    def _summarize_errors(self):
        """总结错误类型"""
        error_analysis = self.results["error_analysis"]

        log_info("\n错误分析摘要:")
        log_info(f"  误检总数 (FP): {len(error_analysis['false_positives'])}")
        log_info(f"  漏检总数 (FN): {len(error_analysis['false_negatives'])}")
        log_info(f"  小目标问题: {len(error_analysis['small_objects'])}")
        log_info(f"  弱纹理/复杂背景问题: {len(error_analysis['weak_texture'])} + {len(error_analysis['complex_background'])}")

        # 按类别统计误检漏检
        fp_by_class = defaultdict(int)
        fn_by_class = defaultdict(int)

        for fp in error_analysis['false_positives']:
            fp_by_class[fp['class_name']] += 1

        for fn in error_analysis['false_negatives']:
            fn_by_class[fn['class_name']] += 1

        log_info("\n误检按类别分布:")
        for cls, count in fp_by_class.items():
            log_info(f"  {cls}: {count}")

        log_info("\n漏检按类别分布:")
        for cls, count in fn_by_class.items():
            log_info(f"  {cls}: {count}")

    def generate_report(self, output_name: str = None):
        """生成评估报告"""
        if output_name is None:
            output_name = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 保存JSON报告
        json_path = self.output_dir / f"{output_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)

        # 保存CSV报告
        csv_path = self.output_dir / f"{output_name}_metrics.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['指标', '值'])
            writer.writerow(['模型路径', self.results['model_path']])
            writer.writerow(['数据集配置', self.results['data_yaml']])
            writer.writerow(['评估时间', self.results['timestamp']])
            writer.writerow([])
            writer.writerow(['评估指标', ''])
            for key, value in self.results['metrics'].items():
                writer.writerow([key, value])
            writer.writerow([])
            writer.writerow(['每类别AP50', ''])
            for cls, metrics in self.results['per_class_metrics'].items():
                writer.writerow([cls, metrics.get('AP50', 0)])

        # 保存错误分析报告
        error_csv_path = self.output_dir / f"{output_name}_errors.csv"
        with open(error_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['错误类型', '图片', '类别', '置信度', ' bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', '面积比例'])

            for error_type in ['false_positives', 'false_negatives', 'small_objects']:
                for error in self.results['error_analysis'][error_type]:
                    writer.writerow([
                        error_type,
                        error['image'],
                        error['class_name'],
                        error.get('confidence', ''),
                        error['bbox'][0],
                        error['bbox'][1],
                        error['bbox'][2],
                        error['bbox'][3],
                        error.get('area_ratio', '')
                    ])

        log_info(f"评估报告已保存:")
        log_info(f"  JSON报告: {json_path}")
        log_info(f"  CSV指标: {csv_path}")
        log_info(f"  CSV错误: {error_csv_path}")

        return {
            'json': str(json_path),
            'csv': str(csv_path),
            'error_csv': str(error_csv_path)
        }

    def print_final_metrics(self):
        """打印最终指标"""
        metrics = self.results['metrics']
        log_info("\n" + "=" * 60)
        log_info("最终评估结果")
        log_info("=" * 60)
        log_info(f"Precision (精确率): {metrics.get('Precision', 0):.4f}")
        log_info(f"Recall (召回率): {metrics.get('Recall', 0):.4f}")
        log_info(f"mAP50: {metrics.get('mAP50', 0):.4f}")
        log_info(f"mAP50-95: {metrics.get('mAP50-95', 0):.4f}")
        log_info("=" * 60)

        log_info("\n每类别AP50:")
        for cls, cls_metrics in self.results['per_class_metrics'].items():
            log_info(f"  {cls}: {cls_metrics.get('AP50', 0):.4f}")

        log_info("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLO模型评估脚本")
    parser.add_argument("--model", type=str, required=True, help="模型文件路径")
    parser.add_argument("--data", type=str, required=True, help="数据集配置文件路径")
    parser.add_argument("--split", type=str, default='val', choices=['train', 'val', 'test'], help="评估数据集")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5, help="IOU阈值")
    parser.add_argument("--analyze_errors", action="store_true", help="执行错误分析")
    parser.add_argument("--output", type=str, default=None, help="输出报告名称")

    args = parser.parse_args()

    # 创建评估器
    evaluator = ModelEvaluator(model_path=args.model, data_yaml=args.data)

    # 加载模型
    if not evaluator.load_model():
        exit(1)

    # 运行评估
    metrics = evaluator.evaluate(split=args.split, conf_threshold=args.conf, iou_threshold=args.iou)

    if metrics is None:
        log_error("评估失败")
        exit(1)

    # 错误分析
    if args.analyze_errors:
        evaluator.analyze_errors(split=args.split, conf_threshold=args.conf)

    # 生成报告
    report_paths = evaluator.generate_report(output_name=args.output)

    # 打印最终指标
    evaluator.print_final_metrics()

    log_info("\n评估完成！报告已保存。")
