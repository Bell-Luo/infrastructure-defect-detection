from ultralytics import YOLO
import os
import sys
from pathlib import Path
from datetime import datetime
import json
import cv2
import numpy as np
from collections import defaultdict
import yaml

def log_info(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] INFO: {message}", flush=True)

def log_error(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] ERROR: {message}", flush=True)

def log_warning(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] WARNING: {message}", flush=True)


class ScenarioErrorAnalyzer:
    """场景化错误分析器"""

    def __init__(self, model_path: str, data_yaml: str):
        """
        初始化分析器

        参数:
            model_path: 模型文件路径
            data_yaml: 数据集配置文件路径
        """
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.model = None
        self.class_names = []
        self.dataset_path = None

        # 分析结果
        self.analysis_results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": model_path,
            "scenarios": {
                "small_cracks": {
                    "description": "细小裂缝检测",
                    "characteristics": "缺陷面积占图片比例 < 1%",
                    "errors": [],
                    "statistics": {}
                },
                "weak_texture_seepage": {
                    "description": "弱纹理渗水检测",
                    "characteristics": "边缘模糊、纹理不明显的渗水区域",
                    "errors": [],
                    "statistics": {}
                },
                "complex_background_spalling": {
                    "description": "复杂背景剥落检测",
                    "characteristics": "背景纹理复杂，与剥落区域对比度低",
                    "errors": [],
                    "statistics": {}
                }
            }
        }

        # 输出目录
        self.output_dir = Path("error_analysis_results")
        self.output_dir.mkdir(exist_ok=True)

        log_info("场景化错误分析器初始化完成")

    def load_model_and_data(self):
        """加载模型和数据集配置"""
        if not os.path.exists(self.model_path):
            log_error(f"模型文件不存在: {self.model_path}")
            return False

        try:
            self.model = YOLO(self.model_path)
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
                if isinstance(self.class_names, dict):
                    self.class_names = [self.class_names[i] for i in sorted(self.class_names.keys())]

            # 读取数据集配置
            with open(self.data_yaml, 'r', encoding='utf-8') as f:
                data_config = yaml.safe_load(f)
                self.dataset_path = Path(data_config.get('path', '.'))

            log_info(f"模型类别: {self.class_names}")
            log_info(f"数据集路径: {self.dataset_path}")
            return True

        except Exception as e:
            log_error(f"加载失败: {e}")
            return False

    def analyze_scenarios(self, split: str = 'val', conf_threshold: float = 0.25):
        """
        分析各种场景下的错误

        参数:
            split: 数据集划分
            conf_threshold: 置信度阈值
        """
        if self.model is None:
            if not self.load_model_and_data():
                return None

        log_info("=" * 60)
        log_info("开始场景化错误分析")
        log_info("=" * 60)

        # 获取数据集路径
        images_dir = self.dataset_path / split / 'images'
        labels_dir = self.dataset_path / split / 'labels'

        if not images_dir.exists():
            log_error(f"图片目录不存在: {images_dir}")
            return None

        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        log_info(f"分析图片数量: {len(image_files)}")

        # 分类统计
        stats = {
            "total_images": len(image_files),
            "total_gt_boxes": 0,
            "total_pred_boxes": 0,
            "small_object_errors": 0,
            "weak_texture_errors": 0,
            "complex_background_errors": 0
        }

        for img_path in image_files:
            try:
                self._analyze_single_image(img_path, labels_dir, conf_threshold, stats)
            except Exception as e:
                log_warning(f"处理图片失败 {img_path.name}: {str(e)}")
                continue

        # 计算统计数据
        self._compute_statistics()

        # 打印分析结果
        self._print_analysis_results()

        return self.analysis_results

    def _analyze_single_image(self, img_path, labels_dir, conf_threshold, stats):
        """分析单张图片"""
        img = cv2.imread(str(img_path))
        if img is None:
            return

        h, w = img.shape[:2]
        img_area = w * h

        # 读取真实标注
        label_path = labels_dir / (img_path.stem + ".txt")
        gt_boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x, y, bw, bh = map(float, parts[1:5])
                        x1 = int((x - bw/2) * w)
                        y1 = int((y - bh/2) * h)
                        x2 = int((x + bw/2) * w)
                        y2 = int((y + bh/2) * h)
                        area = (x2 - x1) * (y2 - y1)
                        area_ratio = area / img_area

                        gt_boxes.append({
                            'class_id': cls_id,
                            'class_name': self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}",
                            'bbox': [x1, y1, x2, y2],
                            'area': area,
                            'area_ratio': area_ratio
                        })

        stats["total_gt_boxes"] += len(gt_boxes)

        # 获取预测结果
        results = self.model.predict(str(img_path), conf=conf_threshold, verbose=False)
        pred_boxes = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    area = (x2 - x1) * (y2 - y1)
                    area_ratio = area / img_area

                    pred_boxes.append({
                        'class_id': cls_id,
                        'class_name': self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}",
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'area': area,
                        'area_ratio': area_ratio
                    })

        stats["total_pred_boxes"] += len(pred_boxes)

        # 匹配并分类错误
        matched_gt, matched_pred = self._match_boxes(gt_boxes, pred_boxes)

        # 分析每种场景的错误
        self._analyze_scenario_errors(
            img_path.name, gt_boxes, pred_boxes, matched_gt, matched_pred, img_area
        )

    def _match_boxes(self, gt_boxes, pred_boxes, iou_threshold=0.5):
        """匹配真实框和预测框"""
        matched_gt = set()
        matched_pred = set()

        for p_idx, pred in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1

            for g_idx, gt in enumerate(gt_boxes):
                if g_idx in matched_gt:
                    continue
                if gt['class_id'] != pred['class_id']:
                    continue

                iou = self._compute_iou(pred['bbox'], gt['bbox'])
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                matched_pred.add(p_idx)

        return matched_gt, matched_pred

    def _compute_iou(self, box1, box2):
        """计算IOU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0

        inter_area = (x2 - x1) * (y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def _analyze_scenario_errors(self, img_name, gt_boxes, pred_boxes, matched_gt, matched_pred, img_area):
        """分析各场景的错误"""
        # 1. 细小裂缝分析 (面积比例 < 1%)
        small_area_threshold = 0.01 * img_area

        for g_idx, gt in enumerate(gt_boxes):
            if g_idx in matched_gt:
                continue

            # 判断是否为小目标漏检
            if gt['area'] < small_area_threshold:
                error = {
                    'image': img_name,
                    'class_name': gt['class_name'],
                    'bbox': gt['bbox'],
                    'area_ratio': gt['area_ratio'],
                    'error_type': '漏检'
                }
                self.analysis_results['scenarios']['small_cracks']['errors'].append(error)

        # 2. 误检分析 (小目标和弱纹理)
        for p_idx, pred in enumerate(pred_boxes):
            if p_idx in matched_pred:
                continue

            # 小目标误检
            if pred['area'] < small_area_threshold:
                error = {
                    'image': img_name,
                    'class_name': pred['class_name'],
                    'confidence': pred['confidence'],
                    'bbox': [int(x) for x in pred['bbox']],
                    'area_ratio': pred['area_ratio'],
                    'error_type': '误检'
                }
                self.analysis_results['scenarios']['small_cracks']['errors'].append(error)

        # 3. 弱纹理和复杂背景分析（通过边缘信息和纹理特征判断）
        self._analyze_texture_and_background(
            img_name, gt_boxes, pred_boxes, matched_gt, matched_pred, img_area
        )

    def _analyze_texture_and_background(self, img_name, gt_boxes, pred_boxes, matched_gt, matched_pred, img_area):
        """分析弱纹理和复杂背景场景"""
        # 读取图片获取纹理信息
        img_path = self.dataset_path / 'val' / 'images' / img_name
        if not img_path.exists():
            return

        img = cv2.imread(str(img_path))
        if img is None:
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 分析未匹配的漏检框
        for g_idx, gt in enumerate(gt_boxes):
            if g_idx in matched_gt:
                continue

            bbox = gt['bbox']
            x1, y1, x2, y2 = bbox

            # 提取ROI
            roi = gray[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else gray

            if roi.size == 0:
                continue

            # 计算纹理特征（边缘密度）
            edges = cv2.Canny(roi, 50, 150)
            edge_density = np.sum(edges > 0) / roi.size

            # 低边缘密度 -> 可能是弱纹理
            if edge_density < 0.1:
                error = {
                    'image': img_name,
                    'class_name': gt['class_name'],
                    'bbox': bbox,
                    'edge_density': float(edge_density),
                    'error_type': '漏检'
                }
                self.analysis_results['scenarios']['weak_texture_seepage']['errors'].append(error)

        # 分析未匹配的误检框
        for p_idx, pred in enumerate(pred_boxes):
            if p_idx in matched_pred:
                continue

            bbox = pred['bbox']
            x1, y1, x2, y2 = map(int, bbox)

            roi = gray[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else gray

            if roi.size == 0:
                continue

            edges = cv2.Canny(roi, 50, 150)
            edge_density = np.sum(edges > 0) / roi.size

            # 低边缘密度误检
            if edge_density < 0.1:
                error = {
                    'image': img_name,
                    'class_name': pred['class_name'],
                    'confidence': pred['confidence'],
                    'bbox': [x1, y1, x2, y2],
                    'edge_density': float(edge_density),
                    'error_type': '误检'
                }
                self.analysis_results['scenarios']['weak_texture_seepage']['errors'].append(error)

    def _compute_statistics(self):
        """计算统计数据"""
        scenarios = self.analysis_results['scenarios']

        # 细小裂缝统计
        small_errors = scenarios['small_cracks']['errors']
        fp_small = [e for e in small_errors if e.get('error_type') == '误检']
        fn_small = [e for e in small_errors if e.get('error_type') == '漏检']
        scenarios['small_cracks']['statistics'] = {
            'total_errors': len(small_errors),
            'false_positives': len(fp_small),
            'false_negatives': len(fn_small),
            'fp_rate': len(fp_small) / self.analysis_results['scenarios'].get('total_pred', 1) if self.analysis_results['scenarios'].get('total_pred', 0) > 0 else 0,
            'fn_rate': len(fn_small) / self.analysis_results['scenarios'].get('total_gt', 1) if self.analysis_results['scenarios'].get('total_gt', 0) > 0 else 0
        }

        # 弱纹理渗水统计
        weak_errors = scenarios['weak_texture_seepage']['errors']
        fp_weak = [e for e in weak_errors if e.get('error_type') == '误检']
        fn_weak = [e for e in weak_errors if e.get('error_type') == '漏检']
        scenarios['weak_texture_seepage']['statistics'] = {
            'total_errors': len(weak_errors),
            'false_positives': len(fp_weak),
            'false_negatives': len(fn_weak)
        }

        # 复杂背景剥落统计
        complex_errors = scenarios['complex_background_spalling']['errors']
        scenarios['complex_background_spalling']['statistics'] = {
            'total_errors': len(complex_errors),
            'false_positives': len([e for e in complex_errors if e.get('error_type') == '误检']),
            'false_negatives': len([e for e in complex_errors if e.get('error_type') == '漏检'])
        }

    def _print_analysis_results(self):
        """打印分析结果"""
        log_info("\n" + "=" * 60)
        log_info("场景化错误分析结果")
        log_info("=" * 60)

        for scenario_key, scenario in self.analysis_results['scenarios'].items():
            log_info(f"\n【{scenario['description']}】")
            log_info(f"  特征: {scenario['characteristics']}")
            stats = scenario['statistics']
            log_info(f"  总错误数: {stats.get('total_errors', 0)}")
            log_info(f"  误检数(FP): {stats.get('false_positives', 0)}")
            log_info(f"  漏检数(FN): {stats.get('false_negatives', 0)}")

        log_info("\n" + "=" * 60)

    def generate_report(self, output_name: str = None):
        """生成分析报告"""
        if output_name is None:
            output_name = f"scenario_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 保存JSON报告
        json_path = self.output_dir / f"{output_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False, default=str)

        log_info(f"\n分析报告已保存: {json_path}")
        return str(json_path)


def generate_recommendations(analysis_results):
    """根据分析结果生成改进建议"""
    recommendations = []

    small_stats = analysis_results['scenarios']['small_cracks']['statistics']
    weak_stats = analysis_results['scenarios']['weak_texture_seepage']['statistics']
    complex_stats = analysis_results['scenarios']['complex_background_spalling']['statistics']

    # 细小裂缝建议
    if small_stats.get('false_negatives', 0) > 0:
        recommendations.append({
            'scenario': '细小裂缝漏检',
            'suggestions': [
                '提高图像分辨率，使用更大输入尺寸(如1280)',
                '使用更小的检测anchor尺寸',
                '添加专门的小目标检测层(FPN/PAN改进)',
                '使用TTA(Test Time Augmentation)增加小目标检出率',
                '针对小目标使用更低的置信度阈值(如0.1)'
            ]
        })

    if small_stats.get('false_positives', 0) > 0:
        recommendations.append({
            'scenario': '细小裂缝误检',
            'suggestions': [
                '提高置信度阈值',
                '添加负样本训练数据',
                '使用更严格的NMS阈值'
            ]
        })

    # 弱纹理渗水建议
    if weak_stats.get('total_errors', 0) > 10:
        recommendations.append({
            'scenario': '弱纹理渗水检测困难',
            'suggestions': [
                '使用边缘保持滤波器增强渗水边界',
                '添加渗水特有的颜色特征(如蓝绿色调)',
                '使用注意力机制增强模型对弱纹理区域的感知',
                '增加渗水场景的训练样本',
                '使用分割任务代替检测任务，提高边缘精度'
            ]
        })

    # 复杂背景剥落建议
    if complex_stats.get('total_errors', 0) > 10:
        recommendations.append({
            'scenario': '复杂背景剥落检测困难',
            'suggestions': [
                '使用更深的backbone提取多尺度特征',
                '添加上下文模块(context module)利用周围区域信息',
                '使用对抗训练增强模型在复杂背景下的鲁棒性',
                '考虑使用分割+分类的级联方法',
                '增加数据增强，如随机噪声、模糊等模拟复杂场景'
            ]
        })

    return recommendations


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="场景化错误分析脚本")
    parser.add_argument("--model", type=str, required=True, help="模型文件路径")
    parser.add_argument("--data", type=str, required=True, help="数据集配置文件路径")
    parser.add_argument("--split", type=str, default='val', help="分析数据集")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--output", type=str, default=None, help="输出报告名称")

    args = parser.parse_args()

    # 创建分析器
    analyzer = ScenarioErrorAnalyzer(model_path=args.model, data_yaml=args.data)

    # 加载模型和数据
    if not analyzer.load_model_and_data():
        log_error("加载失败")
        exit(1)

    # 执行分析
    results = analyzer.analyze_scenarios(split=args.split, conf_threshold=args.conf)

    if results is None:
        log_error("分析失败")
        exit(1)

    # 生成报告
    report_path = analyzer.generate_report(output_name=args.output)

    # 生成改进建议
    recommendations = generate_recommendations(results)

    log_info("\n" + "=" * 60)
    log_info("改进建议")
    log_info("=" * 60)
    for rec in recommendations:
        log_info(f"\n【{rec['scenario']}】")
        for i, suggestion in enumerate(rec['suggestions'], 1):
            log_info(f"  {i}. {suggestion}")

    log_info("\n" + "=" * 60)
    log_info("分析完成！")
