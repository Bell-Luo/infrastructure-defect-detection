from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import yaml
from datetime import datetime


def log_info(message):
    """打印带时间戳的信息日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] INFO: {message}")


def log_error(message):
    """打印带时间戳的错误日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] ERROR: {message}")


def log_warning(message):
    """打印带时间戳的警告日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] WARNING: {message}")


class DatasetManager:
    """数据集管理类，用于处理、校验和解析YOLO格式数据集"""
    
    def __init__(self, data_dir: str):
        """
        构造函数，初始化数据集管理器
        
        参数:
            data_dir: 数据集根目录路径
        """
        self.data_dir = Path(data_dir)
        self.data_yaml = None
        self.class_names = []
        
        log_info(f"正在初始化数据集管理器，目录: {self.data_dir}")
        
        # 读取并解析 data.yaml
        yaml_path = self.data_dir / "data.yaml"
        if yaml_path.exists():
            try:
                with open(yaml_path, "r", encoding="utf-8") as f:
                    self.data_yaml = yaml.safe_load(f)
                # 获取类别名称
                if "names" in self.data_yaml:
                    names_val = self.data_yaml["names"]
                    # 确保 class_names 总是数组
                    if isinstance(names_val, dict):
                        # 如果是字典格式 {0: 'class1', 1: 'class2'}
                        self.class_names = [str(names_val[i]) for i in sorted(names_val.keys())]
                    elif isinstance(names_val, list):
                        self.class_names = [str(name) for name in names_val]
                    else:
                        self.class_names = []
                    log_info(f"成功读取类别列表: {self.class_names}")
                else:
                    log_warning("data.yaml 中未找到 'names' 字段")
            except Exception as e:
                log_error(f"解析 data.yaml 失败: {str(e)}")
        else:
            log_error(f"未找到 data.yaml 文件: {yaml_path}")
    
    def check_dataset_integrity(self, split: str = "train") -> Dict[str, Any]:
        """
        检查数据集完整性
        
        参数:
            split: 要检查的子集名称，如 'train', 'val', 'test'
            
        返回:
            包含检查结果的字典
        """
        log_info(f"开始检查 {split} 数据集完整性...")
        
        # 初始化结果字典
        result = {
            "split": split,
            "total_images": 0,
            "corrupted_images": [],
            "missing_labels": [],
            "empty_labels": [],
            "invalid_labels": []
        }
        
        # 定义路径 - 根据实际结构：data_dir/split/images 和 data_dir/split/labels
        images_dir = self.data_dir / split / "images"
        labels_dir = self.data_dir / split / "labels"
        
        if not images_dir.exists():
            log_error(f"图片目录不存在: {images_dir}")
            return result
        
        # 遍历所有图片文件
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        for image_path in images_dir.iterdir():
            if image_path.is_file() and image_path.suffix.lower() in image_extensions:
                result["total_images"] += 1
                
                # 1. 检查图片是否损坏
                try:
                    with Image.open(image_path) as img:
                        img.verify()
                except Exception as e:
                    log_error(f"图片损坏: {image_path.name} - {str(e)}")
                    result["corrupted_images"].append(image_path.name)
                    continue  # 损坏图片跳过后续检查
                
                # 2. 检查是否有对应的标注文件
                label_path = labels_dir / (image_path.stem + ".txt")
                if not label_path.exists():
                    result["missing_labels"].append(image_path.name)
                    continue
                
                # 3. 检查标注文件内容
                try:
                    with open(label_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    
                    if not content:
                        result["empty_labels"].append(image_path.name)
                    else:
                        # 检查每一行标注的格式
                        lines = content.split("\n")
                        valid = True
                        for line in lines:
                            parts = line.strip().split()
                            if not parts:
                                continue
                            if len(parts) < 5:
                                valid = False
                                break
                            # 尝试解析坐标，检查是否为有效浮点数
                            try:
                                cls_id = int(parts[0])
                                x = float(parts[1])
                                y = float(parts[2])
                                w = float(parts[3])
                                h = float(parts[4])
                                # 检查坐标是否在 [0, 1] 范围内
                                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                                    valid = False
                                    break
                            except ValueError:
                                valid = False
                                break
                        
                        if not valid:
                            result["invalid_labels"].append(image_path.name)
                except Exception as e:
                    log_error(f"读取标注文件失败 {label_path.name}: {str(e)}")
                    result["invalid_labels"].append(image_path.name)
        
        log_info(f"检查完成！总图片数: {result['total_images']}, "
                 f"损坏: {len(result['corrupted_images'])}, "
                 f"缺失标注: {len(result['missing_labels'])}, "
                 f"空标注: {len(result['empty_labels'])}, "
                 f"无效标注: {len(result['invalid_labels'])}")
        
        return result
    
    def list_classes(self) -> List[str]:
        """
        返回所有类别名称的列表
        
        返回:
            类别名称列表
        """
        return self.class_names
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """
        返回数据集的综合信息，供前端API使用
        
        返回:
            包含数据集信息的字典
        """
        log_info("正在生成数据集综合信息...")
        
        summary = {
            "dataset_path": str(self.data_dir),
            "class_names": self.class_names,
            "class_count": len(self.class_names),
            "splits": {},
            "data_yaml": self.data_yaml
        }
        
        # 检查常见的子集，包括 valid
        splits_to_check = ["train", "valid", "test"]
        for split in splits_to_check:
            images_dir = self.data_dir / split / "images"
            if images_dir.exists():
                # 统计图片数量
                image_count = 0
                image_extensions = {"*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"}
                for ext in image_extensions:
                    image_count += len(list(images_dir.glob(ext)))
                # 检查完整性
                integrity = self.check_dataset_integrity(split)
                summary["splits"][split] = {
                    "image_count": image_count,
                    "integrity": integrity
                }
        
        log_info("数据集信息生成完成")
        return summary
