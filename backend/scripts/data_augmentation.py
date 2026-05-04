import argparse
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import yaml
import albumentations as A


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


def load_yolo_labels(label_path: Path):
    """
    读取YOLO格式的标注文件
    
    参数:
        label_path: 标注文件路径
        
    返回:
        bboxes: 边界框列表，格式为 [class_id, x, y, w, h] (归一化坐标)
    """
    bboxes = []
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    bboxes.append([class_id, x, y, w, h])
    except Exception as e:
        log_error(f"读取标注文件失败 {label_path}: {str(e)}")
    return bboxes


def save_yolo_labels(label_path: Path, bboxes):
    """
    保存YOLO格式的标注文件
    
    参数:
        label_path: 保存路径
        bboxes: 边界框列表
    """
    try:
        with open(label_path, "w", encoding="utf-8") as f:
            for bbox in bboxes:
                class_id, x, y, w, h = bbox
                f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    except Exception as e:
        log_error(f"保存标注文件失败 {label_path}: {str(e)}")


def get_class_distribution(data_dir: Path, splits: list = None):
    """
    统计各类的样本分布
    
    参数:
        data_dir: 数据集根目录
        splits: 要统计的子集列表
        
    返回:
        各类别的样本数量统计
    """
    if splits is None:
        splits = ["train", "valid", "test"]
    
    class_counts = {}
    
    for split in splits:
        # 支持两种路径结构
        labels_dir = data_dir / split / "labels"
        if not labels_dir.exists():
            labels_dir = data_dir / "labels" / split
        
        if not labels_dir.exists():
            continue
            
        for label_file in labels_dir.glob("*.txt"):
            bboxes = load_yolo_labels(label_file)
            for bbox in bboxes:
                class_id = bbox[0]
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    return class_counts


def find_minority_classes(class_counts: dict, threshold_ratio: float = 0.3):
    """
    找出少数类
    
    参数:
        class_counts: 各类别样本数量
        threshold_ratio: 少数类阈值比例
        
    返回:
        少数类ID列表
    """
    if not class_counts:
        return []
    
    total_samples = sum(class_counts.values())
    avg_samples = total_samples / len(class_counts)
    
    minority_classes = []
    for class_id, count in class_counts.items():
        if count < avg_samples * threshold_ratio:
            minority_classes.append(class_id)
    
    return minority_classes


def augment_dataset(
    data_dir: str,
    target_class: str = None,
    num_augmentations: int = 1,
    minority_aug_ratio: float = 2.0,
    apply_random_crop: bool = True
):
    """
    对包含目标类别的图片进行数据增强
    
    参数:
        data_dir: 数据集根目录
        target_class: 要增强的目标类别名称（None表示增强所有少数类）
        num_augmentations: 每张图片生成多少个增强样本
        minority_aug_ratio: 少数类增强比例（比正常类多增强的倍数）
        apply_random_crop: 是否应用随机裁剪
    """
    data_dir = Path(data_dir)
    
    # 1. 加载 data.yaml 并获取类别信息
    yaml_path = data_dir / "data.yaml"
    class_names_dict = {}
    class_names_list = []
    target_class_id = -1
    
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if "names" in data:
                names_val = data["names"]
                # 兼容字典和列表两种格式
                if isinstance(names_val, dict):
                    class_names_dict = {int(k): v for k, v in names_val.items()}
                    class_names_list = [class_names_dict[i] for i in sorted(class_names_dict.keys())]
                elif isinstance(names_val, list):
                    class_names_list = [str(name) for name in names_val]
                    class_names_dict = {i: name for i, name in enumerate(class_names_list)}
                
                if target_class:
                    if target_class in class_names_list:
                        target_class_id = class_names_list.index(target_class)
                        log_info(f"目标类别 '{target_class}' 的ID: {target_class_id}")
                    else:
                        log_error(f"目标类别 '{target_class}' 不存在于数据集类别列表中")
                        return
                    
                # 统计类别分布
                log_info("=" * 50)
                log_info("数据集类别分布统计：")
                class_counts = get_class_distribution(data_dir)
                for class_id, count in sorted(class_counts.items()):
                    class_name = class_names_dict.get(class_id, f"class_{class_id}")
                    log_info(f"  {class_name} (ID:{class_id}): {count} 样本")
                
                # 找出少数类
                minority_classes = find_minority_classes(class_counts)
                if minority_classes and not target_class:
                    log_info(f"\n发现的少数类: {[class_names_dict.get(cid, f'class_{cid}') for cid in minority_classes]}")
                log_info("=" * 50)
                
    except Exception as e:
        log_error(f"读取 data.yaml 失败: {str(e)}")
        return
    
    # 2. 定义基础数据增强流水线
    # 包含：随机旋转、水平翻转、亮度/对比度调整、噪声扰动、模糊处理
    base_augmentation = A.Compose([
        A.HorizontalFlip(p=0.5),  # 水平翻转
        A.RandomRotate90(p=0.5),   # 随机90度旋转
        A.RandomBrightnessContrast(  # 亮度/对比度调整
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.3
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),  # 高斯噪声
        A.Blur(blur_limit=3, p=0.2),  # 模糊处理
    ], bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_ids"],
        min_visibility=0.3
    ))
    
    # 3. 定义带随机裁剪的增强流水线
    crop_augmentation = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.3
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Blur(blur_limit=3, p=0.2),
        A.RandomCropFromBorders(crop_height=0.7, crop_width=0.7, p=0.3),  # 随机裁剪
    ], bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_ids"],
        min_visibility=0.3
    ))
    
    # 4. 查找要增强的图片（支持两种路径结构）
    # 结构1: data_dir/train/images, data_dir/train/labels
    # 结构2: data_dir/images/train, data_dir/labels/train
    train_images_dir = None
    train_labels_dir = None
    
    if (data_dir / "train" / "images").exists():
        train_images_dir = data_dir / "train" / "images"
        train_labels_dir = data_dir / "train" / "labels"
    elif (data_dir / "images" / "train").exists():
        train_images_dir = data_dir / "images" / "train"
        train_labels_dir = data_dir / "labels" / "train"
    
    if not train_images_dir or not train_labels_dir:
        log_error("训练集目录不存在")
        return
    
    # 收集要增强的图片
    target_images = []
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    
    for image_path in train_images_dir.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in image_extensions:
            label_path = train_labels_dir / (image_path.stem + ".txt")
            if label_path.exists():
                bboxes = load_yolo_labels(label_path)
                if not bboxes:
                    continue
                
                # 如果指定了目标类别，只增强包含目标类的图片
                if target_class_id >= 0:
                    has_target = any(bbox[0] == target_class_id for bbox in bboxes)
                    if has_target:
                        target_images.append((image_path, label_path, True))  # True表示是目标类
                else:
                    # 检查是否包含少数类
                    class_ids = [bbox[0] for bbox in bboxes]
                    minority_in_image = any(cid in minority_classes for cid in class_ids)
                    target_images.append((image_path, label_path, minority_in_image))
    
    log_info(f"找到 {len(target_images)} 张需要增强的图片")
    
    # 5. 对每张图片进行增强
    processed_count = 0
    generated_samples = 0
    target_processed = 0
    minority_processed = 0
    
    for image_path, label_path, is_target_class in target_images:
        try:
            # 读取图片
            img = cv2.imread(str(image_path))
            if img is None:
                log_error(f"无法读取图片: {image_path}")
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 读取标注
            bboxes = load_yolo_labels(label_path)
            if not bboxes:
                continue
            
            # 分离 class_id 和 bbox 坐标
            class_ids = [bbox[0] for bbox in bboxes]
            bbox_coords = [bbox[1:] for bbox in bboxes]
            
            # 确定增强次数：少数类/目标类使用更高的增强比例
            aug_count = num_augmentations
            if is_target_class:
                aug_count = max(num_augmentations, int(num_augmentations * minority_aug_ratio))
                target_processed += 1
            else:
                minority_processed += 1
            
            # 交替使用两种增强流水线
            for aug_idx in range(aug_count):
                # 选择增强流水线
                if apply_random_crop and aug_idx % 2 == 1:
                    augmentation_pipeline = crop_augmentation
                else:
                    augmentation_pipeline = base_augmentation
                
                try:
                    # 应用增强
                    augmented = augmentation_pipeline(
                        image=img,
                        bboxes=bbox_coords,
                        class_ids=class_ids
                    )
                    
                    aug_img = augmented["image"]
                    aug_bboxes = augmented["bboxes"]
                    aug_class_ids = augmented["class_ids"]
                    
                    # 过滤掉被裁剪掉的bbox
                    valid_bboxes = []
                    for cls_id, bbox in zip(aug_class_ids, aug_bboxes):
                        if bbox and len(bbox) == 4:
                            valid_bboxes.append([cls_id] + list(bbox))
                    
                    if not valid_bboxes:
                        continue
                    
                    # 组合新的标注
                    new_bboxes = valid_bboxes
                    
                    # 生成新文件名
                    aug_suffix = "_aug" + str(aug_idx + 1)
                    new_stem = f"{image_path.stem}{aug_suffix}"
                    new_image_path = train_images_dir / (new_stem + image_path.suffix)
                    new_label_path = train_labels_dir / (new_stem + ".txt")
                    
                    # 保存增强后的图片
                    aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(new_image_path), aug_img_bgr)
                    
                    # 保存增强后的标注
                    save_yolo_labels(new_label_path, new_bboxes)
                    
                    generated_samples += 1
                    
                except Exception as e:
                    log_warning(f"增强失败 {image_path.name}[{aug_idx}]: {str(e)}")
                    continue
            
            processed_count += 1
            
            if processed_count % 50 == 0:
                log_info(f"已处理 {processed_count}/{len(target_images)} 张图片...")
            
        except Exception as e:
            log_error(f"处理图片失败 {image_path.name}: {str(e)}")
            continue
    
    # 6. 输出统计信息
    log_info("=" * 50)
    log_info("数据增强完成！")
    log_info(f"处理图片总数: {processed_count}")
    log_info(f"  - 目标类/少数类图片: {target_processed}")
    log_info(f"  - 普通类图片: {minority_processed}")
    log_info(f"生成新样本总数: {generated_samples}")
    if target_class:
        log_info(f"目标类别: {target_class}")
    else:
        log_info("增强策略: 所有少数类")
    log_info(f"增强比例: 少数类 {minority_aug_ratio}x")
    log_info(f"启用随机裁剪: {apply_random_crop}")
    log_info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据集少数类数据增强脚本")
    parser.add_argument("--data_dir", required=True, help="数据集根目录路径")
    parser.add_argument("--target_class", default=None, help="要增强的目标类别名称（可选）")
    parser.add_argument("--num_aug", type=int, default=1, help="每张图片生成多少个增强样本")
    parser.add_argument("--minority_ratio", type=float, default=2.0, help="少数类增强倍数")
    parser.add_argument("--no_crop", action="store_true", help="禁用随机裁剪")
    
    args = parser.parse_args()
    
    log_info("开始执行数据增强脚本...")
    log_info(f"数据集路径: {args.data_dir}")
    log_info(f"目标类别: {args.target_class or '所有少数类'}")
    log_info(f"每图生成样本数: {args.num_aug}")
    log_info(f"少数类增强倍数: {args.minority_ratio}")
    log_info(f"启用随机裁剪: {not args.no_crop}")
    
    augment_dataset(
        args.data_dir,
        args.target_class,
        args.num_aug,
        args.minority_ratio,
        not args.no_crop
    )
