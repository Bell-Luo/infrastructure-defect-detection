from ultralytics import YOLO
import os
from pathlib import Path
from datetime import datetime
import shutil
import random

def log_info(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] INFO: {message}", flush=True)

def log_error(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] ERROR: {message}", flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="E:/crack-seg")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10, help="每个类别的样本数")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    data_path = Path(args.data)
    mini_dir = Path("runs/mini_dataset")
    mini_images = mini_dir / "images" / "train"
    mini_labels = mini_dir / "labels" / "train"
    mini_dir.mkdir(parents=True, exist_ok=True)
    mini_images.mkdir(parents=True, exist_ok=True)
    mini_labels.mkdir(parents=True, exist_ok=True)

    # 读取原数据集标签
    yaml_path = data_path / "train" / "labels"
    if not yaml_path.exists():
        log_error(f"标签目录不存在: {yaml_path}")
        exit(1)

    # 获取所有标签文件
    all_labels = list(yaml_path.glob("*.txt"))
    random.shuffle(all_labels)
    selected = all_labels[:args.samples]

    log_info(f"使用 {len(selected)} 张图片进行极速训练")

    for label_file in selected:
        img_name = label_file.stem
        img_exts = [".jpg", ".png", ".jpeg"]
        src_img = None
        for ext in img_exts:
            potential = data_path / "train" / "images" / (img_name + ext)
            if potential.exists():
                src_img = potential
                break

        if src_img:
            shutil.copy(src_img, mini_images / src_img.name)
            shutil.copy(label_file, mini_labels / label_file.name)

    # 创建临时yaml
    mini_yaml = mini_dir / "data.yaml"
    with open(mini_yaml, 'w', encoding='utf-8') as f:
        f.write(f"path: {mini_dir.absolute()}\ntrain: images/train\nval: images/train\nnames:\n  0: crack\n  1: spalling\n  2: seepage\n  3: damage\n  4: corrosion\n")

    log_info(f"创建临时数据集: {mini_yaml}")
    log_info(f"开始极速训练: {args.epochs} epochs, batch={args.batch}")

    model = YOLO("yolo11n.pt")
    results = model.train(
        data=str(mini_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=320,
        device=args.device,
        project="runs",
        name="quick_train",
        exist_ok=True,
        cache=False,
        workers=2,
        verbose=True
    )

    best_model = "runs/quick_train/weights/best.pt"
    if os.path.exists(best_model):
        shutil.copy(best_model, "best.pt")
        log_info(f"模型已保存到: best.pt")

    log_info("训练完成！")
