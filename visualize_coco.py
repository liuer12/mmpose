import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def visualize_coco_dataset(coco_ann_file, image_dir, output_dir, num_images=10):
    """
    可视化COCO数据集
    
    Args:
        coco_ann_file: COCO标注文件路径
        image_dir: 图像目录路径
        output_dir: 输出目录路径
        num_images: 要可视化的图像数量
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取COCO标注文件
    with open(coco_ann_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 获取类别信息
    categories = {cat['id']: cat for cat in coco_data['categories']}
    
    # 按图像ID组织标注
    image_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    # 随机选择图像进行可视化
    image_ids = list(image_annotations.keys())
    np.random.shuffle(image_ids)
    selected_images = image_ids[:num_images]
    
    for image_id in tqdm(selected_images):
        # 获取图像信息
        image_info = next(img for img in coco_data['images'] if img['id'] == image_id)
        image_path = os.path.join(image_dir, image_info['file_name'])
        
        # 读取图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 创建图像副本用于绘制
        vis_image = image.copy()
        
        # 获取该图像的所有标注
        annotations = image_annotations[image_id]
        
        # 绘制每个实例的边界框和关键点
        for ann in annotations:
            # 获取边界框
            bbox = ann['bbox']
            x, y, w, h = [int(v) for v in bbox]
            
            # 绘制边界框
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 获取关键点
            keypoints = ann['keypoints']
            num_keypoints = len(keypoints) // 3
            
            # 绘制关键点
            for i in range(num_keypoints):
                x, y, v = keypoints[i*3:(i+1)*3]
                if v > 0:  # 只绘制可见的关键点
                    # 绘制关键点
                    cv2.circle(vis_image, (int(x), int(y)), 4, (255, 0, 0), -1)
        
        # 保存可视化结果
        output_path = os.path.join(output_dir, f'vis_{image_info["file_name"]}')
        plt.figure(figsize=(15, 15))
        plt.imshow(vis_image)
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

if __name__ == '__main__':
    coco_ann_file = 'coco_dataset/Small/annotations.json'
    image_dir = 'coco_dataset/Small/images'
    output_dir = 'coco_dataset/Small/visualization'
    visualize_coco_dataset(coco_ann_file, image_dir, output_dir, num_images=10) 