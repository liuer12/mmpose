import os
import json
import numpy as np
from tqdm import tqdm
import shutil
from pathlib import Path

def create_image_info(image_id, file_name, image_size):
    """创建图像信息"""
    image_info = {
        'id': image_id,
        'file_name': file_name,
        'width': image_size[0],
        'height': image_size[1],
        'date_captured': '',
        'license': 0,
        'coco_url': '',
        'flickr_url': ''
    }
    return image_info

def create_annotation_info(annotation_id, image_id, category_id, keypoints, num_keypoints, bbox):
    """创建标注信息"""
    annotation_info = {
        'id': annotation_id,
        'image_id': image_id,
        'category_id': category_id,
        'keypoints': keypoints,
        'num_keypoints': num_keypoints,
        'bbox': bbox,
        'area': bbox[2] * bbox[3],
        'iscrowd': 0
    }
    return annotation_info

def convert_labelme_to_coco(labelme_dir, output_dir):
    """将labelme格式转换为COCO格式"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    
    # 初始化COCO格式数据
    coco_output = {
        'images': [],
        'annotations': [],
        'categories': [{
            'id': 1,
            'name': 'strawberry',
            'supercategory': 'strawberry',
            'keypoints': ['KP1', 'KP2', 'KP3', 'KP4', 'KP5', 'KP6'],
            'skeleton': [[1, 2], [2, 3], [4, 5], [5, 6]]
        }]
    }
    
    # 获取所有标注文件
    labelme_files = [f for f in os.listdir(labelme_dir) if f.endswith('.json')]
    
    image_id = 1
    annotation_id = 1
    
    for labelme_file in tqdm(labelme_files):
        # 读取labelme标注文件
        with open(os.path.join(labelme_dir, labelme_file), 'r', encoding='utf-8') as f:
            labelme_data = json.load(f)
        
        # 获取图像文件名
        image_file = labelme_file.replace('.json', '.jpg')
        if not os.path.exists(os.path.join(labelme_dir, '..', 'image1', image_file)):
            image_file = labelme_file.replace('.json', '.png')
        
        # 复制图像文件到输出目录
        src_image_path = os.path.join(labelme_dir, '..', 'image1', image_file)
        dst_image_path = os.path.join(output_dir, 'images', image_file)
        shutil.copy2(src_image_path, dst_image_path)
        
        # 获取图像尺寸
        image_size = (labelme_data.get('imageWidth', 0), labelme_data.get('imageHeight', 0))
        
        # 添加图像信息
        image_info = create_image_info(image_id, image_file, image_size)
        coco_output['images'].append(image_info)
        
        # 处理标注信息
        shapes = labelme_data['shapes']
        current_group = None
        keypoints = [None] * 6  # 初始化6个关键点位置为None
        keypoints_visibility = [0] * 6  # 初始化6个关键点的可见性为0
        bbox = None
        
        for shape in shapes:
            if shape['shape_type'] == 'rectangle':
                if current_group is not None and any(keypoints):
                    # 创建关键点标注
                    keypoints_flat = []
                    for i, kp in enumerate(keypoints):
                        if kp is not None:
                            keypoints_flat.extend([kp[0], kp[1], keypoints_visibility[i]])
                        else:
                            keypoints_flat.extend([0, 0, 0])
                    
                    # 计算可见关键点数量
                    num_visible = sum(keypoints_visibility)
                    
                    annotation_info = create_annotation_info(
                        annotation_id, image_id, 1, keypoints_flat, num_visible, bbox)
                    coco_output['annotations'].append(annotation_info)
                    annotation_id += 1
                    keypoints = [None] * 6
                    keypoints_visibility = [0] * 6
                
                current_group = shape['group_id']
                points = shape['points']
                x1, y1 = points[0]
                x2, y2 = points[1]
                bbox = [x1, y1, x2 - x1, y2 - y1]
            
            elif shape['shape_type'] == 'point' and shape['group_id'] == current_group:
                label = shape['label']
                if label.startswith('KP'):
                    # 确定关键点索引
                    idx = int(label[-1]) - 1  # KP1 -> 0, KP2 -> 1, etc.
                    keypoints[idx] = shape['points'][0]
                    # 设置可见性：KP为1，KP-Hide为0
                    keypoints_visibility[idx] = 1 if not label.startswith('KP-Hide') else 0
        
        # 处理最后一个实例
        if any(keypoints):
            keypoints_flat = []
            for i, kp in enumerate(keypoints):
                if kp is not None:
                    keypoints_flat.extend([kp[0], kp[1], keypoints_visibility[i]])
                else:
                    keypoints_flat.extend([0, 0, 0])
            
            num_visible = sum(keypoints_visibility)
            
            annotation_info = create_annotation_info(
                annotation_id, image_id, 1, keypoints_flat, num_visible, bbox)
            coco_output['annotations'].append(annotation_info)
            annotation_id += 1
        
        image_id += 1
    
    # 保存COCO格式标注文件
    with open(os.path.join(output_dir, 'annotations.json'), 'w', encoding='utf-8') as f:
        json.dump(coco_output, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    labelme_dir = 'origina_dataset/Keypoint_ultimate/val/annotation'
    output_dir = 'coco_dataset/Ultimate/val'
    convert_labelme_to_coco(labelme_dir, output_dir) 