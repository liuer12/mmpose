from mmpose.apis import MMPoseInferencer
import cv2
import numpy as np

img_path = 'E:/py_project/mmpose/coco_dataset/Small/images/12.png'   # 修正为正确的图像路径

# 使用模型配置文件和权重文件的路径或 URL 构建推理器
inferencer = MMPoseInferencer(
    pose2d='configs/rtmpose-m_1xb16-200e_coco-640x480.py',
    pose2d_weights='work_dirs/rtmpose-m_1xb16-200e_coco-640x480/best_coco_AP_epoch_200.pth'
)

# # 读取原始图像
# img = cv2.imread(img_path)
# if img is None:
#     raise ValueError(f'无法读取图像: {img_path}')

# MMPoseInferencer采用了惰性推断方法，在给定输入时创建一个预测生成器
result_generator = inferencer(img_path, out_dir='output')
result = next(result_generator)

# # 保存可视化结果
# if 'visualization' in result:
#     vis_img = result['visualization']
#     if isinstance(vis_img, np.ndarray):
#         cv2.imwrite('prediction_result.jpg', vis_img)
#         print('预测结果已保存到: prediction_result.jpg')
#     else:
#         print('警告: 可视化结果不是有效的图像格式')

# # 打印关键点信息
# if 'predictions' in result:
#     predictions = result['predictions']
#     print('\n关键点预测结果:')
#     for i, pred in enumerate(predictions):
#         print(f'实例 {i+1}:')
#         for j, (x, y, score) in enumerate(zip(pred['keypoints'][:, 0], pred['keypoints'][:, 1], pred['keypoint_scores'])):
#             print(f'  关键点 {j+1}: 坐标=({x:.1f}, {y:.1f}), 置信度={score:.3f}')