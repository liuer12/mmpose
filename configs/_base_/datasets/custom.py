dataset_info = dict(
    dataset_name='strawberry',
    paper_info=dict(
        author='zzd',
        title='Strawberry Keypoint Detection',
        container='Custom Dataset',
        year='2024',
        homepage='',
    ),
    keypoint_info={
        0: dict(
            name='KP1',
            id=0,
            color=[0, 255, 0],      # 绿色
            type='upper',
            swap=''),
        1: dict(
            name='KP2',
            id=1,
            color=[255, 0, 0],      # 红色
            type='upper',
            swap=''),
        2: dict(
            name='KP3',
            id=2,
            color=[0, 0, 255],      # 蓝色
            type='upper',
            swap=''),
        3: dict(
            name='KP4',
            id=3,
            color=[255, 255, 0],    # 黄色
            type='middle',
            swap=''),
        4: dict(
            name='KP5',
            id=4,
            color=[255, 0, 255],    # 紫色
            type='middle',
            swap=''),
        5: dict(
            name='KP6',
            id=5,
            color=[0, 255, 255],    # 青色
            type='lower',
            swap=''),
    },
    skeleton_info={
        0: dict(link=('KP1', 'KP2'), id=0, color=[0, 255, 0]),
        1: dict(link=('KP2', 'KP3'), id=1, color=[255, 0, 0]),
        2: dict(link=('KP4', 'KP5'), id=2, color=[255, 255, 0]),
        3: dict(link=('KP5', 'KP6'), id=3, color=[255, 0, 255]),
    },
    joint_weights=[
        1.0,  # KP1
        1.0,  # KP2
        1.0,  # KP3
        1.0,  # KP4
        1.0,  # KP5
        1.0,  # KP6
    ],
    sigmas=[
        0.025,  # KP1
        0.025,  # KP2
        0.025,  # KP3
        0.025,  # KP4
        0.025,  # KP5
        0.025,  # KP6
    ]) 