conf_1p = {
    # please change to your own path
    "WORK_PATH": ".",
    "ASCEND_VISIBLE_DEVICES": "0",
    "data": {
        'dataset_path': "../../CASIA-B-Pre/",
        'resolution': '64',
        'dataset': 'CASIA-B',
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        # For more detail, please refer to
        # function: utils.data_loader.load_data
        'pid_num': 73,
        'pid_shuffle': False,
    },
    "model": {
        'hidden_dim': 256,
        'lr': 8e-4,
        'hard_or_full_trip': 'full',
        'batch_size': (8, 16),
        'restore_iter': 0,
        'total_iter': 40000,
        'margin': 0.2,
        'num_workers': 8,
        'frame_num': 30,
        'model_name': 'GaitSet',
    },
}

conf_8p = {
    # please change to your own path
    "WORK_PATH": ".",
    "ASCEND_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
    "data": {
        'dataset_path': "../../CASIA-B-Pre/",
        'resolution': '64',
        'dataset': 'CASIA-B',
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        # For more detail, please refer to
        # function: utils.data_loader.load_data
        'pid_num': 73,
        'pid_shuffle': False,
    },
    "model": {
        'hidden_dim': 256,
        'lr': 8e-4,
        'hard_or_full_trip': 'full',
        'batch_size': (8, 16),
        'restore_iter': 0,
        'total_iter': 40000,
        'margin': 0.2,
        'num_workers': 8,
        'frame_num': 30,
        'model_name': 'GaitSet',
    },
}
