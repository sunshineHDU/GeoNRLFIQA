import json

""" configuration json """


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)


config = Config({
    # 'db_name': 'win5',
    # "db_path":                      "/media/lin/EAF89872F8983F2F/Dataset/WIN5/distorted_images/",
    # "text_path":                    "/media/lin/EAF89872F8983F2F/Dataset/WIN5/1.txt",

    # # NBU
    'db_name': 'NBU',
    "db_path":                      "/media/lin/EAF89872F8983F2F/Dataset/NBU/distorted_images/",
    "text_path":                    "/media/lin/EAF89872F8983F2F/Dataset/NBU/1.txt",

    # MPI
    # 'db_name': 'MPI',
    # "db_path":                      "/media/lin/EAF89872F8983F2F/Dataset/MPI/distorted_images/",
    # "text_path":                    "/media/lin/EAF89872F8983F2F/Dataset/MPI/1.txt",


    "svPath":                       "./result",
    "mos_sv_path":                  "./data",
    'model_path': './model/resnet50.pth',
    'batch_size': 8,
    'n_epoch': 300,
    'val_freq': 1,
    'crop_size': 224,
    'aug_num': 1,
    'if_avg': True,

    'normal_test': False,
    'if_resize': True,
    'learning_rate': 5e-06,
    'weight_decay': 1e-05,
    'T_max': 50,
    'eta_min': 0,

    'num_workers': 0,
    'input_size': (512, 512),
    'new_size': (224, 224),
    'patch_size': 16,
    'img_size': 224,
    'embed_dim': 768,
    'dim_mlp': 768,
    'num_heads': [4, 4],
    'window_size': 2,
    'depths': [2, 2],
    'num_outputs': 1,
    'num_tab': 2,
    'scale': 0.13,

    'model_name': 'model_name',
    'output_path': './output/',
    'snap_path': './output/models/',
    'log_path': './output/log/',
    'log_file': '.txt',
    'tensorboard_path': './output/tensorboard/'}
)
# 0924_WIN5_Two_4SPT_Two_Atten_train20Crop80Resize_testResize_45fold_scene_group2