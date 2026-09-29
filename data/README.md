# 数据放置与完整性规范

比赛数据不进入Git。默认检查器兼容官方发布包的当前结构：

```text
data/Contest Dataset/
├── images/                 # 001.png–300.png
├── json/                   # 001.json–300.json
├── unlabeled_Images/       # 301.png–400.png
├── camera_parameters/
│   ├── cameras.txt
│   ├── images.txt
│   ├── rigs.txt
│   └── frames.txt
└── README.md
```

Ubuntu GPU主机已存在数据，因此无需再次上传。克隆仓库后仅设置`ICSHM_DATA_ROOT`并运行完整性检查。

后续处理数据建议使用：

```text
data/
├── raw/
│   ├── images_labeled/
│   ├── annotations_labelme/
│   ├── images_unlabeled/
│   └── colmap/
├── processed/
│   ├── masks/
│   ├── undistorted/
│   └── splits/
└── manifests/
```

可以通过配置把任意外部挂载目录设为 `data_root`，不需要复制数据：

```bash
export ICSHM_DATA_ROOT=/path/on/server/Contest_Dataset
python3 scripts/check_data.py --root "$ICSHM_DATA_ROOT"
```

## 预期内容

- 300张标注RGB图像；
- 300个同名LabelMe JSON；
- 100张无标注RGB图像；
- 400个COLMAP相机位姿；
- 图像尺寸1320×989；
- 相机模型`SIMPLE_RADIAL`。

语义映射：

```text
0: background
1: deck
2: stay_cable
3: tower
4: foundation
```

`stay_cable`是Class ID 2的唯一正式名称。任何旧的`main cable`字符串都需要在进入训练前显式映射并登记。

## 完整性检查

```bash
python3 scripts/check_data.py \
  --root "data/Contest Dataset" \
  --write-manifest data/manifests/contest_dataset_v1.json \
  --fingerprint
```

Ubuntu生成临时manifest后，使用：

```bash
python3 scripts/compare_manifests.py \
  data/manifests/contest_dataset_v1.json \
  /tmp/ubuntu_contest_dataset_manifest.json
```

检查器只读原始数据，记录相对目录结构、数量、尺寸、LabelMe标签计数、相机摘要和聚合内容指纹，不写入个人绝对路径。

原始图像和JSON不得修改。mask、去畸变图和split必须写入`processed/`或外部输出目录，并由配置记录生成步骤。
