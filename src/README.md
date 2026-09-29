# Source code

不复制第三方 3DGS 仓库到此目录；它们通过明确的上游 URL、commit 和许可证管理。项目自有的相机适配、语义模块、评测和提交接口放在这里。

`camera_adapter.py`是 R1 共同相机适配层：

- 解释官方 COLMAP `CAMERA_FROM_WORLD`；
- 按保存的 Sim(3) 把官方相机转换到模型坐标；
- 将官方`SIMPLE_RADIAL`输入转换为全帧 PINHOLE 训练空间；
- 将 RGB 和单通道语义渲染映射回官方像素网格，其中语义始终用最近邻。

它只依赖 NumPy 和 OpenCV，二者在冻结训练环境前是可选依赖。真实数据的往返验证使用：

```bash
python3 scripts/validate_camera_adapter.py \
  --dataset-root "$ICSHM_DATA_ROOT" \
  --prepared-scene "$ICSHM_OUTPUT_ROOT/icshm2026/rgb_r1_spatial_v1/scene_400"
```

Geometry-prior preparation lives under `geometry_prior/` and is deliberately
independent of the frozen Scaffold renderer:

- `camera_io.py` parses official cameras/images/rigs/frames, COLMAP quaternions,
  projection and backprojection;
- `build_camera_graph.py` writes the co-visibility graph;
- `segment_trajectory.py` writes audited, non-video trajectory segments;
- `sim3.py` contains Umeyama alignment utilities;
- `run_mapanything.py`, `run_da3.py`, and `run_lingbot.py` only create bounded
  window manifests until an approved model source/checkpoint is installed.

The current host audit produced `experiments/geometry_camera_graph.json` and
`experiments/trajectory_segments.json`, and completed the same-budget
Scaffold control `GEO-000`. Optional forward-model packages were absent and
were not downloaded.
