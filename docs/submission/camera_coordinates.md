# 相机与坐标系

官方 `images.txt` 位姿为世界到相机：

\[
\mathbf{x}_c=\mathbf{R}_o\mathbf{x}_o+\mathbf{t}_o
\]

官方世界坐标中的相机中心：

\[
\mathbf{C}_o=-\mathbf{R}_o^\mathsf{T}\mathbf{t}_o
\]

如果训练坐标和官方坐标满足：

\[
\mathbf{x}_m=s\mathbf{R}_a\mathbf{x}_o+\mathbf{t}_a
\]

则渲染适配层必须保存尺度 \(s\)、旋转 \(\mathbf{R}_a\) 和平移 \(\mathbf{t}_a\)，把官方测试相机正确转换到模型坐标。公式、实现和相机中心闭环测试必须在R2冻结时补齐并通过数值测试。

当前相机摘要：

- 模型：`SIMPLE_RADIAL`；
- 分辨率：1320×989；
- 位姿数：400；
- 畸变处理尚未冻结。

## R1 已验证适配

RGB 比较统一在无裁剪、同 K 的 PINHOLE 像素空间训练；`src/camera_adapter.py`保存官方`CAMERA_FROM_WORLD`到模型坐标的 Sim(3)公式，并将渲染结果回映射至官方`SIMPLE_RADIAL`像素网格。RGB 回映射使用双线性插值，语义回映射使用最近邻插值。

真实数据上，从 R1 准备图像执行“去畸变→回映射”对原始图像的平均绝对误差为 1.51/255（固定门槛 4.0/255）。这证明像素变换本身可闭环；候选 checkpoint 的外部相机渲染闭环由`render_external_colmap_camera.py`完成，并且是每个 R1 候选、以及 RGB-050 语义候选的必过项。

完整官方/邮件说明见[比赛指南](../../项目2比赛说明与执行指南.md)。
