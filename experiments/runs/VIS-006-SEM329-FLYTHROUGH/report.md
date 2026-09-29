# VIS-006 — 阶段报告与最新三臂同轨迹视频

完成日期：2026-09-08。Status: `completed_verified_report_and_flythrough`。

用户授权范围是整理现有研究并补做真实可视化推理。本次未训练、调阈值、覆盖正式包或产生新准确率。事前配置保持原字节及`preregistered`字段，实际完成状态由本报告和registry记录。

## 来源与范围

- 父产物：VIS001相机轨迹、RGB-030-FULL冻结Scaffold、SEM161正式分支、SEM326首源复核和SEM329第二源否决。
- Git基点：`49cdcdb1c9915a9544d568754af95ea20a85ec9c`，附独立封存的工作树源文件hash；没有把未提交源码伪称为该commit的内容。原有用户改动保留。
- config SHA256：`74c8df049ef497ae26d25c37c8622f76dae1976eb8e57e7f6e70df53d52fdf51`。
- 数据manifest为`contest_dataset_v1`，语义划分为`sem_001_scaffold_static_v1`；固定219训练参考库，seed20260805。manifest、split和三项相机文件hash见config。
- 新视角为VIS001原144个插值相机，24fps、6秒；官方COLMAP CAMERA_FROM_WORLD坐标，SIMPLE_RADIAL相机畸变适配。新视角没有GT，真实目标RGB及目标GT不进入推理。
- 报告静态比较另读取用户指定的10张既有开发图、对应封存预测和GT，只用于原样展示与变化图；不重开全74图调参或重新选规则。

## 实际执行

1. 先通过原有单帧接口fresh重放frame0000、frame0072、frame0143。随后只批量复用模型加载，保持原SEM161/326/329规则；三帧的RGB、PINHOLE、三臂语义及选源与原接口精确一致后才展开其余帧复核。
2. 全144帧完成。每帧独立几何与原生语义复算通过，SEM161/326/329的class1/2/3逐像素不变。没有目标GT、新增训练、时间平滑或显示清理。
3. 固定219源库中实际解码15个参考源标签；第二源在90帧产生否决，恢复4605个旧删除像素（旧删除总计31422）。这些是操作计数，没有GT，不能称为4605个正确恢复。
4. 推理共591.997秒；匹配器峰值allocated为2509.354MiB，基础渲染峰值2207.679MiB。三代表帧重放已计入总时间。
5. 全144帧RGB与历史VIS001原始RGB逐像素一致；18项冻结产物hash在执行前、执行后及最终报告审计中均不变。

推理主记录：`outputs/vis006/full-20260908-r1/result.json`，SHA256 `883adc9445ff3bdbaf497d9951f2aa7c4f9ddee1ccad5fa75cc524ca1151bc10`。
媒体清单：同目录`media_manifest.json`，SHA256 `1d8cb5ee05ca313a94d9980b3514585fb93e9be705084747b03ae2a6f624f531`。
该目录还保留源码快照、三个fresh探针、逐帧匹配和预测。此次VIS推理无失败运行；既有实验中的失败记录及旧视频没有删除。报告初版检查发现中英数字字体问题，已修正并重新逐页检查，没有改动预测。

## 交付

- 中文离线网页：`outputs/research_report_20260908/index.html`，九类路线和146项登记索引，原状态、失败、阻塞和未执行项目分别保留。
- 八页中文PDF：同目录`research-summary.pdf`。覆盖任务、RGB骨干、语义主线、路线地图、最新配对比较、局部与动态结果。
- 固定054/081/140/159/160/183，补充180/145/088/061。各例提供原生PNG、GT/预测整数ID、六联全图、同坐标局部、历史语义、RGB骨干和变化像素图；语义放大采用最近邻，Class ID 2正式名称为`stay_cable`。
- 四个当前视频：纯RGB、SEM329纯语义、RGB/SEM329叠加并排、SEM161/326/329三臂并排。各144帧/24fps/6秒；原生图块不缩放，只增加标题图例和H.264所需偶数边界。MP4为有损展示副本，PNG才是逐像素记录。
- 三个历史视频有明确版本标记；保留旧色表，不将旧色表差异解释为类别改变。
- 动态坏例保留frame0072的前景浮点和模糊，以及0071/0072/0073连续语义裁剪；屏幕裁剪不是三维点配准，不把所有帧差称为错误。
- `data/content.json`作为网页/PDF共同内容清单，含Git/config/seed/split映射；`data/provenance.json`及`package-manifest.json`提供来源和文件hash。压缩包为`outputs/research_report_20260908.zip`，只打包报告与选定媒体，不含checkpoint、完整数据集或稠密匹配缓存。

报告清楚区分正式SEM161（83.387017%）与研究候选SEM329（83.705003%）。后者相对SEM161 mIoU+0.317985pp，但foundation recall-1.424762pp；数据是历史复用74图的本地开发证据，非官方隐藏成绩。

## 验证

- 从封存混淆矩阵核对最新三臂mIoU；RGB原始结果、历史registry与报告图表同源。
- 10例预测/GT/RGB来源hash、1320×989空间、类别ID及颜色映射、同位置裁剪和保护类别检查通过。
- 四个MP4实际解码均为144帧；核对0/72/143的预编码PNG，平均绝对像素差0.397–1.628（H.264正常有损误差），未发现顺序错位。
- 浏览器实测本地file打开、全部图片、10例选择、历史/RGB切换、局部放大、索引查找、四视频暂停/定位/0.5倍速；1440、390、320宽度均通过。无外部网络请求，无脚本错误。
- 压缩包解压到另一目录后，同套离线浏览器检查通过。8页PDF渲染后逐页检查中文、英文字体、数字、图表、图例与分页。
- `scripts/audit_research_report.py`和`scripts/check_research_report.mjs`为可重复审计入口；结果保留在`outputs/report_qa_20260908/`。
- 仓库健康检查通过：137份配置，273项单元测试（53项按现有环境跳过）；`git diff --check`通过。没有提交、推送或外部上传。

## 复建与回退

路径由参数和项目根确定，不要求原个人目录。`REPORT_PYTHON`需numpy/Pillow/reportlab/pypdf，`PLOT_PYTHON`需matplotlib；浏览器审计另需Playwright和本地Chrome。

```bash
"$REPORT_PYTHON" scripts/build_research_report.py \
  --project-root "$PWD" \
  --artifact-root "$ICSHM_OUTPUT_ROOT" \
  --data-root "$ICSHM_DATA_ROOT" \
  --output-dir outputs/research_report_20260908 \
  --video-dir outputs/vis006/full-20260908-r1 \
  --plot-python "$PLOT_PYTHON"
```

若需独立重放视频，另选不存在的`VIS_OUTPUT_DIR`，保持此次封存目录不动：

```bash
PYTHONPATH=outputs/sem320/runtime "$RENDER_PYTHON" scripts/render_vis006_sem329_flythrough.py \
  --artifact-root "$ICSHM_OUTPUT_ROOT" --data-root "$ICSHM_DATA_ROOT" \
  --roma-dir outputs/sem320/upstream-romav2 --dinov3-dir outputs/sem320/upstream-dinov3 \
  --weights outputs/sem320/romav2.0.1.pt \
  --bundle "$ICSHM_OUTPUT_ROOT/submission/sem161-20260907-r3" \
  --render-python "$RENDER_PYTHON" --output-dir "$VIS_OUTPUT_DIR"
"$REPORT_PYTHON" scripts/compose_vis006_videos.py \
  --run-dir "$VIS_OUTPUT_DIR" --artifact-root "$ICSHM_OUTPUT_ROOT"
```

本次是附加报告与独立可视化产物；无需恢复任何模型即可继续使用正式SEM161。忽略新报告目录即可回到原入口，VIS001/旧视频仍在原位置。研究风险和技术方向未改变，因此不改写既有DECISIONS/RISKS；正式最终拟合、SEM329提交包整合与官方上传仍属后续任务。
