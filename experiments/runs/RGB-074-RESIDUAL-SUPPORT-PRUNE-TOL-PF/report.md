# RGB-074-RESIDUAL-SUPPORT-PRUNE-TOL-PF — 数值成对29-anchor保守屏蔽

- 状态：completed_numeric_gate_passed_visual_cleanup_failed
- 父实验：`RGB-030-FULL`, `RGB-072`, `RGB-073`
- 唯一研究变量：与RGB-073完全相同的29-anchor推理屏蔽

RGB-073暴露的是重放末位量化问题，不是mask选择结果：054只有25像素各差1/255。故本ID
在运行前注册严格数值等价门：每视角最大uint8差≤1、mean absolute uint8差≤1e-4、
差异像素比例≤1e-4；保存全部同次重放基线。任一视角超门即停止。

候选同时与同次重放基线、历史RGB-030-FULL比较；mean PSNR/SSIM/LPIPS不得反向，历史
PSNR不得降低，最差10%和140–146 PSNR最多下降0.02 dB且至少一项全局指标严格改善。
29-anchor规则、NPZ哈希和预选样例完全不变，不扫描任何新阈值。

## 结果与决定

74-view成对渲染完成。历史基线重放总计只有1,932个像素末位不同，单视角最多38像素
（2.91e-5比例），最大差1/255、单视角最大mean uint8差`9.70e-06`，全部通过注册容差。
29-anchor mask影响1,696,364个验证像素，最大通道差38/255；checkpoint/PLY/MLP哈希
前后相同。

| 指标 | 同次重放基线 | 29-anchor候选 | 差值 |
|---|---:|---:|---:|
| mean PSNR (dB) | 20.596397 | 20.597479 | +0.001081 |
| mean SSIM | 0.850177 | 0.850242 | +0.000065 |
| mean LPIPS-Alex | 0.322373 | 0.322256 | −0.000116 |
| worst-10% PSNR (dB) | 17.733999 | 17.730253 | −0.003746 |
| 140–146 PSNR (dB) | 21.338724 | 21.341829 | +0.003106 |

PSNR/SSIM/LPIPS分别在68/70/44个视角相对重放基线改善，数值门全部通过；但改善量远小于
可感知尺度。原分辨率固定复核054/140/151/159/160，并补看改动像素最多的183，主要
天空横条、幽灵索线、水面/近岸拉伸均仍在，候选与基线肉眼近乎一致。局部变化也不稳定：
例如140的PSNR/SSIM改善但LPIPS恶化约0.00060。

产物位于`ICSHM_OUTPUT_ROOT/r2/rgb_074/residual_support_prune_tol/full-20260903-r1`；
`render_result.json`、候选/重放指标摘要SHA256分别为
`ead1339024f36f87c954ffa1edc51ffbb7ba8161192fec9dfb9c6de0206fcdc7`、
`5db32dea96aa9f44b9615fa4b98ebfb59c11e02667005e65ddd573f490ea392a`、
`60c7accae89a0586ce85680f6d4c8b4bb5e921b0ca11e60b753b5b668fd3fcca`。

keep_or_drop：`drop_as_practical_cleanup_keep_reversible_diagnostic_artifact`。数值门通过，但
预注册视觉目标“真实移除浮点且不伤结构”未达到，不提升为新RGB最佳，也不触发语义重跑。
当前最佳仍为RGB-030-FULL。禁止据74-view结果扩大support/MAE/scale/opacity阈值；下一条
有效路线需要从训练过程处理天空/瞬态内容、尺度/opacity正则或更强重建表示，而不是继续
事后删点。
