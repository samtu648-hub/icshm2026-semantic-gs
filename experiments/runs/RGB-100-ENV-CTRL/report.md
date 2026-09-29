# RGB-100-ENV-CTRL

状态：`completed_matched_control`；不作为新正式模型。

使用与RGB-101相同的隔离代码路径、303训练RGB、seed 0、30k、原始Scaffold loss、densification及appearance；唯一差异是1536参数的环境表恒为零且不加入优化器。

74-view开发审计：PSNR 20.566897 dB、SSIM 0.8513588、LPIPS 0.3202040、worst10 PSNR 17.387185、140–146 PSNR 21.221483。完整训练+原生渲染过程3111.567秒，peak Torch allocated 6941.226MiB；最终189,483 anchors。新进程optimizer/environment重载和官方140相机均通过。

外部产物：`$ICSHM_OUTPUT_ROOT/r2/rgb_101/runs/RGB-100-ENV-CTRL-full-r1/`；execution JSON SHA256 `000638424a7cd85d8764333e0084a17f4d2365dddb45d2a0ebf5e774fb876d5f`。它仅是匹配控制，不改变`RGB-030-FULL + SEM-161`。
