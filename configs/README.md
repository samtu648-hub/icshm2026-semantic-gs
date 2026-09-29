# 配置规范

配置文件使用JSON兼容的YAML 1.2语法，因此可由Python标准库读取，不需要在仓库建设阶段引入PyYAML。

- `local.example.yaml`：本地开发示例。
- `server.example.yaml`：Ubuntu GPU服务器示例。
- `experiments/`：骨干冻结后保存可复现实验配置。

复制示例后再修改，不要在版本库中保存个人绝对路径、服务器地址或秘密：

```bash
cp configs/local.example.yaml configs/local.yaml
```

本地配置文件应由`.gitignore`排除；当前示例只定义工程接口，不虚构未冻结的训练超参数。

