# Experiment tracking

本目录使用可版本控制的配置、注册表和Markdown报告追踪实验，不依赖外部数据库。

`registry.yaml`采用JSON兼容的YAML 1.2语法，以便在尚未锁定环境时由Python标准库读写。

## 创建实验

```bash
python3 scripts/new_experiment.py \
  --id SEM-001 \
  --title "Static five-class semantic baseline" \
  --phase S0 \
  --parent E2 \
  --backbone frozen-rgb
```

规则：

- ID创建后不可复用；
- `config.yaml`记录与父实验唯一差异；
- 原始输出放在实验目录下被忽略的`output/`，或仓库外；
- 报告和registry必须在实验结束后更新；
- `git_commit`不得长期保持`unknown`，提交实验代码后要补回commit；
- 失败实验标记`keep_or_drop: drop`并写原因，不删除登记。

