# 协作规范

## 分支

- `main`：可读、可验证的共享状态。
- `feat/<topic>`：功能开发。
- `fix/<topic>`：缺陷修复。
- `exp/<experiment-id>-<topic>`：实验代码或配置。
- `docs/<topic>`：文档。

三名组员的具体姓名尚未登记，任务负责人暂用 `unassigned`。

## Commit

推荐格式：

```text
<type>(<scope>): <summary>
```

例如：

```text
feat(experiments): add RGB-001 Standard 3DGS configuration
docs(status): record R1 backbone decision
```

代码、配置和实验摘要可以提交；原始数据、大型产物、checkpoint和完整渲染结果不得提交。实验报告中的关键图应经过筛选、体积小且不泄露受限数据。

## 合并检查

```bash
bash scripts/repo_health.sh
```

合并请求需要说明：

- 改变的唯一主要变量；
- 对应任务与实验 ID；
- 验证命令和结果；
- 是否改变冻结决定；
- 已知风险与回退方式。

当前不建立可能持续失败的CI；依赖版本和首个可运行骨干冻结后再添加真实CI。

