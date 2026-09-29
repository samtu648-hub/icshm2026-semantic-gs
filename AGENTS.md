# Agent Working Agreement

本文件约束所有使用 Codex 或其他自动化代理的仓库任务。

## 开始任务前

1. 阅读 `PROJECT_STATUS.md`、`ROADMAP.md`、`TASKS.md`。
2. 阅读与任务有关的 `DECISIONS.md` 条目和 `RISKS.md`。
3. 运行 `git status --short --branch`，保护用户的未提交修改。
4. 确认目标实验 ID、父实验、数据 manifest 和 split。

## 实施规则

- 一次只改变一个主要实验变量。
- 任何指标必须关联实验 ID、Git commit、配置、seed和数据划分。
- 失败实验同样需要登记，不得静默删除。
- 不把未运行的实验、未经验证的指标或计划写成已完成事实。
- 不修改冻结模块，除非 `DECISIONS.md` 中的可推翻条件已经满足并登记新决定。
- 语义实验必须验证冻结后的 RGB 输出是否保持一致。
- 所有外部相机渲染必须兼容官方 COLMAP 参考坐标系。
- 使用 `stay_cable` 作为 Class ID 2 的唯一正式名称。
- 路径通过配置、项目根目录或环境变量确定，不写死个人绝对路径。

## 数据与安全

- 不提交比赛图像、LabelMe JSON、COLMAP大型文件、checkpoint、PLY、日志、渲染输出或缓存。
- 不提交 `.env`、密钥、token、服务器地址、SSH配置或个人信息。
- 引入第三方项目时记录来源、commit和许可证，不整仓复制到Git。
- 不重写Git历史、删除远程资源或执行Git push，除非用户明确授权。

## 完成任务后

1. 更新对应实验的 `report.md` 和 `experiments/registry.yaml`。
2. 更新 `TASKS.md`。
3. 更新 `PROJECT_STATUS.md`，下一步最多保留三个任务。
4. 如改变技术方向，更新 `DECISIONS.md` 和 `RISKS.md`。
5. 运行 `bash scripts/repo_health.sh`。
6. 报告改动、验证证据、剩余风险和回退方式。

