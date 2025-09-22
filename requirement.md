# 需求纪要（MVP）— MOF 数据 t‑SNE 可视化

创建日期: 2025-09-21
议题来源: 端到端竖切（上传 → 设参 → 处理 → 可视化 → 导出）

## 核心议题
以“可演示的 MVP”为目标，稳定、高效地绘制 9 张 t‑SNE 图像：
- 图1：在 12088 的嵌入中，前景高亮“PLD>4.9 的 4519”，其余为灰色背景
- 图2/3：在 4519 的嵌入中，高亮 吸水 / 不吸水
- 图4/5：在 4519 的嵌入中，高亮 水稳定 / 水不稳定
- 图6/7/8/9：在 4519 的嵌入中，分别高亮四个组合类

所有图支持：hover 显示 sample_id 与类别；导出 PNG/SVG；颜色与透明度可自定义。

## 数据与集合定义
- Full 集（E1 背景）：n=12088，完整原始数据
- PLD 筛选集（E2 背景）：n=4519，筛选条件为 PLD>4.9
- 标注子集（前景用于高亮）：n=3264，四类标签来自 Water_stability 与 KH_water 的组合
  - 映射规则：
    - Water_stability>0.5 → 水稳定；否则水不稳定
    - KH_water=='strong' → 吸水；'weak' → 不吸水
    - 组合成四类中文标签：
      吸水&水稳定MOF / 吸水&水不稳定MOF / 不吸水&水稳定MOF / 不吸水&水不稳定MOF

## 输入最小约定（MVP）
- 推荐提供 3 个 CSV（字段一致）：
  1) CSV_full_12088.csv：包含 sample_id、PLD、Water_stability、KH_water、数值特征
  2) CSV_pld_filtered_4519.csv：同上字段（为性能/可控性单独处理）
  3) CSV_labeled_3264.csv：4519 的子集，额外含一列 category（四类中文标签）
- 亦可改为：一张 4519 的 CSV，若方便你把 3264 合并并补上 category 列
- 数值特征：保留用于建模的数值列；非数值列（如 DOI/Source）不参与计算但可用于 hover

## 处理流水线与默认参数
- 预处理（两套嵌入一致，仅用于数值特征）
  - 缺失值填充：mean（或 median，择一）
  - 标准化：StandardScaler
  - 去噪：低方差剔除 var<1e-5；强相关剔除 |r|>0.98（保留一方）
- PCA：n_components=50（去噪+加速）
- t‑SNE：
  - E1（12088）：perplexity=50，n_iter=1000–1500，learning_rate=400–800，angle=0.5，random_state=42
  - E2（4519）：perplexity=30，n_iter=1000，learning_rate=200–400，angle=0.5，random_state=42
- 说明：t‑SNE transform 不稳定，背景与前景必须在一次嵌入中共同计算；因此采用两套嵌入（E1/E2）。

## 可视化规格
- 视图集合：
  - 图1（E1）：12088 嵌入为背景（灰，opacity≈0.25），高亮 4519（彩色，opacity≈0.8）
  - 图2/3（E2）：4519 为背景，分别高亮 吸水 / 不吸水
  - 图4/5（E2）：4519 为背景，分别高亮 水稳定 / 水不稳定
  - 图6/7/8/9（E2）：4519 为背景，分别高亮四个组合类
- 交互：hover 显示 sample_id 与类别；图例可开/关；可调整点大小/透明度
- 自定义：颜色与透明度可通过 UI 配置；默认 size=8、opacity=0.8（前景）/0.25（背景）
- 导出：PNG、SVG（默认 800×600，可调）

## 接口与契约（对齐 OpenAPI）
- 路由前缀统一为 `/api`，子路由消除重复资源名：
  - POST `/api/upload`（上传 CSV，返回 dataset_id）
  - POST `/api/datasets/{dataset_id}/process`（启动处理，支持选 E1/E2 与参数）
  - GET `/api/pipelines/{pipeline_id}/status`（进度）
  - GET `/api/visualizations/{pipeline_id}`（坐标与可视化数据）
  - POST `/api/visualizations/{pipeline_id}/export`（导出 PNG/SVG）
- 前端请求与以上路径一致；progress 统一命名为 status

## 验收标准（MVP）
- 能在 Web UI 中加载两套嵌入（E1/E2）并切换视图，完成 9 张图的呈现
- 交互顺滑（E2 视图为主的 4519 嵌入流畅；E1 可接受轻微卡顿）
- 颜色与透明度可在 UI 调整；导出 PNG/SVG 保持当前样式
- 一键演示：上传 CSV → 设参（默认即可） → 处理 → 可视化 → 导出，一次通过

## 实施计划（竖切优先级）
1) 统一后端 API 路径与前端调用，按 OpenAPI 契约（高优先）
2) 后端支持“双嵌入”处理：加载 E1/E2 数据与参数，生成并缓存坐标
3) 前端视图层：
   - 视图切换（E1/E2 + 9 个高亮模式）
   - 颜色/透明度/点大小 UI 与状态持久
   - 导出按钮（PNG/SVG）
4) 契约/集成测试最小集：上传/处理/取图/导出（绿）
5) 性能与稳定性微调：perplexity/learning_rate 兜底；内存/时间日志

## 关键决策点
- 数据输入采用“三 CSV”或“一 CSV（4519+category）”的模式；若采用三 CSV，后端合并 3264 仅用于高亮，不单独嵌入
- t‑SNE 参数可在 UI 上调；默认值以一次出图为主
- 若未来要把 12088 加入更多视图：需评估渲染与交互负担，可能引入 openTSNE/FIt‑SNE 或采样

## 风险与考虑
- 12088 的嵌入耗时与前端渲染压力偏高（可接受为单图展示）
- t‑SNE 不可复用 transform，任何集合变化需重新嵌入
- 类别分布不均衡可能导致视觉密度差异（可通过透明度/点大小缓解）

