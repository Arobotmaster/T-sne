# Tasks: MOF数据t-SNE交互式可视化

**Input**: Design documents from `/specs/001-md/`
**Prerequisites**: plan.md (completed), research.md, data-model.md, contracts/, quickstart.md

## Execution Flow (main)
```
1. Load plan.md from feature directory ✓
   → Extract: Python 3.11+ + FastAPI + scikit-learn + Plotly.js
   → Structure: Web application (backend/ + frontend/)
2. Load design documents ✓
   → data-model.md: 5 core entities → model tasks
   → contracts/api.yaml: 8 endpoints → contract test tasks
   → research.md: Technology decisions → setup tasks
3. Generate tasks by category:
   → Setup: project structure, dependencies, linting
   → Tests: contract tests, integration tests (TDD first)
   → Core: models, services, algorithms, CLI commands
   → Integration: API, logging, error handling
   → Polish: unit tests, performance, docs
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001, T002...)
6. Generate dependency graph
7. Create parallel execution examples
8. Validate task completeness
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions


## Phase 0: MVP Delta（新增：两套嵌入 + 九视图 + 自定义颜色/透明度）

目标：在不重构的前提下，基于现有实现快速交付 MVP（E1=12088、E2=4519），提供 9 张视图，并允许颜色/透明度/点大小自定义；统一 API 契约。

- [ ] M101 API 路由对齐 contracts（高优先）
  - backend/src/api/__init__.py：保留 include_router 前缀，子路由内部路径改相对，避免重复资源名
  - backend/src/api/datasets.py：列表 `@router.get("")`，详情 `@router.get("/{dataset_id}")`，状态 `@router.get("/{dataset_id}/status")`
  - backend/src/api/pipelines.py：进度统一为 `GET /pipelines/{id}/status`（不再使用 progress）
  - backend/src/api/visualizations.py：保持 `GET /visualizations/{pipeline_id}` / `POST /visualizations/{pipeline_id}/export`

- [ ] M102 前端 API 调整与调用点联动
  - frontend/src/services/api.js：修正 `/datasets/datasets` → `/datasets`；`/pipelines/${id}/progress` → `/pipelines/${id}/status`
  - 关联调用（frontend/src/app.js、components/*）同步更新

- [ ] M103 双嵌入支持与结果缓存（E1=12088，E2=4519）
  - backend/src/services/processing_service.py：增加 embedding 选择参数与结果缓存（results/<embedding_id>/）
  - backend/src/services/visualization_service.py：按 pipeline_id/embedding_id 返回坐标与统计
  - backend/src/api/datasets.py 的 `/process` 入参支持 embedding/参数透传

- [ ] M104 前端视图切换与高亮模式（9 张图）
  - frontend/src/components/visualization.js：添加嵌入(E1/E2)选择与高亮模式（吸水/不吸水；水稳定/不稳定；四组合类；以及 E1 中的 4519 前景）
  - 背景：E1 使用 12088 灰色背景；E2 使用 4519 灰色背景；前景彩色高亮

- [ ] M105 颜色/透明度/点大小自定义 UI（持久化）
  - 可配置四类颜色映射、前景/背景透明度、marker size；使用 localStorage 持久

- [ ] M106 导出一致性验证（样式一致导出）
  - 前端 Plotly 导出 PNG/SVG 需与当前样式一致；后端导出参数校验

- [ ] M107 数据输入与合并逻辑（单表或三表）
  - 支持：一张 4519 CSV（含 category）+ 一张 12088 CSV；或三 CSV（12088、4519、3264标注）
  - 三表模式下，合并 3264 仅用于高亮，不单独嵌入

- [ ] M108 契约/集成测试修复与补充
  - 修复路径差异（progress→status、去重前缀）；新增 E1/E2 产出可视化数据的 happy path

- [ ] M109 文档对齐与验收指引
  - README.md、specs/001-md/spec.md 更新 E1/E2 与 9 视图说明；链接根目录 requirement.md

### 依赖关系（Delta）
- M101 → M102 → (M103 [P], M104 [P]) → M105 → M106 → M108 → M109
- [P] 可并行：M103 与 M104

### 并行执行示例（Delta）
- 路由调整后，可并行推进：
  - M103（后端双嵌入与缓存）
  - M104（前端九视图切换与高亮）


- **Web application**: `backend/src/`, `frontend/src/`, `backend/tests/`
- Based on plan.md structure decision: Option 2 - Web Application

## Phase 3.1: Setup (基础设置)

### 项目结构初始化
- [x] T001 创建Web应用项目结构 (backend/, frontend/, tests/, data/, logs/)
- [x] T002 初始化Python后端项目 (FastAPI + scikit-learn + pandas + numpy)
- [x] T003 [P] 配置Python开发工具 (black, flake8, pytest, mypy)
- [x] T004 初始化前端项目结构 (JavaScript + Plotly.js + Bootstrap)
- [x] T005 [P] 配置前端开发工具 (ESLint, Prettier, Jest)

### 依赖和配置
- [x] T006 创建后端requirements.txt (包含所有必要依赖)
- [x] T007 创建前端package.json (包含Plotly.js等依赖)
- [x] T008 [P] 设置Docker配置 (Dockerfile, docker-compose.yml)
- [x] T009 [P] 配置日志系统 (logging_config.py)
- [x] T010 创建.gitignore和项目文档

## Phase 3.2: Tests First (TDD) ⚠️ 必须在3.3之前完成

**关键：这些测试必须先编写且必须失败，然后才能开始任何实现**

### 契约测试 (基于contracts/api.yaml)
- [x] T011 [P] 契约测试 POST /api/upload 文件上传 (tests/contract/test_upload.py)
- [x] T012 [P] 契约测试 GET /api/datasets/{id}/status 状态查询 (tests/contract/test_datasets.py)
- [x] T013 [P] 契约测试 POST /api/datasets/{id}/process 处理启动 (tests/contract/test_processing.py)
- [x] T014 [P] 契约测试 GET /api/pipelines/{id}/progress 进度查询 (tests/contract/test_pipelines.py)
- [x] T015 [P] 契约测试 GET /api/visualizations/{id} 可视化数据 (tests/contract/test_visualizations.py)
- [x] T016 [P] 契约测试 POST /api/visualizations/{id}/export 图像导出 (tests/contract/test_export.py)
- [x] T017 [P] 契约测试 WebSocket /ws/pipelines/{id} 实时更新 (tests/contract/test_websocket.py)

### 集成测试 (基于用户故事)
- [x] T018 [P] 集成测试 完整数据处理流水线 (tests/integration/test_pipeline.py)
- [x] T019 [P] 集成测试 文件上传到可视化流程 (tests/integration/test_full_workflow.py)
- [x] T020 [P] 集成测试 参数配置和实时更新 (tests/integration/test_parameters.py)
- [x] T021 [P] 集成测试 大数据集处理能力 (tests/integration/test_performance.py)
- [x] T022 [P] 集成测试 错误处理和恢复 (tests/integration/test_error_handling.py)

### 算法测试 (基于Library-First原则)
- [x] T023 [P] 算法测试 PCA处理器功能 (tests/unit/algorithms/test_pca.py)
- [x] T024 [P] 算法测试 t-SNE处理器功能 (tests/unit/algorithms/test_tsne.py)
- [x] T025 [P] 算法测试 数据预处理功能 (tests/unit/algorithms/test_preprocessing.py)

## Phase 3.3: Core Implementation (核心实现 - 仅在测试失败后开始)

### 数据模型 (基于data-model.md)
- [x] T026 [P] MOF数据模型 (backend/src/models/dataset.py)
  - MOFDataset, MOFSample, NumericalFeature, DescriptiveData
- [x] T027 [P] 处理流水线模型 (backend/src/models/pipeline.py)
  - ProcessingPipeline, DataProcessingStep, AlgorithmConfig
- [x] T028 [P] 可视化模型 (backend/src/models/visualization.py)
  - VisualizationConfig, VisualizationResult, TSNECoordinates
- [x] T029 [P] 分类标签模型 (backend/src/models/category.py)
  - CategoryLabel, ColorScheme, MarkerStyle

### 算法实现 (Library-First原则)
- [x] T030 [P] PCA算法实现 (backend/src/algorithms/pca.py)
- [x] T031 [P] t-SNE算法实现 (backend/src/algorithms/tsne.py)
- [x] T032 [P] 数据预处理算法 (backend/src/algorithms/preprocessing.py)
- [x] T033 [P] 基础算法类和接口 (backend/src/algorithms/base.py)

### 服务层
- [x] T034 [P] 文件上传服务 (backend/src/services/upload_service.py)
- [x] T035 [P] 数据处理服务 (backend/src/services/processing_service.py)
- [x] T036 [P] 可视化服务 (backend/src/services/visualization_service.py)
- [x] T037 [P] 导出服务 (backend/src/services/export_service.py)
- [x] T081 [P] 对比可视化服务 (backend/src/services/comparison_service.py)

### CLI接口实现 (CLI Interface原则)
- [x] T038 [P] CLI主程序 (backend/src/cli/main.py)
- [x] T039 [P] PCA命令行工具 (backend/src/cli/pca_command.py)
- [x] T040 [P] t-SNE命令行工具 (backend/src/cli/tsne_command.py)
- [x] T041 [P] 批处理命令 (backend/src/cli/batch_command.py)

### API端点实现
- [x] T042 POST /api/upload 文件上传端点
- [x] T043 GET /api/datasets/{id}/status 状态查询端点
- [x] T044 POST /api/datasets/{id}/process 处理启动端点
- [x] T045 GET /api/pipelines/{id}/progress 进度查询端点
- [x] T046 GET /api/visualizations/{id} 可视化数据端点
- [x] T047 POST /api/visualizations/{id}/export 图像导出端点
- [x] T048 WebSocket /ws/pipelines/{id} 实时更新端点
- [x] T082 POST /api/visualizations/comparison 对比可视化端点
- [x] T083 PUT /api/visualizations/{id}/comparison-config 对比可视化配置端点

## Phase 3.4: Integration (集成)

### 中间件和配置
- [x] T049 连接所有服务到API端点
- [x] T051 配置文件上传大小限制和类型验证
- [x] T052 设置临时文件管理和清理机制

### 日志和监控 (Scientific Observability)
- [x] T053 实现详细的科学计算日志记录
- [x] T054 添加性能指标收集 (计算时间、内存使用)
- [x] T055 配置错误日志和调试信息
- [x] T056 实现处理进度跟踪系统

### 错误处理
- [x] T057 实现统一的错误处理机制 (backend/src/api/error_handlers.py)
- [x] T058 添加输入验证和数据清理 (backend/src/utils/validation.py)
- [x] T059 实现优雅的降级处理 (backend/src/middleware/graceful_degradation.py)
- [x] T060 添加用户友好的错误消息 (backend/src/utils/user_messages.py)

## Phase 3.5: Polish (完善)

### 前端实现
- [ ] T061 [P] 主HTML页面和布局 (frontend/static/index.html)
- [ ] T062 [P] 文件上传组件 (frontend/src/components/upload.js)
- [ ] T063 [P] 参数配置组件 (frontend/src/components/parameters.js)
- [ ] T064 [P] 可视化组件 (frontend/src/components/visualization.js)
- [ ] T065 [P] 导出功能组件 (frontend/src/components/export.js)
- [ ] T066 [P] API服务层 (frontend/src/services/api.js)
- [ ] T067 [P] CSS样式文件 (frontend/static/css/styles.css)
- [ ] T084 [P] 对比可视化组件 (frontend/src/components/comparison.js)
- [ ] T085 [P] 数据集选择器组件 (frontend/src/components/dataset-selector.js)

### 单元测试完善
- [ ] T068 [P] 模型单元测试 (backend/tests/unit/test_models.py)
- [ ] T069 [P] 服务层单元测试 (backend/tests/unit/test_services.py)
- [ ] T070 [P] 工具函数单元测试 (backend/tests/unit/test_utils.py)
- [ ] T071 [P] 前端组件单元测试 (frontend/tests/unit/test_components.js)
- [ ] T086 [P] 对比可视化功能单元测试 (backend/tests/unit/test_comparison.py)

### 性能优化
- [ ] T072 大数据集内存优化
- [ ] T073 并行处理性能提升
- [ ] T074 前端渲染性能优化
- [ ] T075 缓存机制实现

### 文档和完善
- [ ] T076 [P] 更新CLAUDE.md项目说明
- [ ] T077 [P] 创建算法使用文档
- [ ] T078 [P] 完善API文档
- [ ] T079 [P] 添加示例数据和演示
- [ ] T080 运行完整测试套件验证功能

## Dependencies (依赖关系)

### 关键依赖链
1. **Setup (T001-T010)** → **Tests (T011-T025)** → **Implementation (T026-T048)**
2. **Models (T026-T029)** → **Services (T034-T037, T081)** → **API (T042-T048, T082-T083)**
3. **Algorithms (T030-T033)** → **Services (T034-T037, T081)**
4. **CLI (T038-T041)** 依赖 **Algorithms (T030-T033)**
5. **Integration (T049-T060)** → **Polish (T061-T071, T084-T086)**

### 阻塞关系
- T011-T025 (测试) 必须在 T026-T048 (实现) 之前
- T026-T029 (模型) 阻塞 T034-T037, T081 (服务)
- T030-T033 (算法) 阻塞 T038-T041 (CLI)
- T036 (可视化服务) 阻塞 T081 (对比可视化服务)
- T042-T048, T082-T083 (API) 阻塞 T061-T067, T084-T085 (前端)

## Parallel Example (并行执行示例)

### Phase 1: Setup - 并行执行
```bash
# 可以同时运行的任务
Task: "配置Python开发工具 (black, flake8, pytest, mypy)"
Task: "配置前端开发工具 (ESLint, Prettier, Jest)"
Task: "设置Docker配置 (Dockerfile, docker-compose.yml)"
Task: "配置日志系统 (logging_config.py)"
```

### Phase 2: Tests - 并行执行
```bash
# 可以同时运行的所有契约测试
Task: "契约测试 POST /api/upload 文件上传"
Task: "契约测试 GET /api/datasets/{id}/status 状态查询"
Task: "契约测试 POST /api/datasets/{id}/process 处理启动"
Task: "契约测试 GET /api/pipelines/{id}/progress 进度查询"
Task: "契约测试 GET /api/visualizations/{id} 可视化数据"
Task: "契约测试 POST /api/visualizations/{id}/export 图像导出"
Task: "契约测试 WebSocket /ws/pipelines/{id} 实时更新"

# 可以同时运行的集成测试
Task: "集成测试 完整数据处理流水线"
Task: "集成测试 文件上传到可视化流程"
Task: "集成测试 参数配置和实时更新"
Task: "集成测试 大数据集处理能力"
Task: "集成测试 错误处理和恢复"

# 可以同时运行的算法测试
Task: "算法测试 PCA处理器功能"
Task: "算法测试 t-SNE处理器功能"
Task: "算法测试 数据预处理功能"
```

### Phase 3: Core Implementation - 并行执行
```bash
# 可以同时运行的模型创建
Task: "MOF数据模型 (backend/src/models/dataset.py)"
Task: "处理流水线模型 (backend/src/models/pipeline.py)"
Task: "可视化模型 (backend/src/models/visualization.py)"
Task: "分类标签模型 (backend/src/models/category.py)"

# 可以同时运行的算法实现
Task: "PCA算法实现 (backend/src/algorithms/pca.py)"
Task: "t-SNE算法实现 (backend/src/algorithms/tsne.py)"
Task: "数据预处理算法 (backend/src/algorithms/preprocessing.py)"

# 可以同时运行的服务层
Task: "文件上传服务 (backend/src/services/upload_service.py)"
Task: "数据处理服务 (backend/src/services/processing_service.py)"
Task: "可视化服务 (backend/src/services/visualization_service.py)"
Task: "导出服务 (backend/src/services/export_service.py)"
```

## Notes (注意事项)

- [P] tasks = 不同文件，无依赖关系，可并行执行
- **必须先运行测试 (T011-T025) 并确认它们失败，然后再开始实现 (T026-T048)**
- 每个任务完成后建议提交代码
- 避免模糊的任务描述，每个任务都要指定精确的文件路径
- 确保没有两个标记为 [P] 的任务修改同一个文件

## Task Generation Rules (任务生成规则)

### 1. From Contracts (来自契约)
- contracts/api.yaml → 8个端点 → 7个契约测试任务 [P]
- 每个端点 → 对应的实现任务

### 2. From Data Model (来自数据模型)
- 5个核心实体 → 4个模型创建任务 [P]
- 实体关系 → 服务层任务

### 3. From User Stories (来自用户故事)
- 每个主要用户故事 → 集成测试任务 [P]
- quickstart.md场景 → 验证任务

### 4. From SDD Constitution (来自SDD章程)
- Library-First原则 → 算法独立实现 [P]
- CLI Interface原则 → 命令行工具 [P]
- Test-First原则 → 测试先行
- Scientific Observability → 日志和监控任务

## Validation Checklist (验证清单)

- [x] 所有契约都有对应的测试 (7个端点 → 7个契约测试)
- [x] 所有实体都有模型任务 (5个核心实体 → 4个模型文件)
- [x] 所有测试都在实现之前 (T011-T025 在 T026-T048 之前)
- [x] 并行任务真正独立 (标记 [P] 的任务都在不同文件中)
- [x] 每个任务都指定精确文件路径
- [x] 没有任务修改与另一个 [P] 任务相同的文件
- [x] 符合SDD Constitution的所有9条原则
- [x] 完整覆盖功能规格书中的27个需求
- [x] 包含性能目标和约束条件的实现

## Expected Output (预期输出)

**任务统计**: 86个任务
- Setup: 10个任务
- Tests First: 15个任务 (TDD强制要求)
- Core Implementation: 26个任务 (含对比可视化服务)
- Integration: 12个任务
- Polish: 23个任务 (含对比可视化前端组件)

**并行度**: 约60%的任务可以并行执行
- 高并行度阶段: Setup, Tests, Core Implementation
- 顺序执行阶段: Integration dependencies

**预计执行时间**:
- 并行执行: 2-3天
- 顺序执行: 4-5天
- 总计: 5-8天 (取决于团队规模)

## 新增功能：对比可视化

基于用户需求，新增了6个任务（T081-T086）来实现原始数据与筛选后数据的对比可视化功能：

**功能特点**：
- 支持两个数据集的t-SNE坐标对比显示
- 原始数据作为半透明背景，筛选数据突出显示
- 复用现有t-SNE算法，无需修改核心计算逻辑
- 提供专门的对比可视化服务和API端点

**新增任务分布**：
- T081: 对比可视化服务
- T082-T083: 对比可视化API端点
- T084-T085: 前端对比组件
- T086: 对比功能单元测试

---

**SDD Constitution Compliance**: 所有任务都严格遵循SDD Constitution的9条原则，特别是Test-First、Library-First和CLI Interface原则。