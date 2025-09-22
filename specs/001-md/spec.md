# Feature Specification: MOF数据t-SNE交互式可视化

**Feature Branch**: `001-md`
**Created**: 2025-09-19
**Status**: Draft
**Input**: User description: "MOF数据t-SNE交互式可视化需求纪要"

---

## ⚡ Quick Guidelines
- ✅ Focus on WHAT users need and WHY
- ❌ Avoid HOW to implement (no tech stack, APIs, code structure)
- 👥 Written for business stakeholders, not developers

---

## User Scenarios & Testing

### Primary User Story
作为材料科学研究人员，我希望能够上传包含四种MOF分类的CSV数据，通过t-SNE降维算法进行可视化，并实时调整可视化参数，以便探索不同MOF类别的分布模式和特征关系。

### Acceptance Scenarios
1. **Given** 用户有一个包含MOF数据的CSV文件，**When** 用户上传文件并选择处理参数，**Then** 系统必须生成二维t-SNE散点图并显示四种MOF类别的分布

2. **Given** t-SNE可视化图表已生成，**When** 用户悬停在数据点上，**Then** 系统必须显示该MOF的详细信息（如MOF ID、DOI等）

3. **Given** 图表已显示，**When** 用户调整颜色方案或标记样式，**Then** 系统必须实时更新图表外观而无需重新计算

4. **Given** 用户对当前可视化效果满意，**When** 用户点击导出按钮并选择格式和参数，**Then** 系统必须生成与当前显示状态完全一致的图像文件

### Edge Cases
- What happens when CSV文件包含缺失值或异常值？
- How does system handle 超过内存容量的大数据集？
- What happens when 用户选择的t-SNE参数导致计算失败？
- How does system handle 重复的MOF ID或无效的分类标签？
- What happens when 浏览器不支持某些交互功能？

## Requirements

### Functional Requirements

#### 数据处理要求
- **FR-001**: System MUST 加载用户提供的CSV文件并自动识别数据类型
- **FR-002**: System MUST 分离描述性数据（如mofid、DOI、Source）与数值特征
- **FR-003**: System MUST 识别和提取四种MOF类别标签列
- **FR-004**: System MUST 对数值特征进行预处理（缺失值处理、标准化）
- **FR-005**: System MUST 应用PCA降维预处理（支持可配置的维度或方差保留率）
- **FR-006**: System MUST 执行t-SNE降维计算并生成二维坐标
- **FR-007**: System MUST 支持多组不同的Perplexity值（如10, 30, 50）进行计算比较

#### 可视化功能要求
- **FR-008**: System MUST 根据二维坐标和类别标签生成彩色散点图
- **FR-009**: System MUST 为四种不同的MOF类别使用可区分的颜色
- **FR-010**: System MUST 在鼠标悬停时显示数据点的详细信息
- **FR-011**: System MUST 提供实时交互控件来调整可视化外观

#### 交互控制要求
- **FR-012**: Users MUST be able to 自定义每个类别的颜色方案
- **FR-013**: Users MUST be able to 调整图表标题、坐标轴标签、图例的字体大小和粗细
- **FR-014**: Users MUST be able to 为每个类别选择不同的标记形状（圆形、方形、十字形等）
- **FR-015**: Users MUST be able to 调整t-SNE的Perplexity等参数并触发重新计算
- **FR-016**: System MUST 提供参数重置功能以恢复默认设置

#### 导出功能要求
- **FR-017**: System MUST 支持导出当前可视化为PNG格式
- **FR-018**: System MUST 支持导出当前可视化为SVG格式
- **FR-019**: Users MUST be able to 配置导出图像的分辨率/尺寸
- **FR-020**: Users MUST be able to 选择导出图像的背景（透明或白色）
- **FR-021**: Users MUST be able to 自定义导出文件的名称
- **FR-022**: System MUST 确保导出图像与当前交互状态（颜色、标记、标题等）完全一致

#### 性能和可用性要求
- **FR-023**: System MUST 在3秒内完成初始页面加载
- **FR-024**: System MUST 在1秒内响应参数调整并更新显示
- **FR-025**: System MUST 处理至少10,000个数据点的数据集
- **FR-026**: System MUST 提供处理进度指示器
- **FR-027**: System MUST 在计算失败时提供清晰的错误信息

### Key Entities

#### MOF数据实体
- **MOF样本**: 代表单个金属有机框架，包含标识信息、分类标签和数值特征
  - 属性：唯一标识符(mofid)、分类标签、数值特征向量、描述信息(DOI、Source)
  - 关系：属于某个MOF类别，参与t-SNE计算

#### 数据处理流水线实体
- **原始数据**: 用户上传的CSV文件内容
  - 属性：文件路径、数据行数、列名、数据类型
  - 关系：被预处理组件读取和处理

- **预处理数据**: 清洗和标准化后的数值数据
  - 属性：特征矩阵、标签向量、数据质量指标
  - 关系：来源于原始数据，输入到PCA算法

- **PCA结果**: 主成分分析降维后的数据
  - 属性：主成分矩阵、解释方差比、降维后数据
  - 关系：来源于预处理数据，输入到t-SNE算法

- **t-SNE结果**: 最终二维可视化坐标
  - 属性：二维坐标数组、收敛信息、计算参数
  - 关系：来源于PCA结果，用于可视化显示

#### 可视化配置实体
- **颜色方案**: 四个类别的颜色配置
  - 属性：类别颜色映射、默认方案、自定义方案
  - 关系：应用于散点图显示

- **标记样式**: 数据点显示样式配置
  - 属性：形状映射、大小设置、透明度
  - 关系：应用于散点图显示

- **导出配置**: 图像导出参数设置
  - 属性：文件格式、分辨率、背景色、文件名
  - 关系：控制导出图像的生成

#### 计算参数实体
- **t-SNE参数**: 降维算法的计算参数
  - 属性：perplexity值、迭代次数、学习率、随机种子
  - 关系：控制t-SNE计算过程

- **PCA参数**: 主成分分析参数
  - 属性：目标维度、方差保留率、标准化方法
  - 关系：控制PCA计算过程

---

## Review & Acceptance Checklist

### Content Quality
- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness
- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

---

## Execution Status

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [x] Review checklist passed

---