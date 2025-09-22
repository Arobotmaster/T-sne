# Quick Start Guide: MOF数据t-SNE交互式可视化

## 快速开始

本指南将帮助您快速设置和使用MOF数据t-SNE交互式可视化应用。

### 前置要求

#### 系统要求
- **操作系统**: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)
- **Python**: 3.11 或更高版本
- **内存**: 最少4GB RAM，推荐8GB以上
- **存储**: 最少1GB可用空间
- **浏览器**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+

#### Python环境准备
```bash
# 检查Python版本
python --version
# 或
python3 --version

# 推荐使用虚拟环境
python -m venv mof-viz-env
source mof-viz-env/bin/activate  # Linux/macOS
# 或
mof-viz-env\Scripts\activate     # Windows

# 升级pip
pip install --upgrade pip
```

### 安装步骤

#### 1. 克隆项目
```bash
git clone <repository-url>
cd mof-tsne-visualization
```

#### 2. 安装依赖
```bash
# 安装后端依赖
pip install -r requirements.txt

# 安装前端依赖 (如果使用npm)
cd frontend
npm install
cd ..
```

#### 3. 验证安装
```bash
# 验证后端安装
python -c "import fastapi, sklearn, pandas, numpy; print('Backend dependencies OK')"

# 验证前端文件存在
ls -la frontend/static/
```

### 启动应用

#### 方式一：开发模式
```bash
# 启动后端服务器
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# 在另一个终端启动前端开发服务器
cd frontend
npm run dev
```

#### 方式二：生产模式
```bash
# 构建前端
cd frontend
npm run build

# 启动后端 (会自动服务静态文件)
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

#### 方式三：使用Docker (推荐)
```bash
# 构建镜像
docker build -t mof-viz .

# 运行容器
docker run -p 8000:8000 -v $(pwd)/data:/app/data mof-viz
```

### 访问应用

打开浏览器访问：`http://localhost:8000`

您应该看到：
- 应用主界面
- 文件上传区域
- 示例数据链接

## 基本使用流程

### 1. 准备数据文件

#### CSV文件格式要求
```csv
mofid,category,feature1,feature2,feature3,...,DOI,Source
MOF_001,Category_A,1.23,4.56,7.89,...,10.1234/j.example,Source1
MOF_002,Category_B,2.34,5.67,8.90,...,10.2345/j.example,Source2
...
```

#### 数据要求
- **必须包含**:
  - 唯一标识符列 (如 `mofid`)
  - 分类标签列 (如 `category`)
  - 数值特征列 (至少2列)
- **可选包含**:
  - 描述信息列 (如 `DOI`, `Source`)
- **文件大小**: 最大100MB
- **样本数量**: 推荐100-10,000个样本

#### 示例数据
下载示例数据：
```bash
curl -O https://example.com/mof-sample-data.csv
```

### 2. 上传数据

1. 点击主界面的"选择文件"按钮
2. 选择您的CSV文件
3. 系统自动检测格式并验证数据
4. 等待上传完成 (显示数据预览)

### 3. 配置处理参数

#### PCA参数
- **降维维度**: 50 (默认) 或 选择方差保留率 (95%)
- **白化**: 关闭 (默认)
- **随机种子**: 42 (可重复结果)

#### t-SNE参数
- **Perplexity**: 30 (默认)，可选10, 30, 50
- **学习率**: 200 (默认)
- **迭代次数**: 1000 (默认)
- **距离度量**: 欧氏距离 (默认)

### 4. 启动处理

1. 点击"开始处理"按钮
2. 观察进度条和状态更新
3. 等待处理完成 (通常10秒-5分钟，取决于数据大小)
4. 自动跳转到可视化界面

### 5. 探索可视化

#### 基本交互
- **缩放**: 鼠标滚轮或双指缩放
- **平移**: 鼠标拖拽
- **悬停**: 鼠标悬停查看详细信息
- **选择**: 点击数据点进行选择

#### 自定义外观
1. 点击"设置"按钮打开配置面板
2. 调整颜色方案：
   - 为每个类别选择颜色
   - 调整透明度
3. 配置标记样式：
   - 选择形状 (圆形、方形、三角形等)
   - 调整大小
4. 修改图表标题和轴标签

### 6. 导出结果

#### 导出设置
1. 点击"导出"按钮
2. 选择格式 (PNG, SVG, PDF)
3. 配置参数：
   - 图像尺寸
   - 分辨率 (DPI)
   - 背景颜色
   - 文件名

#### 导出选项
- **PNG**: 位图格式，适合网页展示
- **SVG**: 矢量格式，适合论文发表
- **PDF**: 文档格式，适合打印

## 高级功能

### 批量处理
```bash
# 使用CLI处理多个文件
python -m backend.cli.batch_process --input-dir ./data/ --output-dir ./results/

# 查看CLI帮助
python -m backend.cli --help
```

### 参数优化
```python
# Python API示例
from backend.optimization import optimize_tsne_params

best_params = optimize_tsne_params(
    data_path="your_data.csv",
    target_metric="trustworthiness",
    max_trials=50
)
```

### 自定义算法
```python
# 实现自定义降维算法
from backend.algorithms import CustomDimensionalityReduction

class CustomAlgorithm(CustomDimensionalityReduction):
    def fit_transform(self, X, y=None):
        # 实现您的算法
        return reduced_X
```

## 故障排除

### 常见问题

#### 文件上传失败
**问题**: 文件格式不支持
**解决**: 确保文件为CSV格式，使用UTF-8编码

**问题**: 文件过大
**解决**: 压缩文件或减少样本数量

#### 处理失败
**问题**: 内存不足
**解决**:
- 减少数据量
- 增加系统内存
- 使用分块处理

**问题**: 算法不收敛
**解决**:
- 调整perplexity值
- 增加迭代次数
- 检查数据质量

#### 可视化问题
**问题**: 图表显示异常
**解决**: 刷新页面或重新处理数据

**问题**: 交互功能失效
**解决**: 更新浏览器或尝试其他浏览器

### 性能优化

#### 大数据集处理
- 使用数据采样
- 启用分块处理
- 优化算法参数

#### 浏览器性能
- 关闭不必要的标签页
- 清理浏览器缓存
- 使用硬件加速

### 日志和调试

#### 查看日志
```bash
# 查看应用日志
tail -f logs/application.log

# 查看错误日志
tail -f logs/error.log
```

#### 调试模式
```bash
# 启用调试模式
export DEBUG=1
python -m uvicorn backend.main:app --reload
```

## API使用

### 基本API调用
```bash
# 上传文件
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@your_data.csv"

# 检查处理状态
curl "http://localhost:8000/api/pipelines/{pipeline_id}/status"

# 获取可视化数据
curl "http://localhost:8000/api/visualizations/{pipeline_id}"
```

### Python客户端
```python
import requests

# 上传文件
with open('data.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/upload',
        files={'file': f}
    )
    dataset_id = response.json()['data']['dataset_id']

# 启动处理
response = requests.post(
    f'http://localhost:8000/api/datasets/{dataset_id}/process',
    json={
        'tsne_config': {'perplexity': 30},
        'pca_config': {'n_components': 50}
    }
)
```

## 扩展和定制

### 添加新的降维算法
1. 在 `backend/algorithms/` 目录创建新算法类
2. 继承 `BaseDimensionalityReduction` 基类
3. 实现 `fit_transform` 方法
4. 在配置中注册新算法

### 自定义可视化
1. 修改 `frontend/static/js/visualization.js`
2. 添加新的可视化类型
3. 更新API接口支持新类型

### 国际化支持
1. 在 `frontend/static/i18n/` 目录添加语言文件
2. 修改前端代码支持多语言
3. 更新后端API支持语言参数

## 技术支持

### 获取帮助
- **文档**: 查看项目 `docs/` 目录
- **Issues**: 在GitHub仓库提交问题
- **讨论**: 参与GitHub Discussions

### 报告问题
提交问题时请包含：
1. 问题描述
2. 复现步骤
3. 错误信息
4. 系统环境信息
5. 数据样本 (如果可能)

### 贡献代码
1. Fork 项目仓库
2. 创建功能分支
3. 提交代码更改
4. 创建Pull Request
5. 等待代码审查

---

## 下一步

恭喜！您已经成功设置了MOF数据t-SNE交互式可视化应用。接下来您可以：

1. 上传您的MOF数据开始分析
2. 探索不同的参数配置
3. 导出高质量的可视化图像
4. 阅读完整文档了解更多高级功能
5. 参与社区讨论和贡献

祝您研究顺利！