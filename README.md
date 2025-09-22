# MOF数据t-SNE交互式可视化

为材料科学研究人员提供的MOF（金属有机框架）数据t-SNE降维和可视化Web应用。

## 功能特性

- 📊 **CSV数据上传**: 支持MOF数据的CSV文件上传和验证
- 🔧 **数据预处理**: PCA降维预处理，支持可配置参数
- 📈 **t-SNE降维**: 高维数据降维到二维空间
- 🎯 **交互式可视化**: 基于Plotly.js的交互式散点图
- 📥 **多格式导出**: 支持PNG、SVG、PDF格式图像导出
- ⚡ **实时反馈**: WebSocket实时处理进度更新
- 🖥️ **命令行工具**: 完整的CLI接口支持

## 技术架构

### 后端 (Python)
- **框架**: FastAPI
- **数据科学**: scikit-learn, pandas, numpy
- **算法**: PCA, t-SNE
- **测试**: pytest

### 前端 (JavaScript)
- **可视化**: Plotly.js
- **UI框架**: Bootstrap
- **架构**: 原生JavaScript + 模块化设计

## 快速开始

### 环境要求
- Python 3.11+
- 现代Web浏览器

### 安装和运行

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd mof-tsne-visualization
   ```

2. **安装依赖**
   ```bash
   # 后端依赖
   cd backend
   pip install -r requirements.txt

   # 前端依赖
   cd ../frontend
   npm install
   ```

3. **启动应用**
   ```bash
   # 开发模式
   cd backend
   python -m uvicorn main:app --reload

   # 或使用Docker
   docker-compose up
   ```

4. **访问应用**
   打开浏览器访问: http://localhost:8000

## 使用方法

### 数据格式
CSV文件应包含以下列：
- `mofid`: MOF唯一标识符
- `category`: 分类标签 (4种MOF类别之一)
- 数值特征列 (至少2列)
- 可选: `DOI`, `Source` 等描述信息

### 基本流程
1. 上传CSV数据文件
2. 配置PCA和t-SNE参数
3. 启动数据处理
4. 探索交互式可视化
5. 导出结果图像

## 命令行工具

```bash
# PCA降维
python -m backend.src.cli.pca_command --input data.csv --output pca_result.json

# t-SNE降维
python -m backend.src.cli.tsne_command --input pca_result.json --output tsne_result.json

# 批处理
python -m backend.src.cli.batch_command --input-dir ./data/ --output-dir ./results/
```

## 开发

### 项目结构
```
backend/
├── src/
│   ├── models/          # 数据模型
│   ├── services/        # 业务逻辑
│   ├── api/             # API路由
│   ├── algorithms/      # 算法实现
│   └── cli/             # 命令行工具
└── tests/               # 测试文件

frontend/
├── src/
│   ├── components/      # UI组件
│   ├── services/        # API服务
│   └── utils/           # 工具函数
└── static/              # 静态文件
```

### 运行测试
```bash
# 后端测试
cd backend
pytest

# 前端测试
cd frontend
npm test
```

## SDD Constitution

本项目遵循SDD (Specification-Driven Development) Constitution原则：

1. **Library-First**: 每个算法都是独立可重用的库
2. **CLI Interface**: 所有数据处理库支持命令行接口
3. **Test-First**: 强制TDD，测试先行
4. **Integration-First**: 完整的端到端集成测试
5. **Scientific Observability**: 详细的科学计算日志记录
6. **Semantic Versioning**: 科学软件版本管理
7. **Simplicity**: 避免过度工程化
8. **Anti-Abstraction**: 保持科学方法透明度
9. **Web Application Integration**: 完整Web应用集成测试

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

如有问题，请通过GitHub Issues联系我们。# T-sne
