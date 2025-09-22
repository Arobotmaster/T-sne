# Claude Code Instructions: MOF数据t-SNE交互式可视化

## 项目概述

这是一个材料科学研究用的MOF（金属有机框架）数据t-SNE降维和可视化Web应用，基于SDD（Specification-Driven Development）方法论开发。

### 核心功能
- CSV数据上传和预处理
- PCA降维预处理
- t-SNE降维计算
- 交互式Web可视化
- 多格式图像导出（PNG、SVG、PDF）

### 技术架构
- **后端**: Python 3.11+ + FastAPI + scikit-learn + pandas + numpy
- **前端**: 原生JavaScript + Plotly.js + Bootstrap
- **部署**: 本地部署，支持Docker

## 项目治理

### SDD Constitution
项目遵循严格的SDD Constitution，包含9条核心原则：
1. **Library-First**: 每个算法组件都是独立可重用的库
2. **CLI Interface**: 所有数据处理库支持命令行接口
3. **Test-First**: 强制TDD，测试先行
4. **Integration-First**: 完整的端到端集成测试
5. **Scientific Observability**: 详细的科学计算日志记录
6. **Semantic Versioning**: 科学软件版本管理
7. **Simplicity**: 避免过度工程化
8. **Anti-Abstraction**: 保持科学方法透明度
9. **Web Application Integration**: 完整Web应用集成测试

### 文档结构
```
.specify/memory/constitution.md      # 项目治理章程
specs/001-md/spec.md                  # 功能规格说明
specs/001-md/plan.md                  # 实施计划
specs/001-md/research.md              # 技术研究
specs/001-md/data-model.md            # 数据模型设计
specs/001-md/contracts/api.yaml       # API契约
specs/001-md/quickstart.md            # 快速开始指南
specs/001-md/tasks.md                 # 任务列表 (由/tasks生成)
```

## 开发指南

### 代码结构
```
backend/                              # 后端Python应用
├── src/
│   ├── models/                       # 数据模型
│   ├── services/                     # 业务逻辑
│   ├── api/                          # API路由
│   ├── algorithms/                   # 算法实现 (PCA, t-SNE等)
│   └── cli/                          # 命令行接口
├── tests/
│   ├── unit/                         # 单元测试
│   ├── integration/                  # 集成测试
│   └── contract/                    # 契约测试
└── main.py                           # FastAPI应用入口

frontend/                             # 前端Web应用
├── src/
│   ├── components/                   # UI组件
│   ├── services/                     # API服务
│   └── utils/                        # 工具函数
├── static/
│   ├── css/                         # 样式文件
│   ├── js/                          # JavaScript文件
│   └── lib/                         # 第三方库
└── tests/                            # 前端测试

data/                                 # 数据文件目录
logs/                                 # 日志文件目录
docs/                                 # 项目文档
```

### 开发原则

#### 1. Library-First原则
- 每个算法（PCA、t-SNE、预处理）必须实现为独立库
- 库必须自包含、可独立测试、有完整文档
- 支持命令行调用：`python -m algorithms.pca --input data.csv --output result.json`

#### 2. Test-First原则
- 先写测试，再写实现
- 所有测试必须先失败（Red），然后实现使其通过（Green）
- 每个统计算法都有对应的测试用例和预期输出

#### 3. CLI Interface原则
- 所有数据处理库暴露CLI接口
- 支持JSON和人类可读输出格式
- 标准输入输出：`stdin/args → stdout, errors → stderr`

#### 4. Scientific Observability
- 详细记录每个处理步骤的中间结果
- 记录算法性能指标（计算时间、内存使用）
- 支持调试模式输出详细状态信息

### 编码规范

#### Python代码
```python
# 算法库示例 - backend/src/algorithms/pca.py
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.decomposition import PCA
import logging

class PCAProcessor:
    """PCA降维处理器 - 符合Library-First原则"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def fit_transform(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        执行PCA降维

        Args:
            X: 输入特征矩阵 (n_samples, n_features)

        Returns:
            Tuple: (降维后数据, 元数据)
        """
        # 记录输入信息
        self.logger.info(f"PCA input shape: {X.shape}")

        # 应用PCA
        pca = PCA(**self.config)
        X_transformed = pca.fit_transform(X)

        # 记录结果信息
        metadata = {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'n_components': pca.n_components_,
            'input_shape': X.shape,
            'output_shape': X_transformed.shape
        }

        self.logger.info(f"PCA output shape: {X_transformed.shape}")
        self.logger.info(f"Explained variance ratio: {metadata['cumulative_variance'][-1]:.3f}")

        return X_transformed, metadata
```

#### API设计
```python
# API路由示例 - backend/src/api/upload.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from ...services.upload_service import UploadService
from ...models.dataset import DatasetInfo
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload", response_model=UploadResponse)
async def upload_csv_file(file: UploadFile = File(...)):
    """
    上传CSV文件并创建数据集

    符合Constitution的CLI Interface和Scientific Observability原则
    """
    try:
        upload_service = UploadService()
        dataset_info = await upload_service.process_upload(file)

        logger.info(f"File uploaded successfully: {dataset_info.filename}")
        logger.info(f"Dataset ID: {dataset_info.dataset_id}")
        logger.info(f"Data quality score: {dataset_info.data_quality_score}")

        return UploadResponse(
            success=True,
            data=dataset_info.dict()
        )

    except ValueError as e:
        logger.error(f"Upload validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Upload processing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

#### 测试优先
```python
# 测试示例 - backend/tests/unit/test_pca.py
import pytest
import numpy as np
from backend.src.algorithms.pca import PCAProcessor

class TestPCAProcessor:
    """PCA处理器测试 - 符合Test-First原则"""

    def test_pca_basic_functionality(self):
        """测试PCA基本功能"""
        # Arrange
        X = np.random.rand(100, 50)
        config = {'n_components': 10}
        processor = PCAProcessor(config)

        # Act
        result, metadata = processor.fit_transform(X)

        # Assert
        assert result.shape == (100, 10)
        assert 'explained_variance_ratio' in metadata
        assert len(metadata['explained_variance_ratio']) == 10

    def test_pca_variance_retention(self):
        """测试PCA方差保留"""
        # Arrange
        X = np.random.rand(100, 50)
        config = {'n_components': 0.95}  # 保留95%方差
        processor = PCAProcessor(config)

        # Act
        result, metadata = processor.fit_transform(X)

        # Assert
        cumulative_variance = metadata['cumulative_variance']
        assert cumulative_variance[-1] >= 0.95
```

### 前端开发

#### JavaScript模块化
```javascript
// 前端服务示例 - frontend/src/services/api.js
class APIService {
    /** API服务类 - 符合模块化设计 */

    constructor() {
        this.baseURL = '/api';
    }

    async uploadFile(file) {
        /** 上传CSV文件 */
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${this.baseURL}/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Upload failed');
        }

        return await response.json();
    }

    async getVisualizationData(pipelineId) {
        /** 获取可视化数据 */
        const response = await fetch(`${this.baseURL}/visualizations/${pipelineId}`);
        return await response.json();
    }
}
```

#### 可视化组件
```javascript
// 可视化组件示例 - frontend/src/components/visualization.js
import Plotly from 'plotly.js-dist';
import { APIService } from '../services/api.js';

class VisualizationComponent {
    /** 可视化组件 - 使用Plotly.js */

    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.api = new APIService();
        this.config = this.getDefaultConfig();
    }

    async render(data) {
        /** 渲染t-SNE散点图 */
        const trace = this.createScatterTrace(data);
        const layout = this.createLayout(data.config);

        await Plotly.newPlot(this.container, [trace], layout, {
            responsive: true,
            displayModeBar: true,
            displaylogo: false
        });

        this.setupInteractivity();
    }

    createScatterTrace(data) {
        /** 创建散点图轨迹 */
        return {
            x: data.coordinates.map(p => p.x),
            y: data.coordinates.map(p => p.y),
            mode: 'markers',
            type: 'scatter',
            text: data.coordinates.map(p =>
                `${p.sample_id}<br>Category: ${p.category_name}`
            ),
            marker: {
                size: 8,
                color: data.coordinates.map(p => p.category_id),
                colorscale: 'Viridis',
                showscale: true,
                colorbar: {
                    title: 'Categories'
                }
            },
            name: 'MOF Samples'
        };
    }
}
```

### 命令行工具

#### CLI实现
```python
# CLI工具示例 - backend/src/cli/main.py
import argparse
import sys
import json
from ...algorithms.tsne import TSNEProcessor
from ...utils.io import load_csv, save_json

def main():
    """命令行主函数 - 符合CLI Interface原则"""
    parser = argparse.ArgumentParser(
        description='MOF数据t-SNE降维工具',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--input', '-i', required=True,
                       help='输入CSV文件路径')
    parser.add_argument('--output', '-o', required=True,
                       help='输出JSON文件路径')
    parser.add_argument('--perplexity', type=float, default=30.0,
                       help='t-SNE perplexity参数 (默认: 30)')
    parser.add_argument('--pca-components', type=int, default=50,
                       help='PCA降维维度 (默认: 50)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细输出模式')

    args = parser.parse_args()

    try:
        # 加载数据
        data, metadata = load_csv(args.input)
        if args.verbose:
            print(f"Loaded {len(data)} samples with {metadata['n_features']} features")

        # PCA降维
        pca_config = {'n_components': args.pca_components}
        pca_result = TSNEProcessor.reduce_pca(data, pca_config)

        # t-SNE降维
        tsne_config = {'perplexity': args.perplexity}
        tsne_result = TSNEProcessor.fit_transform(pca_result, tsne_config)

        # 保存结果
        result = {
            'coordinates': tsne_result.tolist(),
            'config': {
                'pca': pca_config,
                'tsne': tsne_config
            },
            'metadata': metadata
        }

        save_json(result, args.output)

        if args.verbose:
            print(f"Results saved to {args.output}")
            print(f"Final shape: {tsne_result.shape}")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
```

## 测试策略

### 测试金字塔
```
        /\
       /  \
      /    \     Integration Tests
     /      \
    /        \   Contract Tests
   /          \
  /            \  Unit Tests
 /              \
/________________\
```

### 测试覆盖要求
- **单元测试**: 90%+ 代码覆盖率
- **集成测试**: 完整数据流水线测试
- **契约测试**: API接口兼容性测试
- **性能测试**: 大数据集处理能力测试
- **端到端测试**: 完整用户流程测试

## 性能要求

### 响应时间
- 页面加载: < 3秒
- 参数更新: < 1秒
- t-SNE计算(10K样本): < 30秒
- API响应: < 500ms

### 资源使用
- 内存使用: < 2GB
- CPU使用: 合理利用多核
- 磁盘空间: 及时清理临时文件

## 部署和运维

### 本地部署
```bash
# 开发环境
python -m uvicorn backend.main:app --reload

# 生产环境
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Docker部署
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY backend/ ./backend/
COPY frontend/build/ ./frontend/build/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 调试和日志

### 日志配置
```python
# logging_config.py
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'detailed'
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'logs/application.log',
            'level': 'DEBUG',
            'formatter': 'detailed'
        }
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}
```

## 常见任务

### 添加新算法
1. 在 `backend/src/algorithms/` 创建新文件
2. 继承 `BaseAlgorithm` 基类
3. 实现 `fit_transform` 方法
4. 添加对应的CLI命令
5. 编写完整测试套件

### 修改API接口
1. 更新 `contracts/api.yaml` 契约
2. 修改对应的路由实现
3. 更新前端API调用
4. 运行契约测试验证

### 前端功能扩展
1. 在 `frontend/src/components/` 添加新组件
2. 更新 `frontend/src/services/` API服务
3. 添加对应的测试
4. 更新文档

## 重要提醒

### Constitution合规性
- 所有代码变更必须符合Constitution原则
- 提交前运行完整测试套件
- 确保Library-First和CLI Interface原则得到遵循
- 保持Scientific Observability的日志记录

### 代码质量
- 保持代码简洁明了（Anti-Abstraction原则）
- 避免过度工程化（Simplicity原则）
- 确保所有功能都有对应测试
- 保持文档同步更新

### 性能考虑
- 大数据集处理时注意内存使用
- 优化算法性能和响应时间
- 实现适当的错误处理和用户反馈
- 考虑浏览器兼容性

---

**记住**: 这个项目的核心是为材料科学研究人员提供高质量、可重现的数据可视化工具。每一行代码都应该体现科学软件的严谨性和可追溯性。