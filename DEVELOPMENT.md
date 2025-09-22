# MOF数据t-SNE可视化应用 - 开发指南

## 🚀 快速开始

### 前置要求
- **Python**: 3.11+
- **Node.js**: 16+
- **操作系统**: Linux, macOS, Windows

### 一键设置开发环境

```bash
# Linux/macOS
./setup.sh

# Windows
setup.bat
```

手动设置请参考下面的详细步骤。

## 📁 项目结构

```
.
├── backend/                    # Python后端应用
│   ├── src/
│   │   ├── api/              # FastAPI路由
│   │   ├── algorithms/       # 算法实现 (PCA, t-SNE)
│   │   ├── cli/             # 命令行工具
│   │   ├── config/          # 配置文件
│   │   ├── models/          # 数据模型
│   │   ├── services/        # 业务逻辑
│   │   └── utils/           # 工具函数
│   ├── tests/               # 测试套件
│   ├── main.py             # FastAPI应用入口
│   ├── requirements.txt    # Python依赖
│   └── pyproject.toml      # 项目配置
├── frontend/                # 前端Web应用
│   ├── src/
│   │   ├── components/     # React组件
│   │   ├── services/       # API服务
│   │   └── utils/          # 工具函数
│   ├── static/             # 静态资源
│   │   ├── css/           # CSS样式
│   │   ├── js/            # JavaScript
│   │   └── lib/           # 第三方库
│   ├── tests/              # 前端测试
│   └── package.json        # Node.js依赖
├── data/                   # 数据目录
├── logs/                   # 日志目录
├── docs/                   # 文档目录
├── Dockerfile             # Docker配置
├── docker-compose.yml     # Docker编排
├── setup.sh               # Linux/macOS设置脚本
├── setup.bat              # Windows设置脚本
└── .env.example           # 环境变量示例
```

## 🛠️ 开发环境设置

### 1. Python环境设置

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 安装依赖
pip install -r backend/requirements.txt
pip install -e backend/
```

### 2. 前端环境设置

```bash
# 进入前端目录
cd frontend

# 安装Node.js依赖
npm install

# 返回根目录
cd ..
```

### 3. 环境配置

```bash
# 复制环境变量文件
cp .env.example .env

# 编辑 .env 文件，根据需要修改配置
nano .env
```

### 4. 创建必要目录

```bash
mkdir -p data/uploads logs results docs/api
```

## 🚀 运行应用

### 启动后端服务

```bash
# 激活Python虚拟环境
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate    # Windows

# 启动FastAPI开发服务器
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 启动前端服务

```bash
# 进入前端目录
cd frontend

# 启动开发服务器
npm run dev
```

或者使用Python内置HTTP服务器：

```bash
cd frontend
python -m http.server 3000
```

### 访问应用

- **前端应用**: http://localhost:3000
- **后端API**: http://localhost:8000
- **API文档**: http://localhost:8000/docs
- **API备用文档**: http://localhost:8000/redoc

## 🧪 运行测试

### Python测试

```bash
# 激活虚拟环境后运行所有测试
pytest

# 运行特定测试类型
pytest tests/unit/      # 单元测试
pytest tests/integration/ # 集成测试
pytest tests/contract/   # 契约测试

# 生成覆盖率报告
pytest --cov=src --cov-report=html
```

### 前端测试

```bash
cd frontend
npm test
```

### 代码质量检查

```bash
# Python代码格式化
black backend/src/

# Python代码检查
flake8 backend/src/

# Python类型检查
mypy backend/src/

# 前端代码检查
cd frontend
npm run lint
```

## 🐳 Docker部署

### 本地Docker运行

```bash
# 构建并启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 单独构建后端镜像

```bash
# 构建镜像
docker build -t mof-tsne-backend .

# 运行容器
docker run -p 8000:8000 -v $(pwd)/data:/app/data -v $(pwd)/logs:/app/logs mof-tsne-backend
```

## 📝 开发指南

### 添加新算法

1. **创建算法文件**:
   ```python
   # backend/src/algorithms/new_algorithm.py
   from typing import Dict, Any, Tuple
   import numpy as np
   from ..algorithms.base import BaseAlgorithm

   class NewAlgorithm(BaseAlgorithm):
       def __init__(self, config: Dict[str, Any]):
           super().__init__(config)

       def fit_transform(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
           # 实现算法逻辑
           return result, metadata
   ```

2. **添加CLI命令**:
   ```python
   # backend/src/cli/new_algorithm_command.py
   import argparse
   from ..algorithms.new_algorithm import NewAlgorithm

   def main():
       parser = argparse.ArgumentParser(description='新算法工具')
       parser.add_argument('--input', required=True)
       parser.add_argument('--output', required=True)
       args = parser.parse_args()

       # 实现CLI逻辑
   ```

3. **编写测试**:
   ```python
   # backend/tests/unit/test_new_algorithm.py
   import pytest
   import numpy as np
   from backend.src.algorithms.new_algorithm import NewAlgorithm

   class TestNewAlgorithm:
       def test_basic_functionality(self):
           # 测试算法基本功能
           pass
   ```

### 添加新的API端点

1. **定义API契约**:
   ```yaml
   # specs/001-md/contracts/api.yaml
   /api/new-endpoint:
     post:
       summary: 新端点描述
       requestBody:
         required: true
         content:
           application/json:
             schema:
               type: object
               properties:
                 # 定义请求体
       responses:
         '200':
           description: 成功响应
   ```

2. **实现路由**:
   ```python
   # backend/src/api/new_endpoint.py
   from fastapi import APIRouter, HTTPException
   from ...services.new_service import NewService

   router = APIRouter()

   @router.post("/new-endpoint")
   async def new_endpoint(request_data: RequestModel):
       # 实现端点逻辑
       return response
   ```

3. **更新主路由**:
   ```python
   # backend/src/api/__init__.py
   from .new_endpoint import router as new_router

   api_router = APIRouter()
   api_router.include_router(new_router, prefix="/api", tags=["new"])
   ```

### 添加新的前端组件

1. **创建组件**:
   ```javascript
   // frontend/src/components/NewComponent.js
   class NewComponent {
       constructor(containerId) {
           this.container = document.getElementById(containerId);
           this.init();
       }

       init() {
           // 初始化组件
       }

       render(data) {
           // 渲染组件
       }
   }
   ```

2. **注册组件**:
   ```javascript
   // frontend/static/js/app.js
   import { NewComponent } from '../src/components/NewComponent.js';

   // 在应用初始化时注册组件
   ```

## 🔧 配置说明

### 环境变量

主要环境变量说明：

- `DEBUG`: 是否启用调试模式
- `DATABASE_URL`: 数据库连接字符串
- `REDIS_URL`: Redis连接字符串
- `MAX_UPLOAD_SIZE`: 最大文件上传大小
- `LOG_LEVEL`: 日志级别 (DEBUG, INFO, WARNING, ERROR)
- `ALLOWED_ORIGINS`: 允许的CORS源

### 算法参数配置

t-SNE算法参数：
- `perplexity`: 混乱度 (5-50)
- `n_components`: 输出维度 (通常为2)
- `learning_rate`: 学习率 (10-1000)
- `n_iter`: 迭代次数 (250-1000)

PCA算法参数：
- `n_components`: 主成分数量
- `random_state`: 随机种子
- `whiten`: 是否白化

## 🐛 调试指南

### Python调试

```bash
# 启用调试模式
export DEBUG=true
python -m uvicorn backend.main:app --reload

# 使用pdb调试
python -m pdb -c continue backend/main.py
```

### 前端调试

1. **浏览器开发者工具**:
   - F12 打开开发者工具
   - Console 查看JavaScript错误
   - Network 查看API请求

2. **启用详细日志**:
   ```javascript
   // 在浏览器控制台中
   localStorage.setItem('debug', 'mof-viz:*');
   ```

### 常见问题

1. **依赖冲突**:
   ```bash
   # 检查依赖版本
   pip list

   # 升级/降级特定包
   pip install package==version
   ```

2. **端口占用**:
   ```bash
   # 查看端口占用
   netstat -tlnp | grep :8000

   # 终止进程
   kill -9 <PID>
   ```

3. **权限问题**:
   ```bash
   # 检查文件权限
   ls -la data/ logs/

   # 修改权限
   chmod 755 data/ logs/
   ```

## 📚 相关资源

- [FastAPI文档](https://fastapi.tiangolo.com/)
- [Plotly.js文档](https://plotly.com/javascript/)
- [scikit-learn文档](https://scikit-learn.org/stable/)
- [Bootstrap文档](https://getbootstrap.com/docs/)
- [Pytest文档](https://docs.pytest.org/)

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

请确保：
- 所有测试通过
- 代码符合项目风格
- 添加了必要的文档
- 遵循SDD Constitution原则