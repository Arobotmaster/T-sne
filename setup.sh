#!/bin/bash

# MOF数据t-SNE可视化应用 - 开发环境设置脚本
#
# 此脚本会自动设置完整的开发环境，包括Python依赖、前端依赖和开发工具
#
# 使用方法: ./setup.sh [--dev|--prod|--test]

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 显示欢迎信息
echo "======================================================"
echo "MOF数据t-SNE可视化应用 - 开发环境设置"
echo "======================================================"

# 检查Python版本
check_python() {
    log_info "检查Python环境..."

    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 未安装。请先安装Python 3.11或更高版本。"
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    REQUIRED_VERSION="3.11"

    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
        log_error "Python版本过低。需要Python $REQUIRED_VERSION+，当前版本: $PYTHON_VERSION"
        exit 1
    fi

    log_success "Python版本检查通过: $PYTHON_VERSION"
}

# 检查Node.js版本
check_nodejs() {
    log_info "检查Node.js环境..."

    if ! command -v node &> /dev/null; then
        log_error "Node.js 未安装。请先安装Node.js 16或更高版本。"
        exit 1
    fi

    NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
    REQUIRED_NODE_VERSION="16"

    if [ "$NODE_VERSION" -lt "$REQUIRED_NODE_VERSION" ]; then
        log_error "Node.js版本过低。需要Node.js $REQUIRED_NODE_VERSION+，当前版本: $(node -v)"
        exit 1
    fi

    log_success "Node.js版本检查通过: $(node -v)"
}

# 设置Python虚拟环境
setup_python_env() {
    log_info "设置Python虚拟环境..."

    if [ ! -d "venv" ]; then
        python3 -m venv venv
        log_success "Python虚拟环境创建成功"
    else
        log_info "Python虚拟环境已存在"
    fi

    # 激活虚拟环境
    source venv/bin/activate

    # 升级pip
    pip install --upgrade pip setuptools wheel

    # 安装Python依赖
    log_info "安装Python依赖..."
    pip install -r backend/requirements.txt

    # 安装开发依赖
    pip install -e backend/

    log_success "Python环境设置完成"
}

# 设置前端环境
setup_frontend_env() {
    log_info "设置前端环境..."

    cd frontend

    # 安装Node.js依赖
    log_info "安装前端Node.js依赖..."
    npm install

    # 创建必要的目录
    mkdir -p static/lib

    # 下载外部库文件（如果不存在）
    if [ ! -f "static/lib/plotly-latest.min.js" ]; then
        log_info "下载Plotly.js..."
        curl -o static/lib/plotly-latest.min.js https://cdn.plot.ly/plotly-latest.min.js
    fi

    if [ ! -f "static/lib/bootstrap.min.css" ]; then
        log_info "下载Bootstrap CSS..."
        curl -o static/lib/bootstrap.min.css https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css
    fi

    cd ..
    log_success "前端环境设置完成"
}

# 创建必要的目录
create_directories() {
    log_info "创建项目目录..."

    mkdir -p data/uploads
    mkdir -p logs
    mkdir -p results
    mkdir -p docs/api

    log_success "项目目录创建完成"
}

# 创建环境配置文件
create_env_files() {
    log_info "创建环境配置文件..."

    # 创建环境变量示例文件
    cat > .env.example << 'EOF'
# 应用配置
APP_NAME=MOF数据t-SNE可视化
APP_VERSION=1.0.0
DEBUG=true

# 数据库配置（如果需要）
DATABASE_URL=sqlite:///./data/mof_viz.db

# Redis配置（如果需要）
REDIS_URL=redis://localhost:6379/0

# 安全配置
SECRET_KEY=your-secret-key-here

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# 文件上传配置
MAX_UPLOAD_SIZE=104857600  # 100MB
ALLOWED_EXTENSIONS=csv

# CORS配置
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000
EOF

    # 如果.env文件不存在，则复制示例文件
    if [ ! -f ".env" ]; then
        cp .env.example .env
        log_success "创建.env文件（基于.env.example）"
    fi

    # 创建开发配置文件
    cat > backend/src/config/development.py << 'EOF'
"""
开发环境配置
"""

import os
from .settings import *

# 开发环境特有配置
DEBUG = True
LOG_LEVEL = "DEBUG"

# 开发数据库配置
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/mof_viz_dev.db")

# 允许的CORS源
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000"
]

# 开发时使用更详细的日志
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        },
        "simple": {
            "format": "%(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "detailed",
            "filename": "logs/development.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": False
        }
    }
}
EOF

    log_success "环境配置文件创建完成"
}

# 验证开发工具
verify_tools() {
    log_info "验证开发工具..."

    # 激活Python虚拟环境
    source venv/bin/activate

    # 检查Python包
    python -c "import fastapi; print('FastAPI:', fastapi.__version__)"
    python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
    python -c "import pandas; print('pandas:', pandas.__version__)"
    python -c "import plotly; print('plotly:', plotly.__version__)"

    # 检查代码格式化工具
    if command -v black &> /dev/null; then
        log_success "Black 代码格式化工具可用"
    fi

    if command -v flake8 &> /dev/null; then
        log_success "Flake8 代码检查工具可用"
    fi

    if command -v mypy &> /dev/null; then
        log_success "MyPy 类型检查工具可用"
    fi

    # 检查测试工具
    if command -v pytest &> /dev/null; then
        log_success "Pytest 测试框架可用"
    fi

    # 检查前端工具
    cd frontend
    if command -v npm &> /dev/null; then
        log_success "NPM 包管理器可用"
    fi

    if [ -f "node_modules/.bin/jest" ]; then
        log_success "Jest 测试框架可用"
    fi

    if [ -f "node_modules/.bin/eslint" ]; then
        log_success "ESLint 代码检查工具可用"
    fi

    cd ..

    log_success "开发工具验证完成"
}

# 运行测试
run_tests() {
    log_info "运行测试套件..."

    # 激活Python虚拟环境
    source venv/bin/activate

    # 运行Python测试
    log_info "运行Python测试..."
    cd backend
    python -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing
    cd ..

    # 运行前端测试
    log_info "运行前端测试..."
    cd frontend
    npm test
    cd ..

    log_success "测试运行完成"
}

# 构建项目
build_project() {
    log_info "构建项目..."

    # 激活Python虚拟环境
    source venv/bin/activate

    # 构建后端
    log_info "构建后端..."
    cd backend
    python -m pip install -e .
    cd ..

    # 构建前端
    log_info "构建前端..."
    cd frontend
    npm run build
    cd ..

    log_success "项目构建完成"
}

# 显示使用说明
show_usage() {
    cat << EOF
使用方法:
    ./setup.sh [选项]

选项:
    --help, -h          显示此帮助信息
    --dev, -d           设置开发环境（默认）
    --prod, -p          设置生产环境
    --test, -t          设置测试环境
    --verify, -v        验证当前环境
    --test-run, -tr     运行测试
    --build, -b         构建项目

环境设置完成后，您可以:
    1. 启动后端服务: source venv/bin/activate && python -m uvicorn backend.main:app --reload
    2. 启动前端服务: cd frontend && npm run dev
    3. 运行测试: ./setup.sh --test-run

EOF
}

# 主函数
main() {
    local MODE="dev"

    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                show_usage
                exit 0
                ;;
            --dev|-d)
                MODE="dev"
                shift
                ;;
            --prod|-p)
                MODE="prod"
                shift
                ;;
            --test|-t)
                MODE="test"
                shift
                ;;
            --verify|-v)
                verify_tools
                exit 0
                ;;
            --test-run|-tr)
                run_tests
                exit 0
                ;;
            --build|-b)
                build_project
                exit 0
                ;;
            *)
                log_error "未知选项: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    log_info "开始设置 $MODE 环境..."

    # 执行设置步骤
    check_python
    check_nodejs
    setup_python_env
    setup_frontend_env
    create_directories
    create_env_files
    verify_tools

    echo "======================================================"
    log_success "开发环境设置完成！"
    echo ""
    echo "下一步操作:"
    echo "1. 激活Python环境: source venv/bin/activate"
    echo "2. 启动后端服务: python -m uvicorn backend.main:app --reload"
    echo "3. 启动前端服务: cd frontend && npm run dev"
    echo "4. 运行测试: ./setup.sh --test-run"
    echo ""
    echo "访问应用: http://localhost:8000"
    echo "API文档: http://localhost:8000/docs"
    echo "======================================================"
}

# 运行主函数
main "$@"