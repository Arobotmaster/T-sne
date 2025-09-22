@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM MOF数据t-SNE可视化应用 - Windows开发环境设置脚本
REM 使用方法: setup.bat [--dev|--prod|--test]

echo ======================================================
echo MOF数据t-SNE可视化应用 - 开发环境设置
echo ======================================================

REM 获取脚本目录
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM 默认模式
set "MODE=dev"

REM 解析命令行参数
:parse_args
if "%~1"=="" goto :main
if "%~1"=="--help" goto :show_help
if "%~1"=="-h" goto :show_help
if "%~1"=="--dev" set "MODE=dev" & shift & goto :parse_args
if "%~1"=="-d" set "MODE=dev" & shift & goto :parse_args
if "%~1"=="--prod" set "MODE=prod" & shift & goto :parse_args
if "%~1"=="-p" set "MODE=prod" & shift & goto :parse_args
if "%~1"=="--test" set "MODE=test" & shift & goto :parse_args
if "%~1"=="-t" set "MODE=test" & shift & goto :parse_args
if "%~1"=="--verify" goto :verify_tools
if "%~1"=="-v" goto :verify_tools
if "%~1"=="--test-run" goto :run_tests
if "%~1"=="-tr" goto :run_tests
if "%~1"=="--build" goto :build_project
if "%~1"=="-b" goto :build_project
echo [ERROR] 未知选项: %~1
goto :show_help

:main
echo [INFO] 开始设置 %MODE% 环境...

REM 检查Python
call :check_python
if errorlevel 1 exit /b 1

REM 检查Node.js
call :check_nodejs
if errorlevel 1 exit /b 1

REM 设置Python虚拟环境
call :setup_python_env
if errorlevel 1 exit /b 1

REM 设置前端环境
call :setup_frontend_env
if errorlevel 1 exit /b 1

REM 创建目录
call :create_directories
if errorlevel 1 exit /b 1

REM 创建配置文件
call :create_env_files
if errorlevel 1 exit /b 1

REM 验证工具
call :verify_tools
if errorlevel 1 exit /b 1

echo ======================================================
echo [SUCCESS] 开发环境设置完成！
echo.
echo 下一步操作:
echo 1. 激活Python环境: venv\Scripts\activate
echo 2. 启动后端服务: python -m uvicorn backend.main:app --reload
echo 3. 启动前端服务: cd frontend ^&^& npm run dev
echo 4. 运行测试: setup.bat --test-run
echo.
echo 访问应用: http://localhost:8000
echo API文档: http://localhost:8000/docs
echo ======================================================
goto :eof

:check_python
echo [INFO] 检查Python环境...

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 未安装。请先安装Python 3.11或更高版本。
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set "PYTHON_VERSION=%%i"

echo [SUCCESS] Python版本检查通过: %PYTHON_VERSION%
goto :eof

:check_nodejs
echo [INFO] 检查Node.js环境...

node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js 未安装。请先安装Node.js 16或更高版本。
    exit /b 1
)

for /f "tokens=1" %%i in ('node --version 2^>^&1') do set "NODE_VERSION=%%i"

echo [SUCCESS] Node.js版本检查通过: %NODE_VERSION%
goto :eof

:setup_python_env
echo [INFO] 设置Python虚拟环境...

if not exist "venv" (
    python -m venv venv
    echo [SUCCESS] Python虚拟环境创建成功
) else (
    echo [INFO] Python虚拟环境已存在
)

REM 激活虚拟环境
call venv\Scripts\activate

REM 升级pip
python -m pip install --upgrade pip setuptools wheel

REM 安装Python依赖
echo [INFO] 安装Python依赖...
pip install -r backend\requirements.txt

echo [SUCCESS] Python环境设置完成
goto :eof

:setup_frontend_env
echo [INFO] 设置前端环境...

cd frontend

REM 安装Node.js依赖
echo [INFO] 安装前端Node.js依赖...
npm install

REM 创建必要的目录
if not exist "static\lib" mkdir static\lib

REM 下载外部库文件
if not exist "static\lib\plotly-latest.min.js" (
    echo [INFO] 下载Plotly.js...
    powershell -Command "Invoke-WebRequest -Uri 'https://cdn.plot.ly/plotly-latest.min.js' -OutFile 'static\lib\plotly-latest.min.js'"
)

if not exist "static\lib\bootstrap.min.css" (
    echo [INFO] 下载Bootstrap CSS...
    powershell -Command "Invoke-WebRequest -Uri 'https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css' -OutFile 'static\lib\bootstrap.min.css'"
)

cd ..

echo [SUCCESS] 前端环境设置完成
goto :eof

:create_directories
echo [INFO] 创建项目目录...

if not exist "data\uploads" mkdir data\uploads
if not exist "logs" mkdir logs
if not exist "results" mkdir results
if not exist "docs\api" mkdir docs\api

echo [SUCCESS] 项目目录创建完成
goto :eof

:create_env_files
echo [INFO] 创建环境配置文件...

REM 创建环境变量示例文件
if not exist ".env.example" (
    echo # 应用配置> .env.example
    echo APP_NAME=MOF数据t-SNE可视化>> .env.example
    echo APP_VERSION=1.0.0>> .env.example
    echo DEBUG=true>> .env.example
    echo.>> .env.example
    echo # 数据库配置>> .env.example
    echo DATABASE_URL=sqlite:///./data/mof_viz.db>> .env.example
    echo.>> .env.example
    echo # Redis配置>> .env.example
    echo REDIS_URL=redis://localhost:6379/0>> .env.example
    echo.>> .env.example
    echo # 安全配置>> .env.example
    echo SECRET_KEY=your-secret-key-here>> .env.example
    echo.>> .env.example
    echo # 日志配置>> .env.example
    echo LOG_LEVEL=INFO>> .env.example
    echo LOG_FILE=logs/app.log>> .env.example
    echo.>> .env.example
    echo # 文件上传配置>> .env.example
    echo MAX_UPLOAD_SIZE=104857600>> .env.example
    echo ALLOWED_EXTENSIONS=csv>> .env.example
    echo.>> .env.example
    echo # CORS配置>> .env.example
    echo ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000>> .env.example
)

REM 如果.env文件不存在，则复制示例文件
if not exist ".env" copy .env.example .env

echo [SUCCESS] 环境配置文件创建完成
goto :eof

:verify_tools
echo [INFO] 验证开发工具...

REM 激活Python虚拟环境
if exist "venv\Scripts\activate.bat" call venv\Scripts\activate

REM 检查Python包
python -c "import fastapi; print('FastAPI:', fastapi.__version__)" 2>nul
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)" 2>nul
python -c "import pandas; print('pandas:', pandas.__version__)" 2>nul
python -c "import plotly; print('plotly:', plotly.__version__)" 2>nul

REM 检查代码工具
where black >nul 2>&1
if not errorlevel 1 echo [SUCCESS] Black 代码格式化工具可用

where flake8 >nul 2>&1
if not errorlevel 1 echo [SUCCESS] Flake8 代码检查工具可用

where mypy >nul 2>&1
if not errorlevel 1 echo [SUCCESS] MyPy 类型检查工具可用

where pytest >nul 2>&1
if not errorlevel 1 echo [SUCCESS] Pytest 测试框架可用

REM 检查前端工具
cd frontend
where npm >nul 2>&1
if not errorlevel 1 echo [SUCCESS] NPM 包管理器可用

if exist "node_modules\.bin\jest.cmd" echo [SUCCESS] Jest 测试框架可用
if exist "node_modules\.bin\eslint.cmd" echo [SUCCESS] ESLint 代码检查工具可用

cd ..

echo [SUCCESS] 开发工具验证完成
goto :eof

:run_tests
echo [INFO] 运行测试套件...

REM 激活Python虚拟环境
if exist "venv\Scripts\activate.bat" call venv\Scripts\activate

REM 运行Python测试
echo [INFO] 运行Python测试...
cd backend
python -m pytest tests\ -v --tb=short --cov=src --cov-report=term-missing
cd ..

REM 运行前端测试
echo [INFO] 运行前端测试...
cd frontend
npm test
cd ..

echo [SUCCESS] 测试运行完成
goto :eof

:build_project
echo [INFO] 构建项目...

REM 激活Python虚拟环境
if exist "venv\Scripts\activate.bat" call venv\Scripts\activate

REM 构建后端
echo [INFO] 构建后端...
cd backend
python -m pip install -e .
cd ..

REM 构建前端
echo [INFO] 构建前端...
cd frontend
npm run build
cd ..

echo [SUCCESS] 项目构建完成
goto :eof

:show_help
echo 使用方法:
echo     setup.bat [选项]
echo.
echo 选项:
echo     --help, -h          显示此帮助信息
echo     --dev, -d           设置开发环境（默认）
echo     --prod, -p          设置生产环境
echo     --test, -t          设置测试环境
echo     --verify, -v        验证当前环境
echo     --test-run, -tr     运行测试
echo     --build, -b         构建项目
echo.
echo 环境设置完成后，您可以:
echo     1. 启动后端服务: venv\Scripts\activate ^&^& python -m uvicorn backend.main:app --reload
echo     2. 启动前端服务: cd frontend ^&^& npm run dev
echo     3. 运行测试: setup.bat --test-run
echo.
goto :eof