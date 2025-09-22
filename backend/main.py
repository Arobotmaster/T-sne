"""
MOF数据t-SNE交互式可视化 - FastAPI应用入口

遵循SDD Constitution原则：
- Library-First: 每个算法都是独立库
- CLI Interface: 支持命令行调用
- Test-First: TDD开发模式
- Scientific Observability: 详细日志记录
"""

import logging
import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn

from src.config.settings import settings
from src.config.logging_config import setup_logging, get_logger
from src.api.error_handlers import setup_error_handling, unified_error_handler
from src.middleware.graceful_degradation import degradation_manager
from src.utils.user_messages import user_message_manager

# 设置日志
logger = setup_logging(settings.LOG_LEVEL)
app_logger = get_logger(__name__)

# 全局变量存储应用状态
_app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    app_logger.info("=== MOF t-SNE可视化应用启动 ===")
    app_logger.info(f"应用名称: {settings.APP_NAME}")
    app_logger.info(f"版本: {settings.APP_VERSION}")
    app_logger.info(f"环境: {settings.ENVIRONMENT}")
    app_logger.info(f"调试模式: {settings.DEBUG}")

    # 确保必要目录存在
    settings.ensure_directories()
    app_logger.info("数据目录已初始化")

    # 初始化应用状态
    _app_state['start_time'] = time.time()
    _app_state['processing_jobs'] = {}
    _app_state['active_connections'] = 0

    app_logger.info("应用启动完成")

    yield

    # 应用关闭时的清理
    app_logger.info("=== MOF t-SNE可视化应用关闭 ===")
    app_logger.info(f"运行时间: {time.time() - _app_state['start_time']:.2f}秒")


# 创建FastAPI应用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="金属有机框架(MOF)数据t-SNE降维和交互式可视化Web应用",
    docs_url=settings.API_DOCS_URL,
    redoc_url=settings.API_REDOC_URL,
    lifespan=lifespan,
    debug=settings.DEBUG
)

# 配置CORS
# 更宽松的开发环境 CORS（确保前端 8001 可访问 8000）
_dev_allowed_origins = [
    "http://localhost:8001",
    "http://127.0.0.1:8001",
]

# 合并 settings 中的来源（如有）
try:
    _configured = set(settings.get_cors_origins() or [])
except Exception:
    _configured = set()

_allowed_origins = list(set(_dev_allowed_origins) | _configured)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 设置统一错误处理
setup_error_handling(app)

# 挂载静态文件
if Path("frontend/static").exists():
    app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# 导入API路由
from src.api import api_router
from src.api.websocket import websocket_endpoint
app.include_router(api_router, prefix=settings.API_V1_PREFIX)

# WebSocket端点 - T048
@app.websocket("/ws/pipelines/{pipeline_id}")
async def websocket_route(pipeline_id: str, websocket: WebSocket):
    """WebSocket实时更新端点 - T048"""
    await websocket_endpoint(websocket, pipeline_id)

@app.get("/")
async def root():
    """根路径重定向到前端主页"""
    return JSONResponse(
        status_code=200,
        content={
            "message": f"欢迎使用{settings.APP_NAME}",
            "version": settings.APP_VERSION,
            "docs_url": settings.API_DOCS_URL,
            "frontend_url": "/static/index.html"
        }
    )


@app.get("/health")
async def health_check():
    """健康检查端点"""
    uptime = time.time() - _app_state.get('start_time', time.time())

    return {
        "status": "healthy",
        "timestamp": time.time(),
        "uptime": uptime,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "active_connections": _app_state.get('active_connections', 0),
        "processing_jobs": len(_app_state.get('processing_jobs', {}))
    }


@app.get("/info")
async def app_info():
    """应用信息端点"""
    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "debug": settings.DEBUG,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "supported_formats": settings.SUPPORTED_EXPORT_FORMATS,
        "max_upload_size": settings.MAX_UPLOAD_SIZE,
        "allowed_extensions": settings.ALLOWED_EXTENSIONS
    }


# 错误处理和系统状态端点

@app.get("/error-handling/stats")
async def get_error_handling_stats():
    """获取错误处理统计信息"""
    return {
        "error_statistics": unified_error_handler.get_error_statistics(),
        "degradation_status": degradation_manager.get_system_status(),
        "recent_messages": user_message_manager.get_recent_messages(10)
    }

@app.get("/system/status")
async def get_system_status():
    """获取系统状态"""
    return {
        "system_health": degradation_manager.get_system_status(),
        "error_handling": unified_error_handler.get_error_statistics(),
        "uptime": time.time() - _app_state.get('start_time', time.time()),
        "active_connections": _app_state.get('active_connections', 0)
    }

@app.get("/test-error/{error_type}")
async def test_error_endpoint(error_type: str):
    """测试错误处理端点 - 仅用于开发测试"""
    if settings.ENVIRONMENT != "development":
        raise HTTPException(status_code=404, detail="Not found")

    if error_type == "value":
        raise ValueError("This is a test ValueError")
    elif error_type == "key":
        raise KeyError("This is a test KeyError")
    elif error_type == "validation":
        from fastapi.exceptions import RequestValidationError
        raise RequestValidationError([{
            "loc": ["test", "field"],
            "msg": "Test validation error",
            "type": "value_error"
        }])
    elif error_type == "general":
        raise RuntimeError("This is a test RuntimeError")
    else:
        raise HTTPException(status_code=400, detail="Invalid error type")


# 请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """请求日志中间件"""
    start_time = time.time()
    _app_state['active_connections'] = _app_state.get('active_connections', 0) + 1

    try:
        response = await call_next(request)
        process_time = time.time() - start_time

        app_logger.info(
            f"{request.method} {request.url.path} - "
            f"状态码: {response.status_code} - "
            f"处理时间: {process_time:.3f}s"
        )

        return response
    finally:
        _app_state['active_connections'] = max(0, _app_state.get('active_connections', 1) - 1)

def run_server():
    """运行服务器 - 支持CLI调用"""
    app_logger.info("启动FastAPI服务器...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )


if __name__ == "__main__":
    run_server()