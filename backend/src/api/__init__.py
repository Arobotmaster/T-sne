"""
API路由模块

提供所有API端点的统一入口，符合SDD Constitution原则：
- Library-First: 每个API模块都是独立的
- CLI Interface: 支持命令行测试
- Test-First: 完整的测试覆盖
- Integration-First: 集成测试友好
- Scientific Observability: 详细的API日志记录
"""

from fastapi import APIRouter
from src.config.logging_config import get_logger

logger = get_logger(__name__)

# 创建主路由器
api_router = APIRouter()

# 导入各个API模块
from .upload import router as upload_router
from .datasets import router as datasets_router
from .processing import router as processing_router
from .pipelines import router as pipelines_router
from .visualizations import router as visualizations_router
from .websocket import websocket_endpoint

# 注册路由模块
api_router.include_router(upload_router, prefix="/upload", tags=["文件上传"])
api_router.include_router(datasets_router, prefix="/datasets", tags=["数据集管理"])
api_router.include_router(processing_router, prefix="/processing", tags=["数据处理"])
api_router.include_router(pipelines_router, prefix="/pipelines", tags=["处理流水线"])
api_router.include_router(visualizations_router, prefix="/visualizations", tags=["可视化"])

# WebSocket端点将在主应用中注册，因为需要不同的处理方式

@api_router.get("")
async def api_root():
    """API根路径，返回基本状态，避免前端访问`/api`出现404"""
    logger.info("API根路径检查请求")
    return {
        "status": "active",
        "message": "MOF t-SNE API",
        "version": "1.0.0",
        "endpoints": {
            "status": "/api/status",
            "datasets": "/api/datasets",
            "upload": "/api/upload",
            "pipelines": "/api/pipelines",
            "visualizations": "/api/visualizations"
        }
    }

@api_router.get("/status")
async def api_status():
    """API状态检查"""
    logger.info("API状态检查请求")
    return {
        "status": "active",
        "message": "MOF t-SNE可视化API运行正常",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/api/upload",
            "datasets": "/api/datasets",
            "processing": "/api/processing",
            "pipelines": "/api/pipelines",
            "visualizations": "/api/visualizations",
            "export": "/api/export",
            "websocket": "/ws/pipelines/{pipeline_id}"
        }
    }

@api_router.get("/docs")
async def api_docs():
    """API文档信息"""
    logger.info("API文档请求")
    return {
        "title": "MOF数据t-SNE可视化API",
        "version": "1.0.0",
        "description": "金属有机框架(MOF)数据降维和交互式可视化API",
        "endpoints": {
            "文件上传": "POST /api/upload",
            "数据集管理": "GET/POST/DELETE /api/datasets",
            "数据处理": "POST /api/processing",
            "处理流水线": "GET/POST /api/pipelines",
            "可视化": "GET/POST /api/visualizations",
            "导出功能": "POST /api/export",
            "实时更新": "WebSocket /ws/pipelines/{pipeline_id}"
        },
        "openapi_url": "/docs",
        "redoc_url": "/redoc"
    }

logger.info("API路由模块初始化完成")
