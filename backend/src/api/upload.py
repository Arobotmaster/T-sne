"""
文件上传API模块

实现CSV文件上传和数据集创建功能，符合SDD Constitution原则：
- Library-First: 集成UploadService
- CLI Interface: 支持命令行测试
- Test-First: 完整的测试覆盖
- Integration-First: 与其他服务集成
- Scientific Observability: 详细的处理日志
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any
import time

from src.services.upload_service import UploadService
from src.models.dataset import DatasetInfo, UploadResponse
from src.config.settings import settings
from src.config.logging_config import get_logger
from src.utils.temp_file_manager import get_temp_file_manager

router = APIRouter()
logger = get_logger(__name__)


@router.post("", response_model=UploadResponse)
async def upload_csv_file(file: UploadFile = File(...)):
    """
    上传CSV文件并创建数据集

    符合Constitution的CLI Interface和Scientific Observability原则

    Args:
        file: 上传的CSV文件

    Returns:
        UploadResponse: 上传结果响应

    Raises:
        HTTPException: 文件验证或处理错误
    """
    start_time = time.time()
    logger.info(f"收到文件上传请求: {file.filename}")

    try:
        # 验证文件名
        if not file.filename or file.filename.strip() == '':
            raise HTTPException(
                status_code=400,
                detail="文件名不能为空"
            )

        # 验证文件扩展名
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式 '{file_extension}'，仅支持: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )

        # 验证文件大小
        if file.size and file.size > settings.MAX_UPLOAD_SIZE:
            max_size_mb = settings.MAX_UPLOAD_SIZE // (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=f"文件大小超过限制 (最大 {max_size_mb}MB)"
            )

        # 验证文件名安全性
        import re
        # 允许字母、数字、中文、空格、连字符、点、下划线以及常用符号
        if not re.match(r'^[\w\s\-\.\_\&\u4e00-\u9fff\(\)\[\]\{\}\+\=\%\#\@\!\~\,\;\']+$', file.filename):
            raise HTTPException(
                status_code=400,
                detail="文件名包含非法字符，仅允许使用字母、数字、中文、空格和常用符号"
            )

        # 创建上传服务实例
        upload_service = UploadService()

        # 处理文件上传
        dataset_info = await upload_service.process_upload(file)

        processing_time = time.time() - start_time

        logger.info(f"文件上传成功: {file.filename}")
        logger.info(f"数据集ID: {dataset_info.dataset_id}")
        logger.info(f"数据质量评分: {dataset_info.data_quality_score:.3f}")
        logger.info(f"处理时间: {processing_time:.3f}秒")

        return UploadResponse(
            success=True,
            data=dataset_info.dict(),
            message=f"文件 '{file.filename}' 上传成功",
            processing_time=processing_time
        )

    except ValueError as e:
        logger.error(f"文件验证错误: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"文件处理错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="内部服务器错误")


@router.get("/formats")
async def get_supported_formats():
    """获取支持的文件格式"""
    logger.info("获取支持的文件格式请求")

    max_size_mb = settings.MAX_UPLOAD_SIZE // (1024 * 1024)

    return {
        "supported_formats": settings.ALLOWED_EXTENSIONS,
        "max_file_size": f"{max_size_mb}MB",
        "encoding_support": ["utf-8", "gbk", "gb2312", "ascii"],
        "required_columns": ["sample_id"],
        "recommended_columns": ["sample_id", "category"],
        "max_columns": 1000,
        "max_rows": 100000,
        "upload_directory": settings.UPLOAD_DIRECTORY,
        "temp_directory": settings.TEMP_DIRECTORY
    }


@router.get("/validate/{filename}")
async def validate_file_format(filename: str):
    """验证文件格式和结构（模拟端点）"""
    logger.info(f"验证文件格式请求: {filename}")

    # 模拟验证逻辑
    if not filename.endswith('.csv'):
        return {
            "valid": False,
            "errors": ["文件格式不支持，请使用CSV格式"]
        }

    return {
        "valid": True,
        "warnings": [],
        "suggestions": [
            "确保第一行包含列标题",
            "确保包含sample_id列作为样本标识",
            "建议包含category列用于分类可视化"
        ]
    }


@router.get("/temp-status")
async def get_temp_file_status():
    """获取临时文件状态"""
    logger.info("获取临时文件状态请求")

    try:
        temp_manager = get_temp_file_manager()
        status = temp_manager.get_status()
        return {
            "success": True,
            "data": status,
            "message": "临时文件状态获取成功"
        }
    except Exception as e:
        logger.error(f"获取临时文件状态失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "获取临时文件状态失败"
        }


@router.post("/cleanup")
async def cleanup_temp_files():
    """手动清理临时文件"""
    logger.info("手动清理临时文件请求")

    try:
        temp_manager = get_temp_file_manager()
        cleaned_count = temp_manager.cleanup_all_temp_files()
        return {
            "success": True,
            "data": {
                "cleaned_files": cleaned_count
            },
            "message": f"成功清理 {cleaned_count} 个临时文件"
        }
    except Exception as e:
        logger.error(f"清理临时文件失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "清理临时文件失败"
        }


logger.info("文件上传API模块初始化完成")
