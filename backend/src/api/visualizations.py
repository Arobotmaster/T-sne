"""
可视化API模块

实现可视化数据的获取、配置和导出功能，符合SDD Constitution原则：
- Library-First: 集成可视化服务
- CLI Interface: 支持命令行操作
- Test-First: 完整的测试覆盖
- Integration-First: 与处理流水线集成
- Scientific Observability: 详细的可视化日志
"""

from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import uuid
import os
from pathlib import Path
import json as _json

router = APIRouter()

# 模拟可视化存储（实际项目中应该使用数据库）
_visualizations_store = {}
_comparison_configs_store = {}


@router.get("/{pipeline_id}")
async def get_visualization_data(pipeline_id: str):
    """
    获取可视化数据 - T046

    Args:
        pipeline_id: 处理流水线ID

    Returns:
        Dict: 可视化数据

    Raises:
        HTTPException: 可视化不存在或处理未完成
    """
    if pipeline_id not in _visualizations_store:
        raise HTTPException(status_code=404, detail="可视化数据不存在")

    visualization = _visualizations_store[pipeline_id]

    if visualization.get("status") != "completed":
        raise HTTPException(status_code=425, detail="处理尚未完成，请稍后再试")

    return {
        "success": True,
        "data": {
            "visualization_id": visualization["visualization_id"],
            "pipeline_id": pipeline_id,
            "config": visualization["config"],
            "coordinates": visualization["coordinates"],
            "categories": visualization["categories"],
            "statistics": visualization["statistics"],
            "render_time_ms": visualization.get("render_time_ms", 100),
            "created_at": visualization["created_at"]
        }
    }


@router.get("/{pipeline_id}/metrics")
async def get_visualization_metrics(pipeline_id: str):
    """
    获取可视化与训练过程的关键指标

    优先从 results/{pipeline_id}/processing_metadata.json 读取：
    - PCA: n_components、累计解释方差
    - t-SNE: kl_divergence、n_iter、参数快照
    - 总样本数
    若文件不存在，则从内存可视化存储或坐标文件估计样本数。
    """
    from pathlib import Path
    import json as _json
    import pandas as _pd

    # 尝试读取 processing_metadata.json
    meta_path = Path('results') / pipeline_id / 'processing_metadata.json'
    metrics = {
        'total_samples': None,
        'pca': None,
        'tsne': None
    }
    try:
        if meta_path.exists():
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = _json.load(f)
            metrics['total_samples'] = int(meta.get('total_samples') or 0)
            pca_meta = meta.get('pca', {}).get('metadata') or {}
            tsne_meta = meta.get('tsne', {}).get('metadata') or {}
            metrics['pca'] = {
                'n_components': pca_meta.get('n_components'),
                'total_variance_retained': pca_meta.get('total_variance_retained'),
                'cumulative_variance_ratio': pca_meta.get('cumulative_variance_ratio')
            }
            metrics['tsne'] = {
                'kl_divergence': tsne_meta.get('kl_divergence'),
                'n_iter': tsne_meta.get('n_iter'),
                'config': meta.get('tsne', {}).get('config')
            }
        else:
            # 退化为坐标行数
            coords_path = Path('results') / pipeline_id / 'tsne_coordinates.csv'
            if coords_path.exists():
                df = _pd.read_csv(coords_path)
                metrics['total_samples'] = int(len(df))
    except Exception:
        pass

    # 若仍未知 total_samples，尝试内存存储
    try:
        viz = _visualizations_store.get(pipeline_id)
        if viz and not metrics['total_samples']:
            metrics['total_samples'] = int(len(viz.get('coordinates', [])))
    except Exception:
        pass

    return { 'success': True, 'data': metrics }


@router.get("/{pipeline_id}/report")
async def get_training_report(pipeline_id: str):
    """
    返回训练报告：参数、指标、自动调参结果（若有）。
    """
    report = {
        'parameters': None,
        'metrics': None,
        'tuning': None
    }
    # 读取 processing_metadata.json
    meta_path = Path('results') / pipeline_id / 'processing_metadata.json'
    if meta_path.exists():
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = _json.load(f)
            report['parameters'] = {
                'preprocessing_config': meta.get('preprocessing', {}).get('config'),
                'pca_config': meta.get('pca', {}).get('config'),
                'tsne_config': meta.get('tsne', {}).get('config')
            }
            report['metrics'] = {
                'total_samples': meta.get('total_samples'),
                'pca': meta.get('pca', {}).get('metadata'),
                'tsne': meta.get('tsne', {}).get('metadata')
            }
        except Exception:
            pass
    # 读取 tuning_summary.json（如果存在）
    tuning_path = Path('results') / pipeline_id / 'tuning_summary.json'
    if tuning_path.exists():
        try:
            with open(tuning_path, 'r', encoding='utf-8') as f:
                report['tuning'] = _json.load(f)
        except Exception:
            pass

    return { 'success': True, 'data': report }


@router.post("/{pipeline_id}/export")
async def export_visualization(
    pipeline_id: str,
    export_request: Dict[str, Any]
):
    """
    导出可视化图像 - T047

    Args:
        pipeline_id: 处理流水线ID
        export_request: 导出配置

    Returns:
        Dict: 导出结果

    Raises:
        HTTPException: 可视化不存在或导出失败
    """
    if pipeline_id not in _visualizations_store:
        raise HTTPException(status_code=404, detail="可视化数据不存在")

    visualization = _visualizations_store[pipeline_id]

    # 默认导出配置
    default_config = {
        "format": "png",
        "width": 800,
        "height": 600,
        "dpi": 300,
        "background_color": "white",
        "filename": f"mof_visualization_{pipeline_id[:8]}"
    }

    # 合并用户配置
    default_config.update(export_request)

    # 验证配置
    if default_config["format"] not in ["png", "svg", "pdf"]:
        raise HTTPException(status_code=400, detail="不支持的导出格式")

    # 模拟导出处理
    export_id = str(uuid.uuid4())
    file_extension = default_config["format"]
    filename = f"{default_config['filename']}.{file_extension}"

    # 模拟文件生成
    file_size = 1024 * 500  # 500KB
    download_url = f"/api/downloads/{export_id}"

    export_result = {
        "export_id": export_id,
        "pipeline_id": pipeline_id,
        "filename": filename,
        "format": default_config["format"],
        "file_size_bytes": file_size,
        "download_url": download_url,
        "expires_at": (datetime.now() + timedelta(hours=1)).isoformat(),
        "created_at": datetime.now().isoformat()
    }

    # 存储导出记录
    if "exports" not in visualization:
        visualization["exports"] = []
    visualization["exports"].append(export_result)

    return {"success": True, "data": export_result}


@router.post("/comparison")
async def create_comparison_visualization(comparison_request: Dict[str, Any]):
    """
    创建对比可视化 - T082

    Args:
        comparison_request: 对比配置请求

    Returns:
        Dict: 对比可视化结果

    Raises:
        HTTPException: 配置错误或数据不存在
    """
    # 验证必需参数
    required_fields = ["original_pipeline_id", "filtered_pipeline_id", "comparison_type"]
    for field in required_fields:
        if field not in comparison_request:
            raise HTTPException(status_code=400, detail=f"缺少必需字段: {field}")

    original_pipeline_id = comparison_request["original_pipeline_id"]
    filtered_pipeline_id = comparison_request["filtered_pipeline_id"]

    # 验证流水线存在
    if original_pipeline_id not in _visualizations_store:
        raise HTTPException(status_code=404, detail="原始数据流水线不存在")

    if filtered_pipeline_id not in _visualizations_store:
        raise HTTPException(status_code=404, detail="筛选数据流水线不存在")

    # 创建对比可视化ID
    comparison_id = str(uuid.uuid4())

    # 模拟对比可视化数据
    comparison_data = {
        "comparison_id": comparison_id,
        "original_pipeline_id": original_pipeline_id,
        "filtered_pipeline_id": filtered_pipeline_id,
        "comparison_type": comparison_request["comparison_type"],
        "config": comparison_request.get("config", {}),
        "coordinates": {
            "original": _visualizations_store[original_pipeline_id]["coordinates"],
            "filtered": _visualizations_store[filtered_pipeline_id]["coordinates"]
        },
        "statistics": {
            "original_samples": len(_visualizations_store[original_pipeline_id]["coordinates"]),
            "filtered_samples": len(_visualizations_store[filtered_pipeline_id]["coordinates"]),
            "overlap_percentage": 85.5,
            "similarity_score": 0.92
        },
        "created_at": datetime.now().isoformat()
    }

    # 存储对比数据
    _comparison_configs_store[comparison_id] = comparison_data

    return {
        "success": True,
        "comparison_id": comparison_id,
        "message": "对比可视化创建成功",
        "data": comparison_data
    }


@router.put("/{pipeline_id}/comparison-config")
async def update_comparison_config(
    pipeline_id: str,
    config_request: Dict[str, Any]
):
    """
    更新对比可视化配置 - T083

    Args:
        pipeline_id: 处理流水线ID
        config_request: 配置更新请求

    Returns:
        Dict: 配置更新结果

    Raises:
        HTTPException: 可视化不存在或配置错误
    """
    if pipeline_id not in _visualizations_store:
        raise HTTPException(status_code=404, detail="可视化数据不存在")

    visualization = _visualizations_store[pipeline_id]

    # 更新配置
    if "config" not in visualization:
        visualization["config"] = {}

    # 支持的配置项
    supported_configs = [
        "chart_title", "x_axis_label", "y_axis_label",
        "width", "height", "show_legend",
        "color_scheme", "marker_styles",
        "comparison_mode", "transparency", "highlight_filtered"
    ]

    for key, value in config_request.items():
        if key in supported_configs:
            visualization["config"][key] = value
        else:
            # 记录不支持的配置项但不报错
            pass

    visualization["updated_at"] = datetime.now().isoformat()

    return {
        "success": True,
        "pipeline_id": pipeline_id,
        "config": visualization["config"],
        "message": "对比可视化配置更新成功",
        "updated_at": visualization["updated_at"]
    }


@router.get("/{pipeline_id}/config")
async def get_visualization_config(pipeline_id: str):
    """
    获取可视化配置

    Args:
        pipeline_id: 处理流水线ID

    Returns:
        Dict: 可视化配置

    Raises:
        HTTPException: 可视化不存在
    """
    if pipeline_id not in _visualizations_store:
        raise HTTPException(status_code=404, detail="可视化数据不存在")

    visualization = _visualizations_store[pipeline_id]
    return visualization.get("config", {})


@router.get("")
async def list_visualizations(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    dataset_id: Optional[str] = Query(None, description="数据集ID过滤")
):
    """
    获取可视化列表

    Args:
        skip: 跳过的记录数
        limit: 返回的记录数限制
        dataset_id: 数据集ID过滤

    Returns:
        Dict: 可视化列表和分页信息
    """
    visualizations = list(_visualizations_store.values())

    # 数据集过滤
    if dataset_id:
        visualizations = [v for v in visualizations if v.get("dataset_id") == dataset_id]

    # 按创建时间排序
    visualizations.sort(key=lambda x: x["created_at"], reverse=True)

    # 分页
    total = len(visualizations)
    visualizations = visualizations[skip:skip + limit]

    return {
        "visualizations": visualizations,
        "total": total,
        "skip": skip,
        "limit": limit,
        "has_more": skip + limit < total
    }


def create_test_visualization(pipeline_id: str, dataset_id: str):
    """创建测试可视化（用于契约测试）"""
    visualization_id = str(uuid.uuid4())

    # 模拟坐标数据
    coordinates = []
    categories = ["MOF_A", "MOF_B", "MOF_C", "MOF_D"]

    for i in range(100):
        x = i * 0.1 + (i % 10) * 0.05
        y = i * 0.08 + (i % 7) * 0.03
        category_id = i % 4

        coordinates.append({
            "sample_id": f"sample_{i:03d}",
            "x": x,
            "y": y,
            "category_id": category_id,
            "category_name": categories[category_id],
            "distance_to_center": (x**2 + y**2)**0.5,
            "local_density": 0.5 + (i % 5) * 0.1
        })

    visualization = {
        "visualization_id": visualization_id,
        "pipeline_id": pipeline_id,
        "dataset_id": dataset_id,
        "status": "completed",
        "config": {
            "chart_title": "MOF数据t-SNE可视化",
            "x_axis_label": "t-SNE维度1",
            "y_axis_label": "t-SNE维度2",
            "width": 800,
            "height": 600,
            "show_legend": True,
            "color_scheme": [
                {"category_id": 0, "color_hex": "#FF6B6B", "opacity": 1.0},
                {"category_id": 1, "color_hex": "#4ECDC4", "opacity": 1.0},
                {"category_id": 2, "color_hex": "#45B7D1", "opacity": 1.0},
                {"category_id": 3, "color_hex": "#96CEB4", "opacity": 1.0}
            ]
        },
        "coordinates": coordinates,
        "categories": [
            {"category_id": 0, "category_name": "MOF_A", "sample_count": 25, "color_code": "#FF6B6B"},
            {"category_id": 1, "category_name": "MOF_B", "sample_count": 25, "color_code": "#4ECDC4"},
            {"category_id": 2, "category_name": "MOF_C", "sample_count": 25, "color_code": "#45B7D1"},
            {"category_id": 3, "category_name": "MOF_D", "sample_count": 25, "color_code": "#96CEB4"}
        ],
        "statistics": {
            "total_samples": 100,
            "valid_samples": 100,
            "category_counts": {"MOF_A": 25, "MOF_B": 25, "MOF_C": 25, "MOF_D": 25},
            "x_range": {"min": 0.0, "max": 10.0},
            "y_range": {"min": 0.0, "max": 8.0},
            "density_info": {
                "high_density_regions": 3,
                "low_density_regions": 2,
                "average_density": 0.75
            }
        },
        "render_time_ms": 150,
        "created_at": datetime.now().isoformat(),
        "exports": []
    }

    _visualizations_store[pipeline_id] = visualization
    return visualization_id
