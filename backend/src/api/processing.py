"""
数据处理API模块

实现数据预处理、PCA降维和t-SNE计算功能，符合SDD Constitution原则：
- Library-First: 集成算法库
- CLI Interface: 支持命令行处理
- Test-First: 完整的测试覆盖
- Integration-First: 与其他服务集成
- Scientific Observability: 详细的处理日志
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List
import uuid
import time

router = APIRouter()

# 模拟处理任务存储
_processing_tasks = {}


@router.post("/start")
async def start_processing(
    dataset_id: str,
    config: Dict[str, Any] = None
):
    """
    启动数据处理任务

    Args:
        dataset_id: 数据集ID
        config: 处理配置

    Returns:
        Dict: 任务信息
    """
    if config is None:
        config = {
            "preprocessing": {
                "missing_value_strategy": "mean",
                "scaling_method": "standard"
            },
            "pca": {
                "n_components": 10,
                "random_state": 42
            },
            "tsne": {
                "perplexity": 30,
                "n_components": 2,
                "random_state": 42,
                "learning_rate": 200,
                "n_iter": 1000
            }
        }

    task_id = str(uuid.uuid4())

    _processing_tasks[task_id] = {
        "task_id": task_id,
        "dataset_id": dataset_id,
        "config": config,
        "status": "pending",
        "progress": 0.0,
        "created_at": time.time(),
        "updated_at": time.time(),
        "result": None,
        "error": None
    }

    return {
        "task_id": task_id,
        "status": "pending",
        "message": "处理任务已启动",
        "dataset_id": dataset_id
    }


@router.post("/from-file")
async def start_processing_from_file(payload: Dict[str, Any]):
    """
    开发辅助：从 data/ 路径直接加载 CSV 并启动处理（异步）

    请求JSON示例：
    {
      "file_path": "data/uploads/CSV_pld_filtered_category_4519.csv",
      "embedding": "E2",
      "id_column": "sample_id",
      "category_column": "category",
      "config": { ... 可选 ... }
    }
    """
    file_path = payload.get('file_path')
    if not file_path:
        raise HTTPException(status_code=400, detail="缺少 file_path")

    embedding = payload.get('embedding', 'E2')
    id_column = payload.get('id_column', 'sample_id')
    category_column = payload.get('category_column', 'category')
    config = payload.get('config') or {
        "preprocessing_config": {"missing_value_strategy": "mean", "scaling_method": "standard"},
        "pca_config": {"n_components": 50, "random_state": 42, "whiten": False},
        "tsne_config": {"perplexity": 30 if embedding == 'E2' else 50, "n_components": 2, "learning_rate": 200, "n_iter": 1000, "random_state": 42}
    }

    # 注册任务
    task_id = str(uuid.uuid4())
    _processing_tasks[task_id] = {
        "task_id": task_id,
        "file_path": file_path,
        "embedding": embedding,
        "status": "running",
        "progress": 0.0,
        "created_at": time.time(),
        "updated_at": time.time(),
    }

    async def _run():
        from src.services.processing_service import ProcessingService
        from .pipelines import _pipelines_store
        from .visualizations import _visualizations_store
        from pathlib import Path
        from datetime import datetime
        import pandas as pd

        try:
            svc = ProcessingService(data_dir="data", results_dir="results")
            _processing_tasks[task_id]["progress"] = 0.1

            pipeline_id, summary = await svc.start_processing_from_file(
                file_path=file_path,
                config=config,
                id_column=id_column,
                category_column=category_column,
                embedding_label=embedding
            )

            # 注册到 pipelines store
            now = datetime.now().isoformat()
            _pipelines_store[pipeline_id] = {
                "pipeline_id": pipeline_id,
                "dataset_id": f"from_file::{Path(file_path).name}",
                "status": "completed",
                "start_time": now,
                "end_time": now,
                "progress_percentage": 100,
                "current_step": "completed",
                "steps": [],
                "error_message": None
            }

            # 载入坐标并注册到可视化存储
            coords_path = Path("results") / pipeline_id / "tsne_coordinates.csv"
            df = pd.read_csv(coords_path)
            categories = []
            if 'category' in df.columns:
                counts = df['category'].value_counts().to_dict()
                name_to_id = {name: idx for idx, name in enumerate(sorted(counts.keys()))}
                categories = [
                    {"category_id": cid, "category_name": name, "sample_count": counts[name], "color_code": "#999999"}
                    for name, cid in name_to_id.items()
                ]
            coords = []
            for _, row in df.iterrows():
                cat_name = str(row.get('category')) if 'category' in df.columns else 'unknown'
                cat_id = next((c['category_id'] for c in categories if c['category_name'] == cat_name), 0)
                coords.append({
                    "sample_id": str(row['sample_id']),
                    "x": float(row['x']),
                    "y": float(row['y']),
                    "category_id": int(cat_id),
                    "category_name": cat_name,
                    "distance_to_center": (float(row['x'])**2 + float(row['y'])**2) ** 0.5,
                    "local_density": 0.0
                })

            _visualizations_store[pipeline_id] = {
                "visualization_id": pipeline_id,
                "pipeline_id": pipeline_id,
                "dataset_id": f"from_file::{Path(file_path).name}",
                "status": "completed",
                "config": {"chart_title": f"t-SNE {embedding}", "width": 800, "height": 600, "show_legend": True},
                "coordinates": coords,
                "categories": categories,
                "statistics": {
                    "total_samples": len(coords),
                    "valid_samples": len(coords)
                },
                "render_time_ms": 0,
                "created_at": now
            }

            _processing_tasks[task_id]["status"] = "completed"
            _processing_tasks[task_id]["progress"] = 1.0
            _processing_tasks[task_id]["updated_at"] = time.time()
            _processing_tasks[task_id]["result"] = {"pipeline_id": pipeline_id, "embedding": embedding, "summary": summary}
        except Exception as e:
            _processing_tasks[task_id]["status"] = "failed"
            _processing_tasks[task_id]["error"] = str(e)
            _processing_tasks[task_id]["updated_at"] = time.time()

    # 后台任务运行
    try:
        import asyncio
        asyncio.create_task(_run())
    except RuntimeError:
        # 若无事件循环环境，降级为同步调用（开发场景）
        import asyncio as aio
        aio.get_event_loop().run_until_complete(_run())

    return {"success": True, "task_id": task_id, "message": "from-file 处理已启动"}


@router.get("/{task_id}/status")
async def get_processing_status(task_id: str):
    """
    获取处理任务状态

    Args:
        task_id: 任务ID

    Returns:
        Dict: 任务状态
    """
    if task_id not in _processing_tasks:
        raise HTTPException(status_code=404, detail="处理任务不存在")

    task = _processing_tasks[task_id]

    # 模拟任务进度
    if task["status"] == "pending":
        task["status"] = "running"
        task["progress"] = 0.1
        task["updated_at"] = time.time()
    elif task["status"] == "running" and task["progress"] < 1.0:
        task["progress"] = min(1.0, task["progress"] + 0.1)
        task["updated_at"] = time.time()

        if task["progress"] >= 1.0:
            task["status"] = "completed"
            task["result"] = {
                "pipeline_id": task_id,
                "coordinates": [
                    {"x": 1.2, "y": 3.4, "sample_id": "sample_001", "category_id": 1},
                    {"x": 2.1, "y": 4.3, "sample_id": "sample_002", "category_id": 2},
                    {"x": 1.5, "y": 3.2, "sample_id": "sample_003", "category_id": 1}
                ],
                "config": task["config"],
                "metadata": {
                    "original_shape": [100, 10],
                    "pca_shape": [100, 10],
                    "final_shape": [100, 2],
                    "processing_time": 5.23,
                    "explained_variance": [0.45, 0.32, 0.15]
                }
            }

    return {
        "task_id": task_id,
        "status": task["status"],
        "progress": task["progress"],
        "created_at": task["created_at"],
        "updated_at": task["updated_at"],
        "error": task.get("error"),
        "result": task.get("result")
    }


@router.get("/{task_id}/result")
async def get_processing_result(task_id: str):
    """
    获取处理结果

    Args:
        task_id: 任务ID

    Returns:
        Dict: 处理结果

    Raises:
        HTTPException: 任务不存在或未完成
    """
    if task_id not in _processing_tasks:
        raise HTTPException(status_code=404, detail="处理任务不存在")

    task = _processing_tasks[task_id]

    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"任务状态为 {task['status']}，无法获取结果"
        )

    return task["result"]


@router.get("/configs/default")
async def get_default_configs():
    """
    获取默认处理配置

    Returns:
        Dict: 默认配置
    """
    return {
        "preprocessing": {
            "missing_value_strategy": "mean",
            "scaling_method": "standard",
            "outlier_detection": False
        },
        "pca": {
            "n_components": 50,
            "random_state": 42,
            "whiten": False
        },
        "tsne": {
            "perplexity": 30,
            "n_components": 2,
            "random_state": 42,
            "learning_rate": 200,
            "n_iter": 1000,
            "early_exaggeration": 12,
            "metric": "euclidean"
        },
        "validation": {
            "min_samples": 10,
            "max_features": 1000,
            "missing_value_threshold": 0.5
        }
    }


@router.get("/algorithms/info")
async def get_algorithms_info():
    """
    获取算法信息

    Returns:
        Dict: 算法信息
    """
    return {
        "preprocessing": {
            "name": "数据预处理",
            "description": "处理缺失值、异常值和数据标准化",
            "methods": {
                "missing_value_strategy": ["mean", "median", "mode", "drop"],
                "scaling_method": ["standard", "minmax", "robust", "none"]
            }
        },
        "pca": {
            "name": "主成分分析",
            "description": "线性降维算法，保留主要方差信息",
            "parameters": {
                "n_components": "int or float (0-1)",
                "random_state": "int",
                "whiten": "bool"
            }
        },
        "tsne": {
            "name": "t-分布随机邻域嵌入",
            "description": "非线性降维算法，适用于高维数据可视化",
            "parameters": {
                "perplexity": "float (5-50)",
                "n_components": "int (usually 2)",
                "random_state": "int",
                "learning_rate": "float (10-1000)",
                "n_iter": "int (250-1000)"
            }
        }
    }
