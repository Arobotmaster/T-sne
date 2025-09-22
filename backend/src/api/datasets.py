"""
数据集管理API模块

实现数据集的CRUD操作功能，符合SDD Constitution原则：
- Library-First: 集成数据处理服务
- CLI Interface: 支持命令行操作
- Test-First: 完整的测试覆盖
- Integration-First: 与上传和处理服务集成
- Scientific Observability: 详细的操作日志
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
from pathlib import Path
import json
import os

from src.services.upload_service import UploadService
from .pipelines import _pipelines_store  # 供状态查询模块使用
from .visualizations import create_test_visualization  # 在完成时生成可视化占位数据
from src.config.settings import settings
from src.config.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)

# 获取上传服务实例
upload_service = UploadService()


@router.get("")
async def list_datasets(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    search: Optional[str] = Query(None, description="搜索关键词")
):
    """
    获取数据集列表

    Args:
        skip: 跳过的记录数
        limit: 返回的记录数限制
        search: 搜索关键词

    Returns:
        Dict: 数据集列表和分页信息
    """
    try:
        # 使用上传服务获取真实的数据集列表
        datasets = await upload_service.list_datasets()

        # 搜索过滤
        if search:
            datasets = [d for d in datasets if search.lower() in d['filename'].lower()]

        # 分页
        total = len(datasets)
        datasets = datasets[skip:skip + limit]

        return {
            "datasets": datasets,
            "total": total,
            "skip": skip,
            "limit": limit,
            "has_more": skip + limit < total
        }
    except Exception as e:
        logger.error(f"获取数据集列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail="获取数据集列表失败")


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str):
    """
    获取指定数据集的详细信息

    Args:
        dataset_id: 数据集ID

    Returns:
        Dict: 数据集详细信息

    Raises:
        HTTPException: 数据集不存在
    """
    try:
        # 使用上传服务获取真实的数据集信息
        dataset = await upload_service.get_dataset(dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="数据集不存在")
        return dataset
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取数据集详细信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail="获取数据集详细信息失败")


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """
    删除数据集

    Args:
        dataset_id: 数据集ID

    Returns:
        Dict: 删除结果

    Raises:
        HTTPException: 数据集不存在
    """
    try:
        # 使用上传服务删除真实的数据集
        success = await upload_service.delete_dataset(dataset_id)
        if not success:
            raise HTTPException(status_code=404, detail="数据集不存在")

        return {
            "success": True,
            "message": f"数据集 {dataset_id} 已删除",
            "deleted_at": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除数据集失败: {str(e)}")
        raise HTTPException(status_code=500, detail="删除数据集失败")


@router.get("/{dataset_id}/statistics")
async def get_dataset_statistics(dataset_id: str):
    """
    获取数据集统计信息

    Args:
        dataset_id: 数据集ID

    Returns:
        Dict: 数据集统计信息

    Raises:
        HTTPException: 数据集不存在
    """
    try:
        # 使用上传服务获取真实的数据集统计信息
        dataset = await upload_service.get_dataset(dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="数据集不存在")

        return {
            "dataset_id": dataset_id,
            "basic_stats": {
                "row_count": dataset.get("row_count", 0),
                "column_count": dataset.get("column_count", 0),
                "file_size": dataset.get("file_size", 0),
                "data_quality_score": dataset.get("data_quality_score", 0.0)
            },
            "columns": dataset.get("columns", []),
            "categories": dataset.get("categories", []),
            "missing_values": dataset.get("missing_values", {}),
            "data_types": dataset.get("data_types", {}),
            "created_at": dataset.get("created_at"),
            "updated_at": dataset.get("updated_at")
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取数据集统计信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail="获取数据集统计信息失败")


@router.post("/{dataset_id}/process")
async def start_processing(dataset_id: str, processing_request: Dict[str, Any] = None):
    """
    启动数据处理流水线 - T044

    Args:
        dataset_id: 数据集ID
        processing_request: 处理配置请求

    Returns:
        Dict: 处理任务启动响应

    Raises:
        HTTPException: 数据集不存在或正在处理中
    """
    try:
        # 使用上传服务获取真实的数据集信息
        dataset = await upload_service.get_dataset(dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="数据集不存在")

        current_status = dataset.get("processing_status", "uploaded")
        # 若标记为 processing，但对应流水线已不在运行，则自动纠正状态，允许重新启动
        if current_status == "processing":
            latest = dataset.get("latest_pipeline") or {}
            pid = latest.get("pipeline_id")
            try:
                running = bool(pid and pid in _pipelines_store and _pipelines_store[pid]["status"] == "running")

                # 若传入 force=true：无论是否有运行中流水线，都标记已取消，允许重启
                if processing_request and processing_request.get("force"):
                    if running:
                        _pipelines_store[pid]["status"] = "cancelled"  # type: ignore[index]
                        _pipelines_store[pid]["end_time"] = datetime.now().isoformat()  # type: ignore[index]
                    await upload_service.update_dataset_status(dataset_id, {
                        "processing_status": "cancelled",
                        "latest_pipeline": {
                            "pipeline_id": pid,
                            "status": "cancelled" if pid else "cancelled",
                            "start_time": _pipelines_store.get(pid, {}).get("start_time") if pid else None,
                            "end_time": _pipelines_store.get(pid, {}).get("end_time") if pid else datetime.now().isoformat(),
                        }
                    })
                    current_status = "cancelled"
                else:
                    # 非强制：若没有实际运行中的流水线，则将状态纠正为 uploaded，允许继续
                    if not running:
                        await upload_service.update_dataset_status(dataset_id, {
                            "processing_status": "uploaded",
                            "latest_pipeline": latest
                        })
                        current_status = "uploaded"
            except Exception:
                # 出现异常也不要阻止重启，作为兜底策略：允许继续
                current_status = "uploaded"
            # 仍为 processing 且未允许重启，则报冲突
            if current_status == "processing":
                raise HTTPException(status_code=409, detail="数据集正在处理中")

        # 默认处理配置
        default_config = {
            "pca_config": {
                "n_components": 50,
                "whiten": False,
                "random_state": 42
            },
            "tsne_config": {
                "perplexity": 30,
                "n_components": 2,
                "learning_rate": 200,
                "n_iter": 1000,
                "random_state": 42,
                "metric": "euclidean"
            },
            "preprocessing_config": {
                "missing_value_strategy": "mean",
                "scaling_method": "standard",
                "outlier_detection": True,
                "outlier_threshold": 3.0
            }
        }

    # 合并用户配置
        if processing_request:
            default_config.update(processing_request)

        # 创建处理流水线
        pipeline_id = str(uuid.uuid4())
        start_time = datetime.now()

        pipeline_info = {
            "pipeline_id": pipeline_id,
            "dataset_id": dataset_id,
            "status": "pending",
            "start_time": start_time.isoformat(),
            "end_time": None,
            "config": default_config,
            "progress_percentage": 0,
            "current_step": "初始化",
            "steps": [
                {
                    "step_name": "数据预处理",
                    "step_type": "preprocessing",
                    "status": "pending",
                    "progress_percentage": 0,
                    "start_time": None,
                    "end_time": None,
                    "duration_ms": None,
                    "error_message": None
                },
                {
                    "step_name": "PCA降维",
                    "step_type": "pca",
                    "status": "pending",
                    "progress_percentage": 0,
                    "start_time": None,
                    "end_time": None,
                    "duration_ms": None,
                    "error_message": None
                },
                {
                    "step_name": "t-SNE降维",
                    "step_type": "tsne",
                    "status": "pending",
                    "progress_percentage": 0,
                    "start_time": None,
                    "end_time": None,
                    "duration_ms": None,
                    "error_message": None
                },
                {
                    "step_name": "可视化生成",
                    "step_type": "visualization",
                    "status": "pending",
                    "progress_percentage": 0,
                    "start_time": None,
                    "end_time": None,
                    "duration_ms": None,
                    "error_message": None
                }
            ],
            "error_message": None,
            "estimated_duration_ms": 30000  # 预估30秒
        }

        # 更新数据集状态
        dataset["processing_status"] = "processing"
        dataset["latest_pipeline"] = {
            "pipeline_id": pipeline_id,
            "status": "pending",
            "start_time": start_time.isoformat()
        }

        # 更新真实数据集的元数据
        await upload_service.update_dataset_status(dataset_id, {
            "processing_status": "processing",
            "latest_pipeline": {
                "pipeline_id": pipeline_id,
                "status": "pending",
                "start_time": start_time.isoformat()
            }
        })

        # 在流水线存储中注册，供 /pipelines/{id}/status 查询与进度推进
        _pipelines_store[pipeline_id] = {
            "pipeline_id": pipeline_id,
            "dataset_id": dataset_id,
            "status": "running",
            "start_time": start_time.isoformat(),
            "end_time": None,
            "config": default_config,
            "progress_percentage": 0,
            "current_step": "数据预处理",
            "steps": [
                {
                    "step_name": "数据预处理",
                    "step_type": "preprocessing",
                    "status": "running",
                    "progress_percentage": 0,
                    "start_time": start_time.isoformat(),
                    "end_time": None,
                    "duration_ms": None,
                    "error_message": None
                },
                {
                    "step_name": "PCA降维",
                    "step_type": "pca",
                    "status": "pending",
                    "progress_percentage": 0,
                    "start_time": None,
                    "end_time": None,
                    "duration_ms": None,
                    "error_message": None
                },
                {
                    "step_name": "t-SNE降维",
                    "step_type": "tsne",
                    "status": "pending",
                    "progress_percentage": 0,
                    "start_time": None,
                    "end_time": None,
                    "duration_ms": None,
                    "error_message": None
                },
                {
                    "step_name": "可视化生成",
                    "step_type": "visualization",
                    "status": "pending",
                    "progress_percentage": 0,
                    "start_time": None,
                    "end_time": None,
                    "duration_ms": None,
                    "error_message": None
                }
            ],
            "error_message": None,
            "estimated_duration_ms": pipeline_info["estimated_duration_ms"]
        }

        # 启动后台真实计算任务：直接基于上传的数据文件运行预处理+PCA+t-SNE，并落盘到 results/{pipeline_id}
        try:
            import asyncio
            import pandas as _pd
            import numpy as _np
            from src.algorithms.preprocessing import DataPreprocessor
            from src.algorithms.pca import PCAProcessor
            from src.algorithms.tsne import TSNEProcessor
            from .visualizations import _visualizations_store

            dataset_dir = upload_service.upload_dir / dataset_id
            data_file = dataset_dir / "data.csv"

            async def _run_real_compute():
                try:
                    df = _pd.read_csv(data_file)
                    # 取样本ID列
                    sid_col = 'sample_id' if 'sample_id' in df.columns else None
                    if not sid_col:
                        df['sample_id'] = [f"sample_{i:06d}" for i in range(len(df))]
                        sid_col = 'sample_id'
                    sample_ids = df[sid_col].astype(str).tolist()

                    # 数值特征
                    num_df = df.select_dtypes(include=[_np.number]).copy()
                    X = num_df.values.astype(float)

                    # 预处理
                    pre_cfg = default_config.get('preprocessing_config', {})
                    pre = DataPreprocessor(pre_cfg)
                    pre_res = pre.fit_transform(X)

                    # PCA
                    pca_cfg = default_config.get('pca_config', {})
                    pca = PCAProcessor(pca_cfg)
                    pca_res = pca.fit_transform(pre_res.data)

                    # t-SNE
                    tsne_cfg = default_config.get('tsne_config', {})
                    tsne = TSNEProcessor(tsne_cfg)
                    tsne_res = tsne.fit_transform(pca_res.data)

                    # 写结果
                    out_dir = Path('results') / pipeline_id
                    out_dir.mkdir(parents=True, exist_ok=True)
                    coords_df = _pd.DataFrame(tsne_res.data, columns=['x','y'])
                    coords_df['sample_id'] = sample_ids[:len(coords_df)]
                    # 类别（可选）
                    cat_col = None
                    for c in ['category', 'Category', 'label']:
                        if c in df.columns:
                            cat_col = c; break
                    if cat_col:
                        coords_df['category'] = df[cat_col].astype(str).values[:len(coords_df)]
                    coords_df.to_csv(out_dir / 'tsne_coordinates.csv', index=False)

                    # 保存处理元数据（供报告/metrics使用）
                    meta = {
                        'pipeline_id': pipeline_id,
                        'preprocessing': {
                            'config': pre_cfg,
                            'metadata': pre_res.metadata,
                            'processing_time_ms': pre_res.processing_time_ms
                        },
                        'pca': {
                            'config': pca_cfg,
                            'metadata': pca_res.metadata,
                            'processing_time_ms': pca_res.processing_time_ms
                        },
                        'tsne': {
                            'config': tsne_cfg,
                            'metadata': tsne_res.metadata,
                            'processing_time_ms': tsne_res.processing_time_ms
                        },
                        'total_samples': int(len(coords_df)),
                        'timestamp': datetime.now().isoformat()
                    }
                    with open(out_dir / 'processing_metadata.json', 'w', encoding='utf-8') as f:
                        import json as _json
                        _json.dump(meta, f, ensure_ascii=False, indent=2)

                    # 若包含自动调参摘要，落盘保存
                    if 'tuning_summary' in default_config and default_config['tuning_summary']:
                        with open(out_dir / 'tuning_summary.json', 'w', encoding='utf-8') as f:
                            import json as _json
                            _json.dump(default_config['tuning_summary'], f, ensure_ascii=False, indent=2)

                    # 注册可视化存储
                    cats = []
                    if 'category' in coords_df.columns:
                        counts = coords_df['category'].value_counts().to_dict()
                        name_to_id = {name: idx for idx, name in enumerate(sorted(counts.keys()))}
                        cats = [
                            {"category_id": cid, "category_name": name, "sample_count": int(counts[name]), "color_code": "#999999"}
                            for name, cid in name_to_id.items()
                        ]
                        coords_payload = []
                        for _, r in coords_df.iterrows():
                            name = str(r['category'])
                            cid = name_to_id.get(name, 0)
                            coords_payload.append({
                                "sample_id": str(r['sample_id']),
                                "x": float(r['x']),
                                "y": float(r['y']),
                                "category_id": int(cid),
                                "category_name": name,
                                "distance_to_center": float((r['x']**2 + r['y']**2) ** 0.5),
                                "local_density": 0.0
                            })
                    else:
                        coords_payload = [
                            {
                                "sample_id": str(coords_df.loc[i, 'sample_id']),
                                "x": float(coords_df.loc[i, 'x']),
                                "y": float(coords_df.loc[i, 'y']),
                                "category_id": 0,
                                "category_name": "unknown",
                                "distance_to_center": float((coords_df.loc[i, 'x']**2 + coords_df.loc[i, 'y']**2) ** 0.5),
                                "local_density": 0.0
                            }
                            for i in range(len(coords_df))
                        ]

                    _visualizations_store[pipeline_id] = {
                        "visualization_id": pipeline_id,
                        "pipeline_id": pipeline_id,
                        "dataset_id": dataset_id,
                        "status": "completed",
                        "config": {"chart_title": "t-SNE", "width": 1200, "height": 800, "show_legend": True},
                        "coordinates": coords_payload,
                        "categories": cats,
                        "statistics": {"total_samples": len(coords_payload), "valid_samples": len(coords_payload)},
                        "render_time_ms": 0,
                        "created_at": datetime.now().isoformat()
                    }

                    # 更新流水线/数据集状态
                    _pipelines_store[pipeline_id]["status"] = "completed"
                    _pipelines_store[pipeline_id]["end_time"] = datetime.now().isoformat()
                    await upload_service.update_dataset_status(dataset_id, {
                        "processing_status": "completed",
                        "latest_pipeline": {
                            "pipeline_id": pipeline_id,
                            "status": "completed",
                            "start_time": start_time.isoformat(),
                            "end_time": _pipelines_store[pipeline_id]["end_time"]
                        }
                    })
                except Exception as _e:
                    logger.error(f"后台计算失败: {str(_e)}")
                    _pipelines_store[pipeline_id]["status"] = "failed"
                    _pipelines_store[pipeline_id]["error_message"] = str(_e)

            asyncio.create_task(_run_real_compute())
        except Exception as _:
            # 后台任务失败不影响立即响应，前端仍可轮询到占位进度
            pass

        # 返回统一结构，符合前端期望 { success, data }
        return {
            "success": True,
            "data": {
                "pipeline_id": pipeline_id,
                "dataset_id": dataset_id,
                "status": "running",
                "estimated_duration_ms": pipeline_info["estimated_duration_ms"],
                "start_time": start_time.isoformat(),
                "message": "数据处理任务已启动",
                "config": default_config
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"启动数据处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail="启动数据处理失败")


@router.get("/{dataset_id}/status")
async def get_dataset_status(dataset_id: str):
    """
    获取数据集状态信息 - T043

    Args:
        dataset_id: 数据集ID

    Returns:
        Dict: 数据集状态信息

    Raises:
        HTTPException: 数据集不存在
    """
    try:
        # 使用上传服务获取真实的数据集信息
        dataset = await upload_service.get_dataset(dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="数据集不存在")

        # 模拟处理状态
        processing_status = dataset.get("processing_status", "uploaded")
        latest_pipeline = dataset.get("latest_pipeline")

        return {
            "dataset_id": dataset_id,
            "filename": dataset.get("filename"),
            "upload_timestamp": dataset.get("created_at"),
            "total_rows": dataset.get("row_count", 0),
            "total_columns": dataset.get("column_count", 0),
            "file_size_bytes": dataset.get("file_size", 0),
            "encoding": dataset.get("encoding", "utf-8"),
            "data_quality_score": dataset.get("data_quality_score", 0.0),
            "processing_status": processing_status,
            "categories": dataset.get("categories", []),
            "latest_pipeline": latest_pipeline,
            "available_operations": [
                "process_data",  # 启动数据处理
                "preview_data",  # 预览数据
                "export_data",   # 导出数据
                "delete_data"    # 删除数据
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取数据集状态信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail="获取数据集状态信息失败")


@router.get("/{dataset_id}/preview")
async def get_dataset_preview(
    dataset_id: str,
    limit: int = Query(10, ge=1, le=100)
):
    """
    获取数据集预览

    Args:
        dataset_id: 数据集ID
        limit: 预览行数

    Returns:
        Dict: 数据集预览数据

    Raises:
        HTTPException: 数据集不存在
    """
    try:
        # 使用上传服务获取真实的数据集预览
        dataset = await upload_service.get_dataset(dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="数据集不存在")

        # 读取实际数据文件进行预览
        import pandas as pd
        from pathlib import Path

        dataset_dir = upload_service.upload_dir / dataset_id
        data_file = dataset_dir / "data.csv"

        if not data_file.exists():
            raise HTTPException(status_code=404, detail="数据文件不存在")

        # 读取CSV文件
        df = pd.read_csv(data_file, nrows=limit)

        # 转换为字典格式
        data = df.to_dict('records')
        columns = df.columns.tolist()

        # 获取总行数
        total_rows = len(pd.read_csv(data_file))

        return {
            "dataset_id": dataset_id,
            "columns": columns,
            "data": data,
            "total_rows": total_rows,
            "preview_rows": len(data)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取数据集预览失败: {str(e)}")
        raise HTTPException(status_code=500, detail="获取数据集预览失败")


@router.post("/{dataset_id}/auto-config")
async def auto_tune_parameters(dataset_id: str, request: Dict[str, Any] | None = None):
    """
    网格搜索小范围自动调参

    - 基于数据规模动态生成 t-SNE perplexity 候选，交叉 metric（euclidean/manhattan）
    - 预处理/PCA 使用各自的 recommend_parameters 并在本次搜索中实际应用一次
    - 为搜索提速：若样本数过大，将子采样至最多 5000 行

    Returns:
        {
          success: true,
          data: {
            recommended_config: { preprocessing_config, pca_config, tsne_config },
            search_space: { perplexity: [...], metric: [...] },
            results: [{ perplexity, metric, kl_divergence, success, error? }]
          }
        }
    """
    try:
        import pandas as pd
        import numpy as np
        from pathlib import Path
        from src.services.upload_service import UploadService
        from src.algorithms.pca import PCAProcessor
        from src.algorithms.tsne import TSNEProcessor
        from sklearn.preprocessing import StandardScaler

        req = request or {}
        # 读取数据
        upload_service = UploadService()
        dataset = await upload_service.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="数据集不存在")

        dataset_dir = upload_service.upload_dir / dataset_id
        data_file = dataset_dir / "data.csv"
        if not data_file.exists():
            raise HTTPException(status_code=404, detail="数据文件不存在")

        df = pd.read_csv(data_file)

        # 选择数值特征
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        # 去掉明显不是特征的列（如二值/标识）可选：这里不过滤，交给 PCA/TSNE 处理
        if numeric_df.shape[1] == 0:
            raise HTTPException(status_code=400, detail="数据集中没有数值特征")

        # 缺失填充（均值）
        numeric_df = numeric_df.fillna(numeric_df.mean(numeric_only=True))
        X = numeric_df.values.astype(float)

        n_samples, n_features = X.shape

        # 子采样以加速搜索
        max_search_rows = int(req.get('max_search_rows', 5000))
        rng = np.random.default_rng(42)
        if n_samples > max_search_rows:
            idx = rng.choice(n_samples, size=max_search_rows, replace=False)
            X_small = X[idx]
        else:
            X_small = X

        # 标准化
        scaler = StandardScaler()
        X_small = scaler.fit_transform(X_small)

        # PCA 维度候选（与请求可覆盖）：常用 [10,12,15,20]，受限于特征数
        max_feat = int(X_small.shape[1])
        default_pca_list = [k for k in [10, 12, 15, 20] if k <= max_feat]
        if not default_pca_list:
            default_pca_list = [min(max_feat, 10)]
        pca_list = request.get('pca_components') if request and request.get('pca_components') else default_pca_list
        pca_list = sorted(set(int(k) for k in pca_list if 1 <= int(k) <= max_feat))

        # 生成 perplexity 候选
        def perplexity_candidates(n: int) -> list[int]:
            cands: list[int]
            if n < 100:
                cands = [max(5, n // 10)]
            elif n < 1000:
                cands = [10, 20, 30, 40]
            elif n < 5000:
                cands = [20, 30, 50]
            else:
                cands = [30, 50, 80, 100]
            # 约束 perplexity < n/5
            upper = max(6, n // 5 - 1)
            cands = sorted(set([c for c in cands if 5 <= c <= upper]))
            if not cands:
                cands = [max(5, min(50, upper))]
            return cands

        # 在未计算前无法知道 X_pca 样本数，这里先用 X_small 行数给出候选，后续每个 PCA 结果都能使用同一组
        perps = req.get('perplexities') or perplexity_candidates(X_small.shape[0])
        metrics = req.get('metrics') or ['euclidean', 'manhattan']
        lrs = req.get('learning_rates') or [200, 500]
        exaggerations = req.get('early_exaggerations') or [12.0, 24.0]

        # 其他 t-SNE 固定参数
        if X_small.shape[0] < 1000:
            n_iter = 1000
        elif X_small.shape[0] < 5000:
            n_iter = 1500
        else:
            n_iter = 2000

        results = []
        best = None

        for pca_n in pca_list:
            # 拟合 PCA（白化，减噪）
            pca_cfg = {'n_components': int(pca_n), 'whiten': True, 'svd_solver': 'auto', 'random_state': 42}
            pca = PCAProcessor(pca_cfg)
            pca_res = pca.fit_transform(X_small)
            if not pca_res.success:
                results.append({'pca_n': pca_n, 'success': False, 'error': pca_res.error_message})
                continue
            X_pca = pca_res.data

            for metric in metrics:
                for perp in perps:
                    for lr in lrs:
                        for ex in exaggerations:
                            cfg = {
                                'perplexity': int(perp),
                                'n_components': 2,
                                'learning_rate': float(lr),
                                'n_iter': n_iter,
                                'random_state': 42,
                                'metric': metric,
                                'early_exaggeration': float(ex)
                            }
                            try:
                                tsne = TSNEProcessor(cfg)
                                tsne.fit(X_pca)
                                kl = float(tsne.get_kl_divergence() or 1e9)
                                rec = {
                                    'pca_n': int(pca_n), 'perplexity': int(perp), 'metric': metric,
                                    'learning_rate': float(lr), 'early_exaggeration': float(ex),
                                    'kl_divergence': kl, 'success': True
                                }
                                results.append(rec)
                                if best is None or kl < best['kl_divergence']:
                                    best = rec
                            except Exception as e:
                                results.append({
                                    'pca_n': int(pca_n), 'perplexity': int(perp), 'metric': metric,
                                    'learning_rate': float(lr), 'early_exaggeration': float(ex),
                                    'success': False, 'error': str(e)
                                })

        # 若全部失败，退回默认推荐
        if not best:
            tsne_rec = TSNEProcessor.recommend_parameters(X_pca)
        else:
            tsne_rec = {
                'perplexity': int(best['perplexity']),
                'n_components': 2,
                'learning_rate': float(best.get('learning_rate', 200.0)),
                'n_iter': n_iter,
                'random_state': 42,
                'metric': best['metric'],
                'early_exaggeration': float(best.get('early_exaggeration', 24.0))
            }

        # 推荐的 PCA 配置（若有 best）
        if best:
            pca_rec = {'n_components': int(best['pca_n']), 'whiten': True, 'random_state': 42}
        else:
            # 退回保守默认
            pca_rec = {'n_components': default_pca_list[0], 'whiten': True, 'random_state': 42}

        # 预处理推荐（简单启发式）
        from src.algorithms.preprocessing import DataPreprocessor
        pre_rec = DataPreprocessor.recommend_parameters(X)

        # JSON 安全转换，避免 numpy.* 导致编码失败
        def to_py(obj):
            import numpy as _np  # local alias
            from datetime import datetime as _dt
            try:
                if isinstance(obj, dict):
                    return {str(k): to_py(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple, set)):
                    return [to_py(v) for v in obj]
                if isinstance(obj, _np.ndarray):
                    return obj.tolist()
                if isinstance(obj, _np.generic):
                    return obj.item()
                # 常见标量类型直接返回
                if isinstance(obj, (str, int, float, bool)) or obj is None:
                    return obj
                # pandas/时间类
                if hasattr(obj, 'isoformat'):
                    try:
                        return obj.isoformat()
                    except Exception:
                        pass
                return obj
            except Exception:
                return str(obj)

        payload = {
            'success': True,
            'data': {
                'recommended_config': {
                    'preprocessing_config': pre_rec,
                    'pca_config': pca_rec,
                    'tsne_config': tsne_rec
                },
                'search_space': {
                    'pca_components': pca_list,
                    'perplexity': perps,
                    'metric': metrics,
                    'learning_rate': lrs,
                    'early_exaggeration': exaggerations
                },
                'results': results,
                'sampled_rows_for_search': int(X_small.shape[0]),
                'original_rows': int(n_samples),
                'original_features': int(n_features)
            }
        }
        return to_py(payload)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"自动调参失败: {e}")
        raise HTTPException(status_code=500, detail="自动调参失败")


async def add_test_dataset():
    """添加测试数据集（用于契约测试）"""
    from pathlib import Path
    import pandas as pd

    # 创建测试数据
    test_data = pd.DataFrame({
        "sample_id": [f"sample_{i:03d}" for i in range(1, 101)],
        "feature_1": [1.2 + i * 0.1 for i in range(100)],
        "feature_2": [3.4 + i * 0.2 for i in range(100)],
        "category": ["MOF_A"] * 50 + ["MOF_B"] * 30 + ["MOF_C"] * 20
    })

    # 创建临时CSV文件
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data.to_csv(f.name, index=False)
        temp_file_path = f.name

    try:
        # 模拟上传文件
        from fastapi import UploadFile
        with open(temp_file_path, 'rb') as f:
            upload_file = UploadFile(filename="test_data.csv", file=f)
            dataset = await upload_service.process_upload(upload_file)

        return dataset.dataset_id
    finally:
        # 清理临时文件
        os.unlink(temp_file_path)
