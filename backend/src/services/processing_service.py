"""
数据处理服务

遵循SDD Constitution原则：
- Library-First: 独立可测试的服务组件
- CLI Interface: 支持命令行调用
- Scientific Observability: 详细的处理日志记录
"""

import os
import json
import uuid
import logging
import asyncio
import psutil
import time
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from ..models.pipeline import ProcessingPipeline, DataProcessingStep, AlgorithmConfig, ProcessingResult
from ..algorithms.preprocessing import DataPreprocessor
from ..algorithms.pca import PCAProcessor
from ..algorithms.tsne import TSNEProcessor
from ..config.logging_config import get_logger

logger = get_logger(__name__)


class ProcessingService:
    """数据处理服务

    负责协调数据预处理、PCA降维、t-SNE降维等处理步骤
    """

    def __init__(self, data_dir: str = "data", results_dir: str = "results"):
        """
        初始化处理服务

        Args:
            data_dir: 数据文件目录
            results_dir: 处理结果目录
        """
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 活跃的流水线管理
        self.active_pipelines: Dict[str, ProcessingPipeline] = {}
        self.pipeline_locks: Dict[str, asyncio.Lock] = {}

        logger.info(f"ProcessingService initialized. Data dir: {data_dir}, Results dir: {results_dir}")

    async def start_processing(self, dataset_id: str, config: Dict[str, Any]) -> ProcessingPipeline:
        """
        启动数据处理流水线

        Args:
            dataset_id: 数据集ID
            config: 处理配置

        Returns:
            ProcessingPipeline: 创建的处理流水线

        Raises:
            ValueError: 配置验证失败或数据集不存在
        """
        logger.info(f"Starting processing pipeline for dataset: {dataset_id}")

        # 验证数据集是否存在
        dataset_path = self.data_dir / "uploads" / dataset_id
        if not dataset_path.exists():
            raise ValueError(f"数据集不存在: {dataset_id}")

        # 验证配置
        validated_config = self._validate_config(config)

        # 创建处理流水线
        pipeline = ProcessingPipeline(
            dataset_id=dataset_id,
            start_time=datetime.now(),
            status="running"
        )

        # 注册流水线
        self.active_pipelines[pipeline.pipeline_id] = pipeline
        self.pipeline_locks[pipeline.pipeline_id] = asyncio.Lock()

        logger.info(f"Processing pipeline created: {pipeline.pipeline_id}")

        # 异步启动处理任务
        asyncio.create_task(self._execute_pipeline(pipeline.pipeline_id, validated_config))

        return pipeline

    async def start_processing_from_file(self,
                                         file_path: str,
                                         config: Dict[str, Any],
                                         id_column: str = 'sample_id',
                                         category_column: Optional[str] = None,
                                         embedding_label: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        直接从CSV文件启动处理（开发/快速通道）

        Args:
            file_path: CSV 文件路径（相对项目根或绝对路径）
            config: 处理配置（preprocessing_config/pca_config/tsne_config）
            id_column: 样本ID列名
            category_column: 类别列名（可选）
            embedding_label: 嵌入标签（如 E1/E2），便于结果归档

        Returns:
            (pipeline_id, summary): 流水线ID与摘要信息
        """
        csv_path = Path(file_path)
        if not csv_path.exists():
            raise ValueError(f"CSV文件不存在: {file_path}")

        # 读取数据
        df = pd.read_csv(csv_path)
        if id_column not in df.columns:
            # 若无 sample_id，则生成
            df[id_column] = [f"sample_{i:06d}" for i in range(len(df))]

        # 提取标签列
        y = None
        if category_column and category_column in df.columns:
            y = df[category_column].astype(str)

        # 选择数值特征列（排除ID与类别）
        drop_cols = {id_column}
        if category_column and category_column in df.columns:
            drop_cols.add(category_column)
        X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        X = X.select_dtypes(include=[np.number]).copy()

        if X.shape[1] == 0:
            raise ValueError("未找到可用的数值特征列")

        # 预处理
        pre_cfg = config.get('preprocessing_config', {})
        preprocessor = DataPreprocessor(pre_cfg)
        pre_res = preprocessor.fit_transform(X.values)
        if not pre_res.success:
            raise ValueError(pre_res.error_message or '预处理失败')
        Xp = pre_res.data

        # PCA
        pca_cfg = config.get('pca_config', {}).copy()
        # 若指定的主成分数超过特征维度，进行截断以避免异常
        if isinstance(pca_cfg.get('n_components', None), int):
            max_nc = min(Xp.shape[0], Xp.shape[1])
            if pca_cfg['n_components'] > max_nc:
                logger.warning(
                    f"调整 PCA n_components: {pca_cfg['n_components']} -> {max_nc} (features={Xp.shape[1]})"
                )
                pca_cfg['n_components'] = max_nc
        pca = PCAProcessor(pca_cfg)
        pca_res = pca.fit_transform(Xp)
        if not pca_res.success:
            raise ValueError(pca_res.error_message or 'PCA失败')
        Xpca = pca_res.data

        # t-SNE
        tsne_cfg = config.get('tsne_config', {})
        tsne = TSNEProcessor(tsne_cfg)
        tsne_res = tsne.fit_transform(Xpca)
        if not tsne_res.success:
            raise ValueError(tsne_res.error_message or 't-SNE失败')
        Xt = tsne_res.data

        # 结果组织
        coords = pd.DataFrame({
            'sample_id': df[id_column].values,
            'x': Xt[:, 0],
            'y': Xt[:, 1]
        })
        if y is not None:
            coords['category'] = y.values

        # 写入结果目录
        pipeline_id = str(uuid.uuid4())
        out_dir = self.results_dir / pipeline_id
        out_dir.mkdir(parents=True, exist_ok=True)
        coords.to_csv(out_dir / 'tsne_coordinates.csv', index=False)

        metadata = {
            'pipeline_id': pipeline_id,
            'source_file': str(csv_path),
            'embedding': embedding_label,
            'input_shape': [int(X.shape[0]), int(X.shape[1])],
            'pca_shape': [int(Xpca.shape[0]), int(Xpca.shape[1])],
            'tsne_params': tsne_cfg,
            'created_at': datetime.now().isoformat()
        }
        with open(out_dir / 'processing_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        summary = {
            'n_samples': int(X.shape[0]),
            'n_features': int(X.shape[1]),
            'has_category': bool(category_column and category_column in df.columns),
            'category_column': category_column
        }
        return pipeline_id, summary

    async def get_pipeline_status(self, pipeline_id: str) -> Optional[ProcessingPipeline]:
        """获取处理流水线状态"""
        pipeline = self.active_pipelines.get(pipeline_id)
        if pipeline:
            return pipeline.copy() if hasattr(pipeline, 'copy') else pipeline
        return None

    async def _execute_pipeline(self, pipeline_id: str, config: Dict[str, Any]) -> None:
        """
        执行处理流水线

        Args:
            pipeline_id: 流水线ID
            config: 处理配置
        """
        async with self.pipeline_locks[pipeline_id]:
            pipeline = self.active_pipelines[pipeline_id]
            start_time = time.time()

            try:
                logger.info(f"Executing pipeline: {pipeline_id}")

                # 步骤1: 数据加载
                await self._update_step_progress(pipeline, "数据加载", 0, 10)
                data, metadata = await self._load_dataset_data(pipeline.dataset_id)

                # 步骤2: 数据预处理
                await self._update_step_progress(pipeline, "数据预处理", 10, 30)
                preprocessing_result = await self._run_preprocessing(data, config.get('preprocessing_config', {}))

                # 步骤3: PCA降维
                await self._update_step_progress(pipeline, "PCA降维", 30, 60)
                pca_result = await self._run_pca(preprocessing_result.data, config.get('pca_config', {}))

                # 步骤4: t-SNE降维
                await self._update_step_progress(pipeline, "t-SNE降维", 60, 90)
                tsne_result = await self._run_tsne(pca_result.data, config.get('tsne_config', {}))

                # 步骤5: 保存结果
                await self._update_step_progress(pipeline, "保存结果", 90, 100)
                await self._save_processing_results(pipeline_id, {
                    'preprocessing': preprocessing_result,
                    'pca': pca_result,
                    'tsne': tsne_result
                }, metadata)

                # 完成处理
                pipeline.status = "completed"
                pipeline.progress_percentage = 100.0
                pipeline.end_time = datetime.now()
                pipeline.total_duration_ms = int((time.time() - start_time) * 1000)

                logger.info(f"Pipeline completed successfully: {pipeline_id}")

            except Exception as e:
                logger.error(f"Pipeline execution failed: {pipeline_id}, Error: {str(e)}")

                pipeline.status = "failed"
                pipeline.error_message = str(e)
                pipeline.end_time = datetime.now()
                pipeline.total_duration_ms = int((time.time() - start_time) * 1000)

            finally:
                # 清理锁
                if pipeline_id in self.pipeline_locks:
                    del self.pipeline_locks[pipeline_id]

                # 保存最终状态
                await self._save_pipeline_status(pipeline)

    async def _update_step_progress(self, pipeline: ProcessingPipeline, step_name: str,
                                  start_progress: float, end_progress: float) -> None:
        """更新步骤进度"""
        pipeline.current_step = step_name
        pipeline.progress_percentage = start_progress

        step = DataProcessingStep(
            step_id=str(uuid.uuid4()),
            pipeline_id=pipeline.pipeline_id,
            step_name=step_name,
            step_type="processing",
            start_time=datetime.now(),
            status="running"
        )

        logger.info(f"Starting step: {step_name} for pipeline: {pipeline.pipeline_id}")

    async def _load_dataset_data(self, dataset_id: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """加载数据集数据"""
        dataset_path = self.data_dir / "uploads" / dataset_id / "data.csv"
        metadata_path = self.data_dir / "uploads" / dataset_id / "metadata.json"

        if not dataset_path.exists():
            raise ValueError(f"数据文件不存在: {dataset_path}")

        # 加载元数据
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

        # 加载数据
        df = pd.read_csv(dataset_path)

        logger.info(f"Dataset loaded: {dataset_id}, Shape: {df.shape}")
        return df, metadata

    async def _run_preprocessing(self, data: pd.DataFrame, config: Dict[str, Any]) -> Any:
        """运行数据预处理"""
        logger.info("Running preprocessing step")

        # 提取数值列
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_columns:
            raise ValueError("数据中未找到数值列")

        numeric_data = data[numeric_columns].values

        # 创建预处理器
        processor = DataPreprocessor(config)

        # 执行预处理
        result = processor.fit_transform(numeric_data)

        logger.info(f"Preprocessing completed. Input shape: {numeric_data.shape}, "
                   f"Output shape: {result.data.shape}")

        return result

    async def _run_pca(self, data: np.ndarray, config: Dict[str, Any]) -> Any:
        """运行PCA降维"""
        logger.info("Running PCA step")

        processor = PCAProcessor(config)
        result = processor.fit_transform(data)

        logger.info(f"PCA completed. Input shape: {data.shape}, "
                   f"Output shape: {result.data.shape}, "
                   f"Explained variance: {result.metadata.get('cumulative_variance', [0])[-1]:.3f}")

        return result

    async def _run_tsne(self, data: np.ndarray, config: Dict[str, Any]) -> Any:
        """运行t-SNE降维"""
        logger.info("Running t-SNE step")

        processor = TSNEProcessor(config)
        result = processor.fit_transform(data)

        logger.info(f"t-SNE completed. Input shape: {data.shape}, "
                   f"Output shape: {result.data.shape}")

        return result

    async def _save_processing_results(self, pipeline_id: str, results: Dict[str, Any],
                                      metadata: Dict[str, Any]) -> None:
        """保存处理结果"""
        pipeline_dir = self.results_dir / pipeline_id
        pipeline_dir.mkdir(exist_ok=True)

        # 保存t-SNE坐标
        tsne_coordinates = results['tsne'].data
        coordinates_df = pd.DataFrame(tsne_coordinates, columns=['x', 'y'])

        # 添加样本ID和分类信息
        if 'sample_ids' in metadata:
            coordinates_df['sample_id'] = metadata['sample_ids'][:len(coordinates_df)]

        coordinates_file = pipeline_dir / "tsne_coordinates.csv"
        coordinates_df.to_csv(coordinates_file, index=False)

        # 保存处理元数据
        processing_metadata = {
            'pipeline_id': pipeline_id,
            'preprocessing': {
                'config': results['preprocessing'].config,
                'metadata': results['preprocessing'].metadata,
                'processing_time_ms': results['preprocessing'].processing_time_ms
            },
            'pca': {
                'config': results['pca'].config,
                'metadata': results['pca'].metadata,
                'processing_time_ms': results['pca'].processing_time_ms
            },
            'tsne': {
                'config': results['tsne'].config,
                'metadata': results['tsne'].metadata,
                'processing_time_ms': results['tsne'].processing_time_ms
            },
            'total_samples': len(coordinates_df),
            'timestamp': datetime.now().isoformat()
        }

        metadata_file = pipeline_dir / "processing_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(processing_metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"Processing results saved to: {pipeline_dir}")

    async def _save_pipeline_status(self, pipeline: ProcessingPipeline) -> None:
        """保存流水线状态"""
        status_file = self.results_dir / f"{pipeline.pipeline_id}_status.json"

        status_data = {
            'pipeline_id': pipeline.pipeline_id,
            'dataset_id': pipeline.dataset_id,
            'status': pipeline.status,
            'progress_percentage': pipeline.progress_percentage,
            'start_time': pipeline.start_time.isoformat() if pipeline.start_time else None,
            'end_time': pipeline.end_time.isoformat() if pipeline.end_time else None,
            'total_duration_ms': pipeline.total_duration_ms,
            'error_message': pipeline.error_message
        }

        with open(status_file, 'w', encoding='utf-8') as f:
            json.dump(status_data, f, ensure_ascii=False, indent=2)

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证处理配置"""
        validated_config = {
            'preprocessing_config': config.get('preprocessing_config', {}),
            'pca_config': config.get('pca_config', {}),
            'tsne_config': config.get('tsne_config', {})
        }

        # 设置默认配置
        if not validated_config['pca_config']:
            validated_config['pca_config'] = {'n_components': 50, 'random_state': 42}

        if not validated_config['tsne_config']:
            validated_config['tsne_config'] = {
                'perplexity': 30,
                'n_components': 2,
                'random_state': 42
            }

        if not validated_config['preprocessing_config']:
            validated_config['preprocessing_config'] = {
                'missing_value_strategy': 'mean',
                'scaling_method': 'standard'
            }

        logger.info(f"Configuration validated: {validated_config}")
        return validated_config

    def get_performance_metrics(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """获取性能指标"""
        pipeline_dir = self.results_dir / pipeline_id
        metadata_file = pipeline_dir / "processing_metadata.json"

        if not metadata_file.exists():
            return None

        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # 计算性能指标
            total_time = sum(
                metadata['preprocessing']['processing_time_ms'] +
                metadata['pca']['processing_time_ms'] +
                metadata['tsne']['processing_time_ms']
            )

            metrics = {
                'total_processing_time_ms': total_time,
                'preprocessing_time_ms': metadata['preprocessing']['processing_time_ms'],
                'pca_time_ms': metadata['pca']['processing_time_ms'],
                'tsne_time_ms': metadata['tsne']['processing_time_ms'],
                'memory_usage_mb': self._get_memory_usage(),
                'total_samples': metadata['total_samples']
            }

            return metrics

        except Exception as e:
            logger.error(f"Failed to get performance metrics: {str(e)}")
            return None

    def _get_memory_usage(self) -> float:
        """获取当前内存使用量"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0.0

    def cleanup_old_results(self, max_age_hours: int = 24) -> int:
        """清理旧的处理结果"""
        cleanup_count = 0
        current_time = datetime.now()

        for pipeline_dir in self.results_dir.iterdir():
            if pipeline_dir.is_dir():
                try:
                    # 检查目录年龄
                    dir_age = current_time - datetime.fromtimestamp(pipeline_dir.stat().st_mtime)
                    if dir_age.total_seconds() > max_age_hours * 3600:
                        import shutil
                        shutil.rmtree(pipeline_dir)
                        cleanup_count += 1
                        logger.info(f"Cleaned up old results: {pipeline_dir}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup directory {pipeline_dir}: {str(e)}")

        logger.info(f"Cleanup completed. Removed {cleanup_count} old result directories")
        return cleanup_count
