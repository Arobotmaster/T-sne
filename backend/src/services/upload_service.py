"""
文件上传服务

遵循SDD Constitution原则：
- Library-First: 独立可测试的服务组件
- CLI Interface: 支持命令行调用
- Scientific Observability: 详细的处理日志记录
"""

import os
import csv
import uuid
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import chardet
import pandas as pd
import numpy as np
from fastapi import UploadFile, HTTPException
from tempfile import NamedTemporaryFile

from ..models.dataset import MOFDataset, MOFSample, NumericalFeature, DescriptiveData
from ..models.category import CategoryLabel
from ..config.settings import settings
from ..config.logging_config import get_logger
from ..utils.temp_file_manager import get_temp_file_manager

logger = get_logger(__name__)


class UploadService:
    """文件上传服务

    负责处理CSV文件上传、数据验证、质量评估和数据集创建
    """

    # 支持的编码格式
    SUPPORTED_ENCODINGS = ['utf-8', 'gbk', 'ascii', 'utf-8-sig']

    # 文件大小限制 (100MB)
    MAX_FILE_SIZE = 100 * 1024 * 1024

    # 数据质量阈值
    MIN_DATA_QUALITY_SCORE = 0.3
    MAX_MISSING_RATIO = 0.3

    def __init__(self):
        """
        初始化上传服务
        """
        self.upload_dir = Path(settings.UPLOAD_DIRECTORY)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.temp_file_manager = get_temp_file_manager()
        logger.info(f"UploadService initialized with directory: {self.upload_dir}")

    async def process_upload(self, file: UploadFile) -> MOFDataset:
        """
        处理文件上传主流程

        Args:
            file: 上传的文件对象

        Returns:
            MOFDataset: 创建的数据集对象

        Raises:
            ValueError: 数据验证失败
            HTTPException: 文件处理错误
        """
        start_time = datetime.now()
        logger.info(f"Starting file upload processing: {file.filename}")

        try:
            # 1. 验证文件基本属性
            await self._validate_file(file)

            # 2. 保存临时文件
            temp_file_path = await self._save_temp_file(file)

            # 3. 检测文件编码
            encoding = await self._detect_encoding(temp_file_path)

            # 4. 解析CSV数据
            data, column_info = await self._parse_csv(temp_file_path, encoding)

            # 5. 验证数据质量
            quality_score = self._calculate_data_quality(data, column_info)

            # 6. 创建数据集对象
            dataset = await self._create_dataset(file, data, column_info, quality_score, encoding)

            # 7. 保存处理后的数据
            await self._save_dataset_data(dataset, data)

            # 8. 清理临时文件
            await self._cleanup_temp_file(temp_file_path)

            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"File upload completed successfully. Processing time: {processing_time:.2f}ms")
            logger.info(f"Dataset created: {dataset.dataset_id}, Quality score: {quality_score:.3f}")

            return dataset

        except Exception as e:
            logger.error(f"File upload processing failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"文件处理失败: {str(e)}")

    async def _validate_file(self, file: UploadFile) -> None:
        """验证文件基本属性"""
        # 检查文件名
        if not file.filename or not file.filename.lower().endswith('.csv'):
            raise ValueError("仅支持CSV格式文件")

        # 检查文件大小
        if file.size and file.size > self.MAX_FILE_SIZE:
            raise ValueError(f"文件大小超过限制 (最大 {self.MAX_FILE_SIZE // (1024*1024)}MB)")

        # 检查内容类型
        if file.content_type and not file.content_type.startswith('text/'):
            logger.warning(f"Unexpected content type: {file.content_type}")

        logger.info(f"File validation passed: {file.filename}")

    async def _save_temp_file(self, file: UploadFile) -> Path:
        """保存临时文件"""
        try:
            # 使用临时文件管理器创建临时文件
            temp_file_path = self.temp_file_manager.create_temp_file(
                prefix="upload_",
                suffix=".csv"
            )

            # 注册文件到临时文件管理器
            file_id = self.temp_file_manager.register_file(
                str(temp_file_path),
                purpose="upload",
                original_filename=file.filename
            )

            # 逐块读取文件内容
            with open(temp_file_path, 'wb') as temp_file:
                chunk_size = 8192
                while chunk := await file.read(chunk_size):
                    temp_file.write(chunk)

            logger.info(f"Temporary file saved: {temp_file_path}")
            return temp_file_path

        except Exception as e:
            raise ValueError(f"文件保存失败: {str(e)}")

    async def _detect_encoding(self, file_path: Path) -> str:
        """检测文件编码"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(1024)  # 读取前1KB用于编码检测

            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']

            if encoding.lower() not in self.SUPPORTED_ENCODINGS:
                logger.warning(f"Detected encoding {encoding} not in supported list, using utf-8")
                encoding = 'utf-8'

            logger.info(f"Encoding detected: {encoding} (confidence: {confidence:.2f})")
            return encoding

        except Exception as e:
            logger.warning(f"Encoding detection failed, using utf-8: {str(e)}")
            return 'utf-8'

    async def _parse_csv(self, file_path: Path, encoding: str) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """解析CSV文件"""
        try:
            # 尝试不同的分隔符
            separators = [',', ';', '\t', '|']
            df = None
            used_separator = ','

            for sep in separators:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, sep=sep, nrows=10)
                    if len(df.columns) > 1:  # 至少需要2列
                        used_separator = sep
                        break
                except:
                    continue

            if df is None:
                raise ValueError("无法解析CSV文件，请检查文件格式")

            # 重新读取完整文件
            df = pd.read_csv(file_path, encoding=encoding, sep=used_separator)

            # 清理列名
            df.columns = df.columns.str.strip().str.replace(' ', '_')

            # 分析列信息
            column_info = []
            for col in df.columns:
                col_type = self._detect_column_type(df[col])
                nullable_ratio = df[col].isnull().sum() / len(df)
                unique_values = df[col].nunique()

                column_info.append({
                    'name': col,
                    'type': col_type,
                    'nullable_ratio': nullable_ratio,
                    'unique_values': unique_values,
                    'sample_values': df[col].dropna().head(3).tolist()
                })

            logger.info(f"CSV parsed successfully. Shape: {df.shape}, Separator: '{used_separator}'")
            return df, column_info

        except Exception as e:
            raise ValueError(f"CSV解析失败: {str(e)}")

    def _detect_column_type(self, series: pd.Series) -> str:
        """检测列数据类型"""
        # 尝试转换为数值类型
        try:
            pd.to_numeric(series, errors='raise')
            return 'numerical'
        except:
            pass

        # 检查是否为分类数据
        unique_ratio = series.nunique() / len(series)
        if unique_ratio < 0.1 or series.nunique() <= 10:
            return 'categorical'

        return 'descriptive'

    def _calculate_data_quality(self, df: pd.DataFrame, column_info: List[Dict[str, Any]]) -> float:
        """计算数据质量分数"""
        quality_factors = []

        # 1. 完整性评分
        completeness = 1 - sum(col['nullable_ratio'] for col in column_info) / len(column_info)
        quality_factors.append(completeness)

        # 2. 数据类型一致性评分
        type_consistency = 1.0
        for col in column_info:
            if col['nullable_ratio'] > self.MAX_MISSING_RATIO:
                type_consistency -= 0.1
        type_consistency = max(0, type_consistency)
        quality_factors.append(type_consistency)

        # 3. 数据量评分
        data_volume_score = min(1.0, len(df) / 1000)  # 1000行得满分
        quality_factors.append(data_volume_score)

        # 4. 列数评分
        column_score = min(1.0, len(df.columns) / 10)  # 10列得满分
        quality_factors.append(column_score)

        # 计算综合分数
        overall_score = np.mean(quality_factors)

        logger.info(f"Data quality calculated: {overall_score:.3f}")
        logger.info(f"Quality factors: completeness={completeness:.3f}, "
                   f"type_consistency={type_consistency:.3f}, "
                   f"data_volume={data_volume_score:.3f}, "
                   f"columns={column_score:.3f}")

        return overall_score

    async def _create_dataset(self, file: UploadFile, df: pd.DataFrame,
                             column_info: List[Dict[str, Any]],
                             quality_score: float, encoding: str) -> MOFDataset:
        """创建数据集对象"""
        dataset = MOFDataset(
            filename=file.filename or "unknown.csv",
            total_rows=len(df),
            total_columns=len(df.columns),
            file_size_bytes=file.size or 0,
            encoding=encoding,
            data_quality_score=quality_score
        )

        logger.info(f"Dataset object created: {dataset.dataset_id}")
        return dataset

    async def _save_dataset_data(self, dataset: MOFDataset, df: pd.DataFrame) -> None:
        """保存数据集数据"""
        # 创建数据集目录
        dataset_dir = self.upload_dir / dataset.dataset_id
        dataset_dir.mkdir(exist_ok=True)

        # 保存原始数据
        data_file = dataset_dir / "data.csv"
        df.to_csv(data_file, index=False)

        # 保存元数据
        metadata_file = dataset_dir / "metadata.json"
        import json
        metadata = {
            'dataset_id': dataset.dataset_id,
            'filename': dataset.filename,
            'upload_timestamp': dataset.upload_timestamp.isoformat(),
            'total_rows': dataset.total_rows,
            'total_columns': dataset.total_columns,
            'file_size_bytes': dataset.file_size_bytes,
            'encoding': dataset.encoding,
            'data_quality_score': dataset.data_quality_score,
            'columns': df.columns.tolist()
        }

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"Dataset data saved to: {dataset_dir}")

    async def _cleanup_temp_file(self, temp_file_path: Path) -> None:
        """清理临时文件"""
        try:
            file_id = temp_file_path.name
            if self.temp_file_manager.delete_file(file_id):
                logger.info(f"Temporary file cleaned up via manager: {temp_file_path}")
            else:
                # 如果管理器中没有找到，直接删除
                temp_file_path.unlink(missing_ok=True)
                logger.info(f"Temporary file cleaned up directly: {temp_file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary file: {str(e)}")

    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """获取数据集信息"""
        dataset_dir = self.upload_dir / dataset_id
        if not dataset_dir.exists():
            return None

        metadata_file = dataset_dir / "metadata.json"
        if not metadata_file.exists():
            return None

        try:
            import json
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read dataset metadata: {str(e)}")
            return None

    async def delete_dataset(self, dataset_id: str) -> bool:
        """删除数据集"""
        dataset_dir = self.upload_dir / dataset_id
        if not dataset_dir.exists():
            return False

        try:
            import shutil
            shutil.rmtree(dataset_dir)
            logger.info(f"Dataset deleted: {dataset_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete dataset: {str(e)}")
            return False

    async def list_datasets(self) -> List[Dict[str, Any]]:
        """获取所有数据集列表"""
        datasets = []

        try:
            # 遍历上传目录中的所有子目录
            for dataset_dir in self.upload_dir.iterdir():
                if dataset_dir.is_dir():
                    metadata_file = dataset_dir / "metadata.json"
                    if metadata_file.exists():
                        try:
                            import json
                            with open(metadata_file, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                                datasets.append(metadata)
                        except Exception as e:
                            logger.error(f"Failed to read metadata for dataset {dataset_dir.name}: {str(e)}")
                            continue

            # 按创建时间降序排序
            datasets.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            logger.info(f"Listed {len(datasets)} datasets")
            return datasets

        except Exception as e:
            logger.error(f"Failed to list datasets: {str(e)}")
            return []

    async def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """获取指定数据集的详细信息"""
        try:
            dataset_dir = self.upload_dir / dataset_id
            if not dataset_dir.exists():
                return None

            metadata_file = dataset_dir / "metadata.json"
            if not metadata_file.exists():
                return None

            import json
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            logger.info(f"Retrieved dataset: {dataset_id}")
            return metadata

        except Exception as e:
            logger.error(f"Failed to get dataset {dataset_id}: {str(e)}")
            return None

    async def update_dataset_status(self, dataset_id: str, status_updates: Dict[str, Any]) -> bool:
        """更新数据集状态"""
        try:
            dataset_dir = self.upload_dir / dataset_id
            if not dataset_dir.exists():
                return False

            metadata_file = dataset_dir / "metadata.json"
            if not metadata_file.exists():
                return False

            import json
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # 更新状态信息
            metadata.update(status_updates)
            metadata['updated_at'] = datetime.now().isoformat()

            # 保存更新后的元数据
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            logger.info(f"Updated dataset status: {dataset_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update dataset status {dataset_id}: {str(e)}")
            return False