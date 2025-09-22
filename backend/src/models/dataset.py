"""
MOF数据模型

遵循SDD Constitution的Library-First原则，
实现独立可测试的数据模型。
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
from uuid import uuid4
import numpy as np


class MOFDataset(BaseModel):
    """MOF数据集模型"""

    dataset_id: str = Field(default_factory=lambda: str(uuid4()))
    filename: str
    upload_timestamp: datetime = Field(default_factory=datetime.now)
    total_rows: int
    total_columns: int
    file_size_bytes: int
    encoding: str = "utf-8"
    separator: str = ","
    data_quality_score: float = Field(ge=0.0, le=1.0)

    @validator('data_quality_score')
    def validate_quality_score(cls, v):
        """验证数据质量分数"""
        if not 0.0 <= v <= 1.0:
            raise ValueError('数据质量分数必须在0-1之间')
        return v

    @validator('encoding')
    def validate_encoding(cls, v):
        """验证文件编码"""
        valid_encodings = ['utf-8', 'utf-8-sig', 'gbk', 'ascii']
        if v.lower() not in valid_encodings:
            raise ValueError(f'不支持的编码格式: {v}')
        # 将 utf-8-sig 统一转换为 utf-8
        if v.lower() == 'utf-8-sig':
            return 'utf-8'
        return v.lower()

    class Config:
        """Pydantic配置"""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class MOFSample(BaseModel):
    """MOF样本模型"""

    sample_id: str
    dataset_id: str
    row_index: int
    category_label: str
    category_id: int = Field(ge=1, le=4)
    is_valid: bool = True
    validation_errors: List[str] = Field(default_factory=list)

    @validator('category_label')
    def validate_category_label(cls, v):
        """验证分类标签"""
        valid_categories = ['Category_A', 'Category_B', 'Category_C', 'Category_D']
        if v not in valid_categories:
            raise ValueError(f'无效的分类标签: {v}')
        return v

    @validator('category_id')
    def validate_category_id(cls, v, values):
        """验证分类ID与标签一致性"""
        if 'category_label' in values:
            category_map = {
                'Category_A': 1,
                'Category_B': 2,
                'Category_C': 3,
                'Category_D': 4
            }
            expected_id = category_map.get(values['category_label'])
            if expected_id and v != expected_id:
                raise ValueError(f'分类ID {v} 与标签 {values["category_label"]} 不匹配')
        return v


class NumericalFeature(BaseModel):
    """数值特征模型"""

    feature_id: str = Field(default_factory=lambda: str(uuid4()))
    sample_id: str
    feature_name: str
    feature_value: Optional[float] = None
    is_missing: bool = False
    imputed_value: Optional[float] = None
    normalized_value: Optional[float] = None
    z_score: Optional[float] = None
    is_outlier: bool = False

    @validator('feature_value')
    def validate_feature_value(cls, v):
        """验证特征值"""
        if v is not None and not np.isfinite(v):
            raise ValueError('特征值必须是有限数值')
        return v

    @validator('z_score')
    def validate_z_score(cls, v):
        """验证Z分数"""
        if v is not None and abs(v) > 5:
            raise ValueError('Z分数绝对值过大，可能是异常值')
        return v


class DescriptiveData(BaseModel):
    """描述性数据模型"""

    data_id: str = Field(default_factory=lambda: str(uuid4()))
    sample_id: str
    field_name: str
    field_value: str
    data_type: str = Field(default="text", pattern="^(text|url|identifier)$")


class DatasetStatistics(BaseModel):
    """数据集统计信息"""

    dataset_id: str
    total_samples: int
    valid_samples: int
    missing_value_rate: float
    feature_statistics: Dict[str, Dict[str, float]]
    category_distribution: Dict[str, int]
    quality_metrics: Dict[str, Any]


class DatasetProcessingResult(BaseModel):
    """数据集处理结果"""

    dataset_id: str
    processing_status: str = Field(pattern="^(pending|processing|completed|failed)$")
    processed_samples: int
    processing_time_ms: int
    error_message: Optional[str] = None
    result_metadata: Dict[str, Any] = Field(default_factory=dict)


class DatasetInfo(BaseModel):
    """数据集信息模型"""

    dataset_id: str
    filename: str
    file_size: int
    row_count: int
    column_count: int
    data_quality_score: float
    columns: List[Dict[str, Any]]
    categories: List[str]
    missing_values: Dict[str, int]
    data_types: Dict[str, str]
    created_at: str
    updated_at: str


class UploadResponse(BaseModel):
    """上传响应模型"""

    success: bool
    data: Optional[Dict[str, Any]] = None
    message: str
    processing_time: Optional[float] = None