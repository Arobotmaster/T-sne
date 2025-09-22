"""
处理流水线模型

遵循SDD Constitution的Library-First原则
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
from uuid import uuid4


class ProcessingPipeline(BaseModel):
    """处理流水线模型"""

    pipeline_id: str = Field(default_factory=lambda: str(uuid4()))
    dataset_id: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: str = Field(default="pending", pattern="^(pending|running|completed|failed)$")
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    error_message: Optional[str] = None
    total_duration_ms: Optional[int] = None

    @validator('progress_percentage')
    def validate_progress(cls, v, values):
        """验证进度百分比"""
        if 'status' in values:
            status = values['status']
            if status == 'completed' and v != 100.0:
                raise ValueError('已完成状态的进度必须为100%')
            if status == 'pending' and v != 0.0:
                raise ValueError('待处理状态的进度必须为0%')
        return v

    def update_progress(self, percentage: float, message: str = None):
        """更新进度"""
        self.progress_percentage = max(0.0, min(100.0, percentage))
        if message:
            self.error_message = message

    def mark_completed(self):
        """标记为已完成"""
        self.status = "completed"
        self.progress_percentage = 100.0
        self.end_time = datetime.now()
        if self.start_time:
            self.total_duration_ms = int((self.end_time - self.start_time).total_seconds() * 1000)

    def mark_failed(self, error_message: str):
        """标记为失败"""
        self.status = "failed"
        self.error_message = error_message
        self.end_time = datetime.now()
        if self.start_time:
            self.total_duration_ms = int((self.end_time - self.start_time).total_seconds() * 1000)


class DataProcessingStep(BaseModel):
    """数据处理步骤模型"""

    step_id: str = Field(default_factory=lambda: str(uuid4()))
    pipeline_id: str
    step_name: str
    step_type: str = Field(pattern="^(preprocessing|pca|tsne|visualization|export)$")
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: str = Field(default="pending", pattern="^(pending|running|completed|failed)$")
    input_rows: int = 0
    output_rows: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    error_message: Optional[str] = None

    @validator('step_type')
    def validate_step_type(cls, v):
        """验证步骤类型"""
        valid_types = ['preprocessing', 'pca', 'tsne', 'visualization', 'export']
        if v not in valid_types:
            raise ValueError(f'无效的步骤类型: {v}')
        return v

    def start_execution(self):
        """开始执行步骤"""
        self.start_time = datetime.now()
        self.status = "running"

    def complete_execution(self, output_rows: int = None):
        """完成执行"""
        self.end_time = datetime.now()
        self.status = "completed"
        if output_rows is not None:
            self.output_rows = output_rows

    def fail_execution(self, error_message: str):
        """执行失败"""
        self.end_time = datetime.now()
        self.status = "failed"
        self.error_message = error_message


class AlgorithmConfig(BaseModel):
    """算法配置模型"""

    config_id: str = Field(default_factory=lambda: str(uuid4()))
    step_id: str
    algorithm_name: str
    parameters: Dict[str, Any]
    version: str = "1.0.0"

    @validator('algorithm_name')
    def validate_algorithm_name(cls, v):
        """验证算法名称"""
        valid_algorithms = ['pca', 'tsne', 'standard_scaler', 'minmax_scaler']
        if v.lower() not in valid_algorithms:
            raise ValueError(f'无效的算法名称: {v}')
        return v.lower()


class TSNEConfig(BaseModel):
    """t-SNE配置模型"""

    perplexity: float = Field(default=30.0, ge=5.0, le=50.0)
    n_components: int = Field(default=2, ge=1, le=3)
    learning_rate: float = Field(default=200.0, ge=10.0, le=1000.0)
    n_iter: int = Field(default=1000, ge=250, le=2000)
    random_state: Optional[int] = Field(default=42)
    metric: str = Field(default="euclidean", pattern="^(euclidean|manhattan|chebyshev|minkowski)$")
    angle: float = Field(default=0.5, ge=0.1, le=0.8)
    early_exaggeration: float = Field(default=12.0, ge=1.0, le=50.0)


class PCAConfig(BaseModel):
    """PCA配置模型"""

    n_components: Optional[int] = Field(default=50, ge=2, le=100)
    whiten: bool = Field(default=False)
    svd_solver: str = Field(default="auto", pattern="^(auto|full|arpack|randomized)$")
    random_state: Optional[int] = Field(default=42)
    tol: float = Field(default=0.0, ge=0.0)


class PreprocessingConfig(BaseModel):
    """预处理配置模型"""

    handle_missing: str = Field(default="mean", pattern="^(mean|median|most_frequent|drop)$")
    scaling_method: str = Field(default="standard", pattern="^(standard|minmax|robust|none)$")
    remove_outliers: bool = Field(default=True)
    outlier_threshold: float = Field(default=3.0, ge=1.5, le=5.0)


class ProcessingResult(BaseModel):
    """处理结果模型"""

    result_id: str = Field(default_factory=lambda: str(uuid4()))
    step_id: str
    result_type: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    file_path: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    @validator('result_type')
    def validate_result_type(cls, v):
        """验证结果类型"""
        valid_types = ['pca_result', 'tsne_result', 'preprocessing_result', 'visualization_data']
        if v not in valid_types:
            raise ValueError(f'无效的结果类型: {v}')
        return v


class PipelineStepResult(BaseModel):
    """流水线步骤结果"""

    step_id: str
    step_name: str
    status: str
    duration_ms: int
    output_shape: Optional[tuple] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Dict[str, str] = Field(default_factory=dict)  # 文件路径等