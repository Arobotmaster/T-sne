"""
可视化模型

遵循SDD Constitution的Library-First原则
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
from uuid import uuid4


class VisualizationConfig(BaseModel):
    """可视化配置模型"""

    config_id: str = Field(default_factory=lambda: str(uuid4()))
    pipeline_id: str
    chart_title: str = Field(default="MOF数据t-SNE可视化")
    x_axis_label: str = Field(default="t-SNE维度1")
    y_axis_label: str = Field(default="t-SNE维度2")
    width: int = Field(default=1200, ge=400, le=4000)
    height: int = Field(default=800, ge=300, le=3000)
    show_legend: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.now)

    @validator('width', 'height')
    def validate_dimensions(cls, v):
        """验证图表尺寸"""
        if v < 400 or v > 4000:
            raise ValueError('图表尺寸必须在400-4000像素之间')
        return v


class ColorScheme(BaseModel):
    """颜色方案模型"""

    scheme_id: str = Field(default_factory=lambda: str(uuid4()))
    config_id: str
    category_id: int = Field(ge=1, le=4)
    color_hex: str = Field(default="#000000", pattern="^#[0-9A-Fa-f]{6}$")
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)
    border_color: Optional[str] = Field(default=None, pattern="^#[0-9A-Fa-f]{6}$")

    @validator('opacity')
    def validate_opacity(cls, v):
        """验证透明度"""
        if not 0.0 <= v <= 1.0:
            raise ValueError('透明度必须在0-1之间')
        return v


class MarkerStyle(BaseModel):
    """标记样式模型"""

    style_id: str = Field(default_factory=lambda: str(uuid4()))
    config_id: str
    category_id: int = Field(ge=1, le=4)
    marker_type: str = Field(default="circle", pattern="^(circle|square|triangle|diamond|cross|x)$")
    size: int = Field(default=8, ge=2, le=50)
    line_width: int = Field(default=1, ge=0, le=10)

    @validator('marker_type')
    def validate_marker_type(cls, v):
        """验证标记类型"""
        valid_types = ['circle', 'square', 'triangle', 'diamond', 'cross', 'x']
        if v not in valid_types:
            raise ValueError(f'无效的标记类型: {v}')
        return v


class ExportConfig(BaseModel):
    """导出配置模型"""

    export_id: str = Field(default_factory=lambda: str(uuid4()))
    config_id: str
    format: str = Field(default="png", pattern="^(png|svg|pdf)$")
    width: int = Field(default=1200, ge=400, le=4000)
    height: int = Field(default=800, ge=300, le=3000)
    dpi: int = Field(default=300, ge=72, le=600)
    background_color: str = Field(default="#ffffff", pattern="^#[0-9A-Fa-f]{6}$")
    filename: Optional[str] = None

    @validator('format')
    def validate_format(cls, v):
        """验证导出格式"""
        valid_formats = ['png', 'svg', 'pdf']
        if v.lower() not in valid_formats:
            raise ValueError(f'无效的导出格式: {v}')
        return v.lower()

    @validator('dpi')
    def validate_dpi(cls, v):
        """验证DPI"""
        if v < 72 or v > 600:
            raise ValueError('DPI必须在72-600之间')
        return v


class TSNECoordinates(BaseModel):
    """t-SNE坐标模型"""

    coordinate_id: str = Field(default_factory=lambda: str(uuid4()))
    visualization_id: str
    sample_id: str
    x_coordinate: float
    y_coordinate: float
    distance_to_center: float = Field(default=0.0, ge=0.0)
    local_density: float = Field(default=0.0, ge=0.0)
    category_id: int = Field(ge=1, le=4)
    category_name: str

    @validator('distance_to_center', 'local_density')
    def validate_metrics(cls, v):
        """验证度量指标"""
        if v < 0:
            raise ValueError('距离和密度必须是非负数')
        return v


class VisualizationData(BaseModel):
    """可视化数据模型"""

    coordinates: List[TSNECoordinates]
    config: Dict[str, Any]
    metadata: Dict[str, Any]
    total_samples: int
    categories: List[Dict[str, Any]]
    color_mapping: Dict[int, str]


class VisualizationResult(BaseModel):
    """可视化结果模型"""

    visualization_id: str = Field(default_factory=lambda: str(uuid4()))
    pipeline_id: str
    config_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    chart_data: Dict[str, Any]
    total_samples: int
    render_time_ms: int
    file_path: Optional[str] = None

    @validator('render_time_ms')
    def validate_render_time(cls, v):
        """验证渲染时间"""
        if v < 0:
            raise ValueError('渲染时间不能为负数')
        return v


class ExportResult(BaseModel):
    """导出结果模型"""

    export_id: str = Field(default_factory=lambda: str(uuid4()))
    visualization_id: str
    format: str
    file_path: str
    file_size_bytes: int
    export_time_ms: int
    created_at: datetime = Field(default_factory=datetime.now)

    @validator('file_size_bytes', 'export_time_ms')
    def validate_positive_values(cls, v):
        """验证正值"""
        if v < 0:
            raise ValueError('文件大小和导出时间必须为正数')
        return v


class ChartLayout(BaseModel):
    """图表布局模型"""

    title: str = Field(default="MOF数据t-SNE可视化")
    xaxis: Dict[str, Any] = Field(default_factory=lambda: {"title": "t-SNE维度1"})
    yaxis: Dict[str, Any] = Field(default_factory=lambda: {"title": "t-SNE维度2"})
    width: int = Field(default=1200)
    height: int = Field(default=800)
    showlegend: bool = Field(default=True)
    hovermode: str = Field(default="closest")
    plot_bgcolor: str = Field(default="white")
    paper_bgcolor: str = Field(default="white")


class PlotlyTrace(BaseModel):
    """Plotly轨迹模型"""

    x: List[float]
    y: List[float]
    mode: str = Field(default="markers")
    type: str = Field(default="scatter")
    text: List[str]
    marker: Dict[str, Any]
    name: str
    showlegend: bool = Field(default=True)


class VisualizationRequest(BaseModel):
    """可视化请求模型"""

    pipeline_id: str
    config: Optional[VisualizationConfig] = None
    custom_layout: Optional[ChartLayout] = None
    include_metadata: bool = Field(default=True)