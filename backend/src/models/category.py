"""
分类标签模型

遵循SDD Constitution的Library-First原则
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class CategoryLabel(BaseModel):
    """分类标签模型"""

    category_id: int = Field(ge=1, le=4)
    category_name: str
    display_name: str
    color_code: str = Field(default="#000000", pattern="^#[0-9A-Fa-f]{6}$")
    description: Optional[str] = None
    sample_count: int = Field(default=0, ge=0)

    @validator('category_name')
    def validate_category_name(cls, v):
        """验证分类名称"""
        valid_names = ['Category_A', 'Category_B', 'Category_C', 'Category_D']
        if v not in valid_names:
            raise ValueError(f'无效的分类名称: {v}')
        return v

    @validator('category_id')
    def validate_category_id(cls, v, values):
        """验证分类ID与名称一致性"""
        if 'category_name' in values:
            category_map = {
                'Category_A': 1,
                'Category_B': 2,
                'Category_C': 3,
                'Category_D': 4
            }
            expected_id = category_map.get(values['category_name'])
            if expected_id and v != expected_id:
                raise ValueError(f'分类ID {v} 与名称 {values["category_name"]} 不匹配')
        return v


class CategoryConfig(BaseModel):
    """分类配置模型"""

    categories: List[CategoryLabel]
    default_colors: Dict[str, str] = Field(default_factory=lambda: {
        'Category_A': '#1f77b4',
        'Category_B': '#ff7f0e',
        'Category_C': '#2ca02c',
        'Category_D': '#d62728'
    })

    @validator('categories')
    def validate_categories(cls, v):
        """验证分类列表"""
        if len(v) != 4:
            raise ValueError('必须包含4个分类')

        category_names = [cat.category_name for cat in v]
        expected_names = ['Category_A', 'Category_B', 'Category_C', 'Category_D']

        for expected in expected_names:
            if expected not in category_names:
                raise ValueError(f'缺少分类: {expected}')

        return v


class CategoryStats(BaseModel):
    """分类统计模型"""

    category_id: int
    category_name: str
    count: int
    percentage: float = Field(ge=0.0, le=100.0)
    avg_features: Dict[str, float]
    std_features: Dict[str, float]


class CategoryDistribution(BaseModel):
    """分类分布模型"""

    total_samples: int
    categories: List[CategoryStats]
    distribution_balance: float = Field(ge=0.0, le=1.0)  # 分布平衡度，1为完全平衡

    @validator('distribution_balance')
    def validate_balance(cls, v):
        """验证分布平衡度"""
        if not 0.0 <= v <= 1.0:
            raise ValueError('分布平衡度必须在0-1之间')
        return v


class CategoryMapping(BaseModel):
    """分类映射模型"""

    id_to_name: Dict[int, str] = Field(default_factory=lambda: {
        1: 'Category_A',
        2: 'Category_B',
        3: 'Category_C',
        4: 'Category_D'
    })

    name_to_id: Dict[str, int] = Field(default_factory=lambda: {
        'Category_A': 1,
        'Category_B': 2,
        'Category_C': 3,
        'Category_D': 4
    })

    id_to_color: Dict[int, str] = Field(default_factory=lambda: {
        1: '#1f77b4',
        2: '#ff7f0e',
        3: '#2ca02c',
        4: '#d62728'
    })

    def get_category_id(self, name: str) -> Optional[int]:
        """根据名称获取分类ID"""
        return self.name_to_id.get(name)

    def get_category_name(self, category_id: int) -> Optional[str]:
        """根据ID获取分类名称"""
        return self.id_to_name.get(category_id)

    def get_category_color(self, category_id: int) -> Optional[str]:
        """根据ID获取分类颜色"""
        return self.id_to_color.get(category_id)

    def validate_category(self, category_name: str, category_id: int) -> bool:
        """验证分类信息"""
        expected_id = self.get_category_id(category_name)
        return expected_id == category_id


class PredefinedCategories(CategoryConfig):
    """预定义分类配置"""

    def __init__(self, **data):
        # 设置默认分类
        default_categories = [
            CategoryLabel(
                category_id=1,
                category_name='Category_A',
                display_name='类别A',
                color_code='#1f77b4',
                description='第一类MOF材料'
            ),
            CategoryLabel(
                category_id=2,
                category_name='Category_B',
                display_name='类别B',
                color_code='#ff7f0e',
                description='第二类MOF材料'
            ),
            CategoryLabel(
                category_id=3,
                category_name='Category_C',
                display_name='类别C',
                color_code='#2ca02c',
                description='第三类MOF材料'
            ),
            CategoryLabel(
                category_id=4,
                category_name='Category_D',
                display_name='类别D',
                color_code='#d62728',
                description='第四类MOF材料'
            )
        ]

        super().__init__(categories=default_categories, **data)