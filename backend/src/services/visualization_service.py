"""
可视化服务

遵循SDD Constitution原则：
- Library-First: 独立可测试的服务组件
- CLI Interface: 支持命令行调用
- Scientific Observability: 详细的处理日志记录
"""

import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde

from ..models.visualization import (
    VisualizationConfig, VisualizationResult, TSNECoordinates,
    ColorScheme, MarkerStyle
)
from ..models.category import CategoryLabel
from ..config.logging_config import get_logger

logger = get_logger(__name__)


class VisualizationService:
    """可视化服务

    负责生成t-SNE可视化数据、配置管理和统计分析
    """

    def __init__(self, results_dir: str = "results"):
        """
        初始化可视化服务

        Args:
            results_dir: 处理结果目录
        """
        self.results_dir = Path(results_dir)
        self.default_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
        ]
        self.default_markers = ['circle', 'square', 'triangle', 'diamond']

        logger.info(f"VisualizationService initialized. Results dir: {results_dir}")

    async def get_visualization_data(self, pipeline_id: str,
                                   config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        获取可视化数据

        Args:
            pipeline_id: 处理流水线ID
            config: 可视化配置

        Returns:
            Dict[str, Any]: 可视化数据，包含坐标、配置和统计信息

        Raises:
            ValueError: 流水线不存在或处理未完成
        """
        logger.info(f"Generating visualization data for pipeline: {pipeline_id}")

        # 检查流水线状态
        pipeline_status = self._get_pipeline_status(pipeline_id)
        if not pipeline_status:
            raise ValueError(f"处理流水线不存在: {pipeline_id}")

        if pipeline_status.get('status') != 'completed':
            raise ValueError(f"处理尚未完成，无法生成可视化: {pipeline_id}")

        # 加载t-SNE坐标数据
        coordinates_data = await self._load_tsne_coordinates(pipeline_id)
        if not coordinates_data:
            raise ValueError(f"无法加载t-SNE坐标数据: {pipeline_id}")

        # 加载原始数据以获取分类信息
        original_data = await self._load_original_data(pipeline_id)

        # 生成可视化数据
        visualization_data = await self._generate_visualization_data(
            coordinates_data, original_data, config
        )

        # 计算统计信息
        statistics = await self._calculate_visualization_statistics(coordinates_data)

        # 创建可视化结果
        result = {
            'visualization_id': str(pipeline_id),  # 使用pipeline_id作为visualization_id
            'pipeline_id': pipeline_id,
            'config': visualization_data['config'],
            'coordinates': visualization_data['coordinates'],
            'categories': visualization_data['categories'],
            'statistics': statistics,
            'render_time_ms': visualization_data.get('render_time_ms', 0)
        }

        logger.info(f"Visualization data generated successfully for pipeline: {pipeline_id}")
        return result

    async def update_visualization_config(self, pipeline_id: str,
                                         config: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新可视化配置

        Args:
            pipeline_id: 处理流水线ID
            config: 新的配置参数

        Returns:
            Dict[str, Any]: 更新后的配置
        """
        logger.info(f"Updating visualization config for pipeline: {pipeline_id}")

        # 验证配置
        validated_config = self._validate_visualization_config(config)

        # 保存配置
        config_file = self.results_dir / pipeline_id / "visualization_config.json"
        config_file.parent.mkdir(exist_ok=True)

        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(validated_config, f, ensure_ascii=False, indent=2)

        logger.info(f"Visualization config updated for pipeline: {pipeline_id}")
        return validated_config

    async def _load_tsne_coordinates(self, pipeline_id: str) -> Optional[pd.DataFrame]:
        """加载t-SNE坐标数据"""
        coordinates_file = self.results_dir / pipeline_id / "tsne_coordinates.csv"

        if not coordinates_file.exists():
            logger.error(f"Coordinates file not found: {coordinates_file}")
            return None

        try:
            df = pd.read_csv(coordinates_file)
            logger.info(f"Loaded {len(df)} coordinates for pipeline: {pipeline_id}")
            return df
        except Exception as e:
            logger.error(f"Failed to load coordinates: {str(e)}")
            return None

    async def _load_original_data(self, pipeline_id: str) -> Optional[pd.DataFrame]:
        """加载原始数据以获取分类信息"""
        # 从处理元数据中获取数据集ID
        metadata_file = self.results_dir / pipeline_id / "processing_metadata.json"
        if not metadata_file.exists():
            return None

        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # 这里可以根据需要加载原始数据的分类信息
            # 目前返回None，表示需要从其他地方获取分类信息
            return None
        except Exception as e:
            logger.warning(f"Failed to load original data metadata: {str(e)}")
            return None

    async def _generate_visualization_data(self, coordinates_df: pd.DataFrame,
                                         original_data: Optional[pd.DataFrame],
                                         config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """生成可视化数据"""
        start_time = datetime.now()

        # 应用默认配置
        if config is None:
            config = self._get_default_config()

        # 生成坐标数据
        coordinates = []
        for idx, row in coordinates_df.iterrows():
            coord = {
                'sample_id': row.get('sample_id', f'sample_{idx}'),
                'x': float(row['x']),
                'y': float(row['y']),
                'category_id': 1,  # 默认分类
                'category_name': 'Default',
                'distance_to_center': np.sqrt(row['x']**2 + row['y']**2),
                'local_density': 0.0  # 将在统计计算中更新
            }
            coordinates.append(coord)

        # 生成分类信息
        categories = [
            {
                'category_id': 1,
                'category_name': 'Default',
                'sample_count': len(coordinates),
                'color_code': self.default_colors[0]
            }
        ]

        # 应用自定义配置
        visualization_config = {
            'chart_title': config.get('chart_title', 'MOF数据t-SNE可视化'),
            'x_axis_label': config.get('x_axis_label', 't-SNE维度1'),
            'y_axis_label': config.get('y_axis_label', 't-SNE维度2'),
            'width': config.get('width', 1200),
            'height': config.get('height', 800),
            'show_legend': config.get('show_legend', True),
            'color_scheme': config.get('color_scheme', []),
            'marker_styles': config.get('marker_styles', [])
        }

        render_time = (datetime.now() - start_time).total_seconds() * 1000

        return {
            'config': visualization_config,
            'coordinates': coordinates,
            'categories': categories,
            'render_time_ms': int(render_time)
        }

    async def _calculate_visualization_statistics(self, coordinates_df: pd.DataFrame) -> Dict[str, Any]:
        """计算可视化统计信息"""
        try:
            coordinates = coordinates_df[['x', 'y']].values

            # 基础统计
            total_samples = len(coordinates)
            x_range = {'min': float(coordinates[:, 0].min()), 'max': float(coordinates[:, 0].max())}
            y_range = {'min': float(coordinates[:, 1].min()), 'max': float(coordinates[:, 1].max())}

            # 计算局部密度
            local_densities = self._calculate_local_densities(coordinates)

            # 密度分析
            high_density_threshold = np.percentile(local_densities, 75)
            low_density_threshold = np.percentile(local_densities, 25)

            high_density_regions = np.sum(local_densities > high_density_threshold)
            low_density_regions = np.sum(local_densities < low_density_threshold)
            average_density = float(np.mean(local_densities))

            statistics = {
                'total_samples': total_samples,
                'valid_samples': total_samples,  # 假设所有样本都有效
                'category_counts': {'Default': total_samples},
                'x_range': x_range,
                'y_range': y_range,
                'density_info': {
                    'high_density_regions': int(high_density_regions),
                    'low_density_regions': int(low_density_regions),
                    'average_density': average_density
                }
            }

            logger.info(f"Visualization statistics calculated: {statistics}")
            return statistics

        except Exception as e:
            logger.error(f"Failed to calculate statistics: {str(e)}")
            return {
                'total_samples': len(coordinates_df),
                'valid_samples': len(coordinates_df),
                'category_counts': {'Default': len(coordinates_df)},
                'x_range': {'min': 0, 'max': 0},
                'y_range': {'min': 0, 'max': 0},
                'density_info': {
                    'high_density_regions': 0,
                    'low_density_regions': 0,
                    'average_density': 0.0
                }
            }

    def _calculate_local_densities(self, coordinates: np.ndarray, n_neighbors: int = 10) -> np.ndarray:
        """计算局部密度"""
        try:
            # 计算距离矩阵
            distances = cdist(coordinates, coordinates)

            # 对每个点，取最近的n_neighbors个点的距离
            n_neighbors = min(n_neighbors, len(coordinates) - 1)
            nearest_distances = np.partition(distances, n_neighbors + 1, axis=1)[:, 1:n_neighbors + 1]

            # 计算局部密度（距离的倒数）
            local_densities = 1.0 / (np.mean(nearest_distances, axis=1) + 1e-8)

            return local_densities
        except Exception as e:
            logger.warning(f"Failed to calculate local densities: {str(e)}")
            return np.ones(len(coordinates))

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认可视化配置"""
        return {
            'chart_title': 'MOF数据t-SNE可视化',
            'x_axis_label': 't-SNE维度1',
            'y_axis_label': 't-SNE维度2',
            'width': 1200,
            'height': 800,
            'show_legend': True,
            'color_scheme': [
                {
                    'category_id': 1,
                    'color_hex': self.default_colors[0],
                    'opacity': 1.0
                }
            ],
            'marker_styles': [
                {
                    'category_id': 1,
                    'marker_type': 'circle',
                    'size': 6
                }
            ]
        }

    def _validate_visualization_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证可视化配置"""
        validated_config = self._get_default_config()
        validated_config.update(config)

        # 验证数值范围
        validated_config['width'] = max(400, min(4000, validated_config.get('width', 1200)))
        validated_config['height'] = max(300, min(3000, validated_config.get('height', 800)))

        # 验证颜色配置
        if 'color_scheme' in validated_config:
            for color_config in validated_config['color_scheme']:
                if 'color_hex' in color_config:
                    # 简单的颜色格式验证
                    color_hex = color_config['color_hex']
                    if not (color_hex.startswith('#') and len(color_hex) == 7):
                        color_config['color_hex'] = self.default_colors[0]

        logger.info(f"Visualization config validated")
        return validated_config

    def _get_pipeline_status(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """获取流水线状态"""
        status_file = self.results_dir / f"{pipeline_id}_status.json"

        if not status_file.exists():
            return None

        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load pipeline status: {str(e)}")
            return None

    async def get_visualization_config(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """获取可视化配置"""
        config_file = self.results_dir / pipeline_id / "visualization_config.json"

        if not config_file.exists():
            return self._get_default_config()

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load visualization config, using default: {str(e)}")
            return self._get_default_config()

    async def generate_comparison_data(self, pipeline_id_1: str, pipeline_id_2: str) -> Optional[Dict[str, Any]]:
        """生成对比可视化数据"""
        logger.info(f"Generating comparison data for pipelines: {pipeline_id_1} vs {pipeline_id_2}")

        # 加载两个流水线的坐标数据
        coords_1 = await self._load_tsne_coordinates(pipeline_id_1)
        coords_2 = await self._load_tsne_coordinates(pipeline_id_2)

        if coords_1 is None or coords_2 is None:
            logger.error("Failed to load coordinates for comparison")
            return None

        # 生成对比数据
        comparison_data = {
            'pipeline_1': {
                'pipeline_id': pipeline_id_1,
                'coordinates': coords_1.to_dict('records'),
                'color': 'rgba(31, 119, 180, 0.6)'  # 半透明蓝色
            },
            'pipeline_2': {
                'pipeline_id': pipeline_id_2,
                'coordinates': coords_2.to_dict('records'),
                'color': 'rgba(255, 127, 14, 0.8)'  # 不透明橙色
            },
            'comparison_statistics': self._calculate_comparison_statistics(coords_1, coords_2)
        }

        logger.info(f"Comparison data generated for pipelines: {pipeline_id_1} vs {pipeline_id_2}")
        return comparison_data

    def _calculate_comparison_statistics(self, coords_1: pd.DataFrame, coords_2: pd.DataFrame) -> Dict[str, Any]:
        """计算对比统计信息"""
        try:
            # 计算两个数据集的中心点
            center_1 = coords_1[['x', 'y']].mean()
            center_2 = coords_2[['x', 'y']].mean()

            # 计算中心点距离
            center_distance = np.sqrt((center_1['x'] - center_2['x'])**2 +
                                    (center_1['y'] - center_2['y'])**2)

            # 计算数据集大小差异
            size_diff = abs(len(coords_1) - len(coords_2))

            statistics = {
                'center_distance': float(center_distance),
                'size_difference': size_diff,
                'dataset_1_size': len(coords_1),
                'dataset_2_size': len(coords_2),
                'center_1': {'x': float(center_1['x']), 'y': float(center_1['y'])},
                'center_2': {'x': float(center_2['x']), 'y': float(center_2['y'])}
            }

            return statistics

        except Exception as e:
            logger.error(f"Failed to calculate comparison statistics: {str(e)}")
            return {}