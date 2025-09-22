"""
对比可视化服务

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
from scipy.stats import pearsonr, spearmanr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..config.logging_config import get_logger

logger = get_logger(__name__)


class ComparisonService:
    """对比可视化服务

    负责处理两个数据集的t-SNE对比可视化分析
    """

    def __init__(self, results_dir: str = "results"):
        """
        初始化对比服务

        Args:
            results_dir: 处理结果目录
        """
        self.results_dir = Path(results_dir)
        self.default_colors = {
            'background': 'rgba(31, 119, 180, 0.3)',  # 半透明蓝色
            'foreground': 'rgba(255, 127, 14, 0.8)',   # 不透明橙色
            'overlay': 'rgba(44, 160, 44, 0.6)'      # 半透明绿色
        }

        logger.info(f"ComparisonService initialized. Results dir: {results_dir}")

    async def create_comparison(self, original_pipeline_id: str,
                              filtered_pipeline_id: str,
                              config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        创建对比可视化

        Args:
            original_pipeline_id: 原始数据流水线ID
            filtered_pipeline_id: 筛选后数据流水线ID
            config: 对比配置

        Returns:
            Dict[str, Any]: 对比可视化结果

        Raises:
            ValueError: 流水线不存在或数据不完整
        """
        logger.info(f"Creating comparison: {original_pipeline_id} vs {filtered_pipeline_id}")

        try:
            # 验证流水线状态
            await self._validate_pipelines_for_comparison(original_pipeline_id, filtered_pipeline_id)

            # 加载两个数据集
            original_data = await self._load_pipeline_data(original_pipeline_id)
            filtered_data = await self._load_pipeline_data(filtered_pipeline_id)

            if not original_data or not filtered_data:
                raise ValueError("无法加载对比数据")

            # 应用配置
            comparison_config = self._apply_default_config(config)

            # 生成对比数据
            comparison_result = await self._generate_comparison_data(
                original_data, filtered_data, comparison_config
            )

            # 计算对比统计
            comparison_statistics = await self._calculate_comparison_statistics(
                original_data, filtered_data
            )

            # 生成可视化配置
            visualization_config = await self._generate_comparison_visualization_config(
                comparison_result, comparison_statistics, comparison_config
            )

            # 保存对比结果
            await self._save_comparison_result(
                original_pipeline_id, filtered_pipeline_id,
                comparison_result, comparison_statistics
            )

            result = {
                'comparison_id': self._generate_comparison_id(original_pipeline_id, filtered_pipeline_id),
                'original_pipeline_id': original_pipeline_id,
                'filtered_pipeline_id': filtered_pipeline_id,
                'config': comparison_config,
                'comparison_data': comparison_result,
                'statistics': comparison_statistics,
                'visualization_config': visualization_config,
                'created_at': datetime.now().isoformat()
            }

            logger.info(f"Comparison created successfully: {result['comparison_id']}")
            return result

        except Exception as e:
            logger.error(f"Failed to create comparison: {str(e)}")
            raise ValueError(f"对比创建失败: {str(e)}")

    async def get_comparison_data(self, comparison_id: str) -> Optional[Dict[str, Any]]:
        """获取对比数据"""
        comparison_file = self.results_dir / "comparisons" / f"{comparison_id}.json"

        if not comparison_file.exists():
            return None

        try:
            with open(comparison_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load comparison data: {str(e)}")
            return None

    async def update_comparison_config(self, comparison_id: str,
                                     config: Dict[str, Any]) -> Dict[str, Any]:
        """更新对比配置"""
        logger.info(f"Updating comparison config: {comparison_id}")

        # 获取现有对比数据
        comparison_data = await self.get_comparison_data(comparison_id)
        if not comparison_data:
            raise ValueError(f"对比不存在: {comparison_id}")

        # 更新配置
        comparison_data['config'].update(config)

        # 保存更新后的数据
        await self._save_comparison_data(comparison_id, comparison_data)

        logger.info(f"Comparison config updated: {comparison_id}")
        return comparison_data

    async def generate_comparison_plot_data(self, comparison_id: str) -> Optional[Dict[str, Any]]:
        """生成对比图表数据"""
        comparison_data = await self.get_comparison_data(comparison_id)
        if not comparison_data:
            return None

        try:
            # 创建背景层（原始数据）
            background_trace = self._create_background_trace(
                comparison_data['comparison_data']['original_coordinates'],
                comparison_data['config']
            )

            # 创建前景层（筛选数据）
            foreground_trace = self._create_foreground_trace(
                comparison_data['comparison_data']['filtered_coordinates'],
                comparison_data['config']
            )

            # 创建对比统计信息
            stats_trace = self._create_statistics_trace(comparison_data['statistics'])

            plot_data = {
                'background': background_trace,
                'foreground': foreground_trace,
                'statistics': stats_trace,
                'layout': self._generate_comparison_layout(comparison_data)
            }

            return plot_data

        except Exception as e:
            logger.error(f"Failed to generate comparison plot data: {str(e)}")
            return None

    async def _validate_pipelines_for_comparison(self, pipeline_id_1: str, pipeline_id_2: str) -> None:
        """验证流水线是否适合对比"""
        # 检查流水线状态
        status_1 = self._get_pipeline_status(pipeline_id_1)
        status_2 = self._get_pipeline_status(pipeline_id_2)

        if not status_1 or not status_2:
            raise ValueError("一个或多个流水线不存在")

        if status_1.get('status') != 'completed' or status_2.get('status') != 'completed':
            raise ValueError("一个或多个流水线处理未完成")

        logger.info(f"Pipeline validation passed for comparison")

    async def _load_pipeline_data(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """加载流水线数据"""
        try:
            # 加载坐标数据
            coordinates_file = self.results_dir / pipeline_id / "tsne_coordinates.csv"
            if not coordinates_file.exists():
                return None

            coordinates_df = pd.read_csv(coordinates_file)

            # 加载元数据
            metadata_file = self.results_dir / pipeline_id / "processing_metadata.json"
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

            return {
                'pipeline_id': pipeline_id,
                'coordinates': coordinates_df,
                'metadata': metadata,
                'sample_count': len(coordinates_df)
            }

        except Exception as e:
            logger.error(f"Failed to load pipeline data for {pipeline_id}: {str(e)}")
            return None

    def _apply_default_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """应用默认配置"""
        default_config = {
            'background_opacity': 0.3,
            'foreground_opacity': 0.8,
            'show_original': True,
            'show_filtered': True,
            'show_statistics': True,
            'color_scheme': {
                'background_color': self.default_colors['background'],
                'foreground_color': self.default_colors['foreground'],
                'overlay_color': self.default_colors['overlay']
            },
            'marker_size': {
                'background': 6,
                'foreground': 8
            },
            'chart_title': 'MOF数据对比可视化',
            'width': 1200,
            'height': 800
        }

        if config:
            default_config.update(config)

        return default_config

    async def _generate_comparison_data(self, original_data: Dict[str, Any],
                                       filtered_data: Dict[str, Any],
                                       config: Dict[str, Any]) -> Dict[str, Any]:
        """生成对比数据"""
        logger.info("Generating comparison data")

        # 处理坐标数据
        original_coords = original_data['coordinates'][['x', 'y']].values
        filtered_coords = filtered_data['coordinates'][['x', 'y']].values

        # 计算相对位置关系
        relative_positions = self._calculate_relative_positions(original_coords, filtered_coords)

        # 识别重叠区域
        overlap_regions = self._identify_overlap_regions(original_coords, filtered_coords)

        # 计算密度变化
        density_changes = self._calculate_density_changes(original_coords, filtered_coords)

        comparison_data = {
            'original_coordinates': original_data['coordinates'].to_dict('records'),
            'filtered_coordinates': filtered_data['coordinates'].to_dict('records'),
            'relative_positions': relative_positions,
            'overlap_regions': overlap_regions,
            'density_changes': density_changes,
            'sample_counts': {
                'original': len(original_coords),
                'filtered': len(filtered_coords),
                'retention_rate': len(filtered_coords) / len(original_coords) if len(original_coords) > 0 else 0
            }
        }

        return comparison_data

    async def _calculate_comparison_statistics(self, original_data: Dict[str, Any],
                                             filtered_data: Dict[str, Any]) -> Dict[str, Any]:
        """计算对比统计信息"""
        original_coords = original_data['coordinates'][['x', 'y']].values
        filtered_coords = filtered_data['coordinates'][['x', 'y']].values

        # 基础统计
        original_center = np.mean(original_coords, axis=0)
        filtered_center = np.mean(filtered_coords, axis=0)

        # 计算中心偏移
        center_shift = np.linalg.norm(filtered_center - original_center)

        # 计算分布变化
        original_std = np.std(original_coords, axis=0)
        filtered_std = np.std(filtered_coords, axis=0)

        # 计算相关性（如果样本有对应关系）
        correlation_data = self._calculate_sample_correlation(original_coords, filtered_coords)

        # 计算覆盖范围变化
        original_range = np.ptp(original_coords, axis=0)
        filtered_range = np.ptp(filtered_coords, axis=0)

        statistics = {
            'sample_counts': {
                'original': len(original_coords),
                'filtered': len(filtered_coords),
                'retention_rate': len(filtered_coords) / len(original_coords) if len(original_coords) > 0 else 0
            },
            'center_shift': {
                'distance': float(center_shift),
                'original_center': original_center.tolist(),
                'filtered_center': filtered_center.tolist()
            },
            'distribution_changes': {
                'original_std': original_std.tolist(),
                'filtered_std': filtered_std.tolist(),
                'std_change_ratio': (filtered_std / (original_std + 1e-8)).tolist()
            },
            'range_changes': {
                'original_range': original_range.tolist(),
                'filtered_range': filtered_range.tolist(),
                'range_change_ratio': (filtered_range / (original_range + 1e-8)).tolist()
            },
            'correlation': correlation_data,
            'quality_metrics': self._calculate_quality_metrics(original_coords, filtered_coords)
        }

        return statistics

    def _calculate_relative_positions(self, original_coords: np.ndarray, filtered_coords: np.ndarray) -> Dict[str, Any]:
        """计算相对位置关系"""
        try:
            # 计算原始数据的中心点
            original_center = np.mean(original_coords, axis=0)

            # 计算筛选点相对于原始中心的分布
            relative_positions = {
                'distance_from_original_center': [],
                'angle_from_original_center': [],
                'local_density': []
            }

            for coord in filtered_coords:
                # 计算距离
                distance = np.linalg.norm(coord - original_center)
                relative_positions['distance_from_original_center'].append(float(distance))

                # 计算角度
                diff = coord - original_center
                angle = np.arctan2(diff[1], diff[0])
                relative_positions['angle_from_original_center'].append(float(angle))

                # 计算局部密度
                local_density = self._calculate_point_density(coord, original_coords)
                relative_positions['local_density'].append(float(local_density))

            return relative_positions

        except Exception as e:
            logger.error(f"Failed to calculate relative positions: {str(e)}")
            return {}

    def _identify_overlap_regions(self, original_coords: np.ndarray, filtered_coords: np.ndarray) -> Dict[str, Any]:
        """识别重叠区域"""
        try:
            # 使用网格划分来识别重叠区域
            x_min = min(original_coords[:, 0].min(), filtered_coords[:, 0].min())
            x_max = max(original_coords[:, 0].max(), filtered_coords[:, 0].max())
            y_min = min(original_coords[:, 1].min(), filtered_coords[:, 1].min())
            y_max = max(original_coords[:, 1].max(), filtered_coords[:, 1].max())

            # 创建网格
            grid_size = 20
            x_bins = np.linspace(x_min, x_max, grid_size)
            y_bins = np.linspace(y_min, y_max, grid_size)

            overlap_regions = []
            for i in range(len(x_bins) - 1):
                for j in range(len(y_bins) - 1):
                    # 检查当前网格单元
                    x_mask = (original_coords[:, 0] >= x_bins[i]) & (original_coords[:, 0] < x_bins[i + 1])
                    y_mask = (original_coords[:, 1] >= y_bins[j]) & (original_coords[:, 1] < y_bins[j + 1])
                    original_in_cell = np.sum(x_mask & y_mask)

                    x_mask_f = (filtered_coords[:, 0] >= x_bins[i]) & (filtered_coords[:, 0] < x_bins[i + 1])
                    y_mask_f = (filtered_coords[:, 1] >= y_bins[j]) & (filtered_coords[:, 1] < y_bins[j + 1])
                    filtered_in_cell = np.sum(x_mask_f & y_mask_f)

                    if original_in_cell > 0 or filtered_in_cell > 0:
                        overlap_regions.append({
                            'grid_cell': (i, j),
                            'original_count': int(original_in_cell),
                            'filtered_count': int(filtered_in_cell),
                            'center_x': float((x_bins[i] + x_bins[i + 1]) / 2),
                            'center_y': float((y_bins[j] + y_bins[j + 1]) / 2),
                            'overlap_ratio': float(filtered_in_cell / (original_in_cell + 1e-8)) if original_in_cell > 0 else 0.0
                        })

            return {'regions': overlap_regions}

        except Exception as e:
            logger.error(f"Failed to identify overlap regions: {str(e)}")
            return {'regions': []}

    def _calculate_density_changes(self, original_coords: np.ndarray, filtered_coords: np.ndarray) -> Dict[str, Any]:
        """计算密度变化"""
        try:
            # 计算原始数据的密度分布
            original_density = self._estimate_density_2d(original_coords)
            filtered_density = self._estimate_density_2d(filtered_coords)

            # 计算密度变化
            density_ratio = filtered_density / (original_density + 1e-8)

            return {
                'original_density': original_density.tolist(),
                'filtered_density': filtered_density.tolist(),
                'density_ratio': density_ratio.tolist(),
                'density_change_score': float(np.std(density_ratio))
            }

        except Exception as e:
            logger.error(f"Failed to calculate density changes: {str(e)}")
            return {}

    def _estimate_density_2d(self, coords: np.ndarray, bandwidth: float = 1.0) -> np.ndarray:
        """估计2D密度"""
        if len(coords) == 0:
            return np.array([])

        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(coords.T, bw_method=bandwidth)
            density = kde(coords.T)
            return density
        except:
            # 如果无法使用KDE，使用简单的距离方法
            return self._calculate_simple_density(coords)

    def _calculate_simple_density(self, coords: np.ndarray, n_neighbors: int = 10) -> np.ndarray:
        """计算简单密度"""
        densities = np.zeros(len(coords))
        for i, coord in enumerate(coords):
            distances = np.linalg.norm(coords - coord, axis=1)
            nearest_distances = np.partition(distances, n_neighbors + 1)[1:n_neighbors + 1]
            densities[i] = 1.0 / (np.mean(nearest_distances) + 1e-8)
        return densities

    def _calculate_sample_correlation(self, original_coords: np.ndarray, filtered_coords: np.ndarray) -> Dict[str, Any]:
        """计算样本相关性"""
        try:
            # 这里简化处理，实际应用中可能需要样本对应关系
            # 计算整体分布的相关性
            if len(original_coords) == 0 or len(filtered_coords) == 0:
                return {'pearson_correlation': 0.0, 'spearman_correlation': 0.0}

            # 计算坐标分布的相关性
            original_flat = original_coords.flatten()
            filtered_flat = filtered_coords.flatten()

            # 对齐长度
            min_length = min(len(original_flat), len(filtered_flat))
            original_aligned = original_flat[:min_length]
            filtered_aligned = filtered_flat[:min_length]

            if len(original_aligned) < 2:
                return {'pearson_correlation': 0.0, 'spearman_correlation': 0.0}

            pearson_corr, _ = pearsonr(original_aligned, filtered_aligned)
            spearman_corr, _ = spearmanr(original_aligned, filtered_aligned)

            return {
                'pearson_correlation': float(pearson_corr if not np.isnan(pearson_corr) else 0.0),
                'spearman_correlation': float(spearman_corr if not np.isnan(spearman_corr) else 0.0)
            }

        except Exception as e:
            logger.error(f"Failed to calculate sample correlation: {str(e)}")
            return {'pearson_correlation': 0.0, 'spearman_correlation': 0.0}

    def _calculate_quality_metrics(self, original_coords: np.ndarray, filtered_coords: np.ndarray) -> Dict[str, Any]:
        """计算质量指标"""
        try:
            # 计算信息保留率
            information_retention = len(filtered_coords) / len(original_coords) if len(original_coords) > 0 else 0

            # 计算分布相似性
            distribution_similarity = self._calculate_distribution_similarity(original_coords, filtered_coords)

            # 计算结构保持性
            structure_preservation = self._calculate_structure_preservation(original_coords, filtered_coords)

            return {
                'information_retention_rate': float(information_retention),
                'distribution_similarity': float(distribution_similarity),
                'structure_preservation_score': float(structure_preservation),
                'overall_quality_score': float((information_retention + distribution_similarity + structure_preservation) / 3)
            }

        except Exception as e:
            logger.error(f"Failed to calculate quality metrics: {str(e)}")
            return {
                'information_retention_rate': 0.0,
                'distribution_similarity': 0.0,
                'structure_preservation_score': 0.0,
                'overall_quality_score': 0.0
            }

    def _calculate_distribution_similarity(self, original_coords: np.ndarray, filtered_coords: np.ndarray) -> float:
        """计算分布相似性"""
        try:
            # 使用Wasserstein距离来衡量分布相似性
            from scipy.stats import wasserstein_distance

            if len(original_coords) == 0 or len(filtered_coords) == 0:
                return 0.0

            # 计算每个维度的Wasserstein距离
            dist_x = wasserstein_distance(original_coords[:, 0], filtered_coords[:, 0])
            dist_y = wasserstein_distance(original_coords[:, 1], filtered_coords[:, 1])

            # 归一化相似性得分
            avg_distance = (dist_x + dist_y) / 2
            similarity = max(0, 1 - avg_distance / 10)  # 假设距离10对应相似性0

            return similarity

        except:
            # 如果无法计算，使用简单的均值和标准差比较
            return 0.5

    def _calculate_structure_preservation(self, original_coords: np.ndarray, filtered_coords: np.ndarray) -> float:
        """计算结构保持性"""
        try:
            if len(original_coords) < 3 or len(filtered_coords) < 3:
                return 0.5

            # 计算相对距离的保持性
            original_distances = cdist(original_coords, original_coords)
            filtered_distances = cdist(filtered_coords, filtered_coords)

            # 计算距离分布的相似性
            original_dist_flat = original_distances[np.triu_indices_from(original_distances, k=1)]
            filtered_dist_flat = filtered_distances[np.triu_indices_from(filtered_distances, k=1)]

            if len(original_dist_flat) == 0 or len(filtered_dist_flat) == 0:
                return 0.5

            # 计算相关性
            min_length = min(len(original_dist_flat), len(filtered_dist_flat))
            original_aligned = original_dist_flat[:min_length]
            filtered_aligned = filtered_dist_flat[:min_length]

            if len(original_aligned) < 2:
                return 0.5

            correlation, _ = pearsonr(original_aligned, filtered_aligned)
            return float(abs(correlation) if not np.isnan(correlation) else 0.5)

        except:
            return 0.5

    def _calculate_point_density(self, point: np.ndarray, coords: np.ndarray, n_neighbors: int = 10) -> float:
        """计算点的局部密度"""
        if len(coords) == 0:
            return 0.0

        distances = np.linalg.norm(coords - point, axis=1)
        nearest_distances = np.partition(distances, n_neighbors + 1)[1:n_neighbors + 1]
        return 1.0 / (np.mean(nearest_distances) + 1e-8)

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

    def _generate_comparison_id(self, pipeline_id_1: str, pipeline_id_2: str) -> str:
        """生成对比ID"""
        return f"cmp_{pipeline_id_1[:8]}_{pipeline_id_2[:8]}"

    async def _save_comparison_result(self, pipeline_id_1: str, pipeline_id_2: str,
                                    comparison_data: Dict[str, Any],
                                    statistics: Dict[str, Any]) -> None:
        """保存对比结果"""
        comparison_dir = self.results_dir / "comparisons"
        comparison_dir.mkdir(exist_ok=True)

        comparison_id = self._generate_comparison_id(pipeline_id_1, pipeline_id_2)
        comparison_file = comparison_dir / f"{comparison_id}.json"

        result_data = {
            'comparison_id': comparison_id,
            'pipeline_1_id': pipeline_id_1,
            'pipeline_2_id': pipeline_id_2,
            'comparison_data': comparison_data,
            'statistics': statistics,
            'created_at': datetime.now().isoformat()
        }

        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Comparison result saved: {comparison_file}")

    async def _save_comparison_data(self, comparison_id: str, data: Dict[str, Any]) -> None:
        """保存对比数据"""
        comparison_dir = self.results_dir / "comparisons"
        comparison_dir.mkdir(exist_ok=True)

        comparison_file = comparison_dir / f"{comparison_id}.json"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    async def _generate_comparison_visualization_config(self, comparison_data: Dict[str, Any],
                                                       statistics: Dict[str, Any],
                                                       config: Dict[str, Any]) -> Dict[str, Any]:
        """生成对比可视化配置"""
        return {
            'chart_title': config.get('chart_title', 'MOF数据对比可视化'),
            'background_config': {
                'opacity': config.get('background_opacity', 0.3),
                'color': config['color_scheme']['background_color'],
                'marker_size': config['marker_size']['background']
            },
            'foreground_config': {
                'opacity': config.get('foreground_opacity', 0.8),
                'color': config['color_scheme']['foreground_color'],
                'marker_size': config['marker_size']['foreground']
            },
            'layout': {
                'width': config.get('width', 1200),
                'height': config.get('height', 800),
                'show_legend': True
            }
        }

    def _create_background_trace(self, coordinates: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """创建背景轨迹"""
        return {
            'x': [coord['x'] for coord in coordinates],
            'y': [coord['y'] for coord in coordinates],
            'mode': 'markers',
            'type': 'scatter',
            'marker': {
                'size': config['marker_size']['background'],
                'color': config['color_scheme']['background_color'],
                'opacity': config['background_opacity']
            },
            'name': '原始数据',
            'showlegend': config.get('show_original', True)
        }

    def _create_foreground_trace(self, coordinates: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """创建前景轨迹"""
        return {
            'x': [coord['x'] for coord in coordinates],
            'y': [coord['y'] for coord in coordinates],
            'mode': 'markers',
            'type': 'scatter',
            'marker': {
                'size': config['marker_size']['foreground'],
                'color': config['color_scheme']['foreground_color'],
                'opacity': config['foreground_opacity']
            },
            'name': '筛选数据',
            'showlegend': config.get('show_filtered', True)
        }

    def _create_statistics_trace(self, statistics: Dict[str, Any]) -> Dict[str, Any]:
        """创建统计信息轨迹"""
        # 这里可以添加统计信息的可视化
        return {
            'type': 'scatter',
            'mode': 'markers',
            'x': [],
            'y': [],
            'name': '统计信息',
            'showlegend': False
        }

    def _generate_comparison_layout(self, comparison_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成对比布局"""
        return {
            'title': comparison_data['config'].get('chart_title', 'MOF数据对比可视化'),
            'xaxis': {'title': 't-SNE维度1'},
            'yaxis': {'title': 't-SNE维度2'},
            'width': comparison_data['config']['layout']['width'],
            'height': comparison_data['config']['layout']['height'],
            'showlegend': comparison_data['config']['layout']['show_legend']
        }

    def cleanup_old_comparisons(self, max_age_hours: int = 48) -> int:
        """清理旧的对比数据"""
        cleanup_count = 0
        current_time = datetime.now()
        comparison_dir = self.results_dir / "comparisons"

        if not comparison_dir.exists():
            return 0

        for comparison_file in comparison_dir.glob("*.json"):
            try:
                file_age = current_time - datetime.fromtimestamp(comparison_file.stat().st_mtime)
                if file_age.total_seconds() > max_age_hours * 3600:
                    comparison_file.unlink()
                    cleanup_count += 1
                    logger.info(f"Cleaned up old comparison: {comparison_file}")
            except Exception as e:
                logger.warning(f"Failed to cleanup comparison file {comparison_file}: {str(e)}")

        logger.info(f"Comparison cleanup completed. Removed {cleanup_count} old comparisons")
        return cleanup_count