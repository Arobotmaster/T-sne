"""
导出服务

遵循SDD Constitution原则：
- Library-First: 独立可测试的服务组件
- CLI Interface: 支持命令行调用
- Scientific Observability: 详细的处理日志记录
"""

import os
import json
import uuid
import logging
import tempfile
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import kaleido
from PIL import Image, ImageDraw, ImageFont

from ..config.logging_config import get_logger

logger = get_logger(__name__)


class ExportService:
    """导出服务

    负责将t-SNE可视化导出为多种格式的图像文件
    """

    def __init__(self, export_dir: str = "exports"):
        """
        初始化导出服务

        Args:
            export_dir: 导出文件目录
        """
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

        # 支持的导出格式
        self.supported_formats = {
            'png': {'mime_type': 'image/png', 'extension': '.png'},
            'svg': {'mime_type': 'image/svg+xml', 'extension': '.svg'},
            'pdf': {'mime_type': 'application/pdf', 'extension': '.pdf'}
        }

        # 文件清理配置
        self.file_expiry_hours = 24

        logger.info(f"ExportService initialized. Export dir: {export_dir}")

    async def export_visualization(self, pipeline_id: str, export_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        导出可视化图像

        Args:
            pipeline_id: 处理流水线ID
            export_config: 导出配置

        Returns:
            Dict[str, Any]: 导出结果，包含文件信息和下载链接

        Raises:
            ValueError: 配置验证失败或数据不存在
        """
        logger.info(f"Starting export for pipeline: {pipeline_id}")

        try:
            # 验证导出配置
            validated_config = self._validate_export_config(export_config)

            # 获取可视化数据
            visualization_data = await self._get_visualization_data(pipeline_id)
            if not visualization_data:
                raise ValueError(f"无法获取可视化数据: {pipeline_id}")

            # 创建Plotly图表
            fig = await self._create_plotly_figure(visualization_data, validated_config)

            # 生成文件名
            filename = self._generate_filename(pipeline_id, validated_config)

            # 导出图像
            file_info = await self._export_image(fig, filename, validated_config)

            # 创建下载记录
            download_record = self._create_download_record(file_info, validated_config)

            logger.info(f"Export completed successfully: {file_info['file_path']}")
            return download_record

        except Exception as e:
            logger.error(f"Export failed for pipeline {pipeline_id}: {str(e)}")
            raise ValueError(f"导出失败: {str(e)}")

    async def export_comparison(self, pipeline_id_1: str, pipeline_id_2: str,
                               export_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        导出对比可视化图像

        Args:
            pipeline_id_1: 第一个处理流水线ID
            pipeline_id_2: 第二个处理流水线ID
            export_config: 导出配置

        Returns:
            Dict[str, Any]: 导出结果
        """
        logger.info(f"Starting comparison export for pipelines: {pipeline_id_1} vs {pipeline_id_2}")

        try:
            # 验证导出配置
            validated_config = self._validate_export_config(export_config)

            # 获取对比可视化数据
            comparison_data = await self._get_comparison_data(pipeline_id_1, pipeline_id_2)
            if not comparison_data:
                raise ValueError("无法获取对比可视化数据")

            # 创建对比图表
            fig = await self._create_comparison_figure(comparison_data, validated_config)

            # 生成文件名
            filename = self._generate_comparison_filename(pipeline_id_1, pipeline_id_2, validated_config)

            # 导出图像
            file_info = await self._export_image(fig, filename, validated_config)

            # 创建下载记录
            download_record = self._create_download_record(file_info, validated_config)

            logger.info(f"Comparison export completed: {file_info['file_path']}")
            return download_record

        except Exception as e:
            logger.error(f"Comparison export failed: {str(e)}")
            raise ValueError(f"对比导出失败: {str(e)}")

    async def export_data_summary(self, pipeline_id: str, format_type: str = 'csv') -> Dict[str, Any]:
        """
        导出数据摘要信息

        Args:
            pipeline_id: 处理流水线ID
            format_type: 导出格式 (csv, json, excel)

        Returns:
            Dict[str, Any]: 导出结果
        """
        logger.info(f"Exporting data summary for pipeline: {pipeline_id}, format: {format_type}")

        try:
            # 获取处理元数据
            metadata = await self._get_processing_metadata(pipeline_id)
            if not metadata:
                raise ValueError(f"无法获取处理元数据: {pipeline_id}")

            # 获取t-SNE坐标
            coordinates_df = await self._get_coordinates_dataframe(pipeline_id)

            # 创建摘要数据
            summary_data = self._create_data_summary(metadata, coordinates_df)

            # 根据格式导出
            if format_type.lower() == 'csv':
                file_info = await self._export_csv_summary(summary_data, pipeline_id)
            elif format_type.lower() == 'json':
                file_info = await self._export_json_summary(summary_data, pipeline_id)
            elif format_type.lower() == 'excel':
                file_info = await self._export_excel_summary(summary_data, pipeline_id)
            else:
                raise ValueError(f"不支持的导出格式: {format_type}")

            # 创建下载记录
            download_record = self._create_download_record(file_info, {'format': format_type})

            logger.info(f"Data summary exported: {file_info['file_path']}")
            return download_record

        except Exception as e:
            logger.error(f"Data summary export failed: {str(e)}")
            raise ValueError(f"数据摘要导出失败: {str(e)}")

    async def _get_visualization_data(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """获取可视化数据"""
        try:
            # 这里应该调用VisualizationService获取数据
            # 为了简化，我们直接从文件系统读取
            results_dir = Path("results") / pipeline_id
            coordinates_file = results_dir / "tsne_coordinates.csv"

            if not coordinates_file.exists():
                return None

            coordinates_df = pd.read_csv(coordinates_file)

            # 构造可视化数据结构
            visualization_data = {
                'coordinates': coordinates_df.to_dict('records'),
                'config': {
                    'chart_title': 'MOF数据t-SNE可视化',
                    'x_axis_label': 't-SNE维度1',
                    'y_axis_label': 't-SNE维度2'
                },
                'statistics': {
                    'total_samples': len(coordinates_df),
                    'x_range': {'min': float(coordinates_df['x'].min()), 'max': float(coordinates_df['x'].max())},
                    'y_range': {'min': float(coordinates_df['y'].min()), 'max': float(coordinates_df['y'].max())}
                }
            }

            return visualization_data

        except Exception as e:
            logger.error(f"Failed to get visualization data: {str(e)}")
            return None

    async def _get_comparison_data(self, pipeline_id_1: str, pipeline_id_2: str) -> Optional[Dict[str, Any]]:
        """获取对比可视化数据"""
        try:
            # 获取两个流水线的坐标数据
            coords_1 = await self._get_coordinates_dataframe(pipeline_id_1)
            coords_2 = await self._get_coordinates_dataframe(pipeline_id_2)

            if coords_1 is None or coords_2 is None:
                return None

            # 构造对比数据
            comparison_data = {
                'pipeline_1': {
                    'coordinates': coords_1.to_dict('records'),
                    'name': f'Pipeline {pipeline_id_1[:8]}'
                },
                'pipeline_2': {
                    'coordinates': coords_2.to_dict('records'),
                    'name': f'Pipeline {pipeline_id_2[:8]}'
                }
            }

            return comparison_data

        except Exception as e:
            logger.error(f"Failed to get comparison data: {str(e)}")
            return None

    async def _create_plotly_figure(self, visualization_data: Dict[str, Any], config: Dict[str, Any]) -> go.Figure:
        """创建Plotly图表"""
        try:
            coordinates = visualization_data['coordinates']
            viz_config = visualization_data['config']

            # 提取坐标数据
            x_coords = [coord['x'] for coord in coordinates]
            y_coords = [coord['y'] for coord in coordinates]

            # 创建散点图
            fig = go.Figure()

            # 添加主数据点
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers',
                marker=dict(
                    size=config.get('marker_size', 8),
                    color=config.get('marker_color', '#1f77b4'),
                    opacity=config.get('opacity', 0.8),
                    line=dict(width=1, color='white')
                ),
                name='MOF Samples',
                text=[coord.get('sample_id', '') for coord in coordinates],
                hovertemplate='<b>%{text}</b><br>' +
                           'X: %{x:.3f}<br>' +
                           'Y: %{y:.3f}<br>' +
                           '<extra></extra>'
            ))

            # 设置图表布局
            fig.update_layout(
                title=viz_config.get('chart_title', 'MOF数据t-SNE可视化'),
                xaxis_title=viz_config.get('x_axis_label', 't-SNE维度1'),
                yaxis_title=viz_config.get('y_axis_label', 't-SNE维度2'),
                width=config.get('width', 1200),
                height=config.get('height', 800),
                showlegend=config.get('show_legend', True),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(size=12)
            )

            # 设置坐标轴样式
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

            return fig

        except Exception as e:
            logger.error(f"Failed to create Plotly figure: {str(e)}")
            raise

    async def _create_comparison_figure(self, comparison_data: Dict[str, Any], config: Dict[str, Any]) -> go.Figure:
        """创建对比图表"""
        try:
            # 创建子图
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=[
                    comparison_data['pipeline_1']['name'],
                    comparison_data['pipeline_2']['name']
                ],
                specs=[[{'type': 'scatter'}, {'type': 'scatter'}]]
            )

            # 添加第一个数据集
            coords_1 = comparison_data['pipeline_1']['coordinates']
            fig.add_trace(
                go.Scatter(
                    x=[coord['x'] for coord in coords_1],
                    y=[coord['y'] for coord in coords_1],
                    mode='markers',
                    marker=dict(size=6, color='rgba(31, 119, 180, 0.6)'),
                    name='Dataset 1',
                    showlegend=False
                ),
                row=1, col=1
            )

            # 添加第二个数据集
            coords_2 = comparison_data['pipeline_2']['coordinates']
            fig.add_trace(
                go.Scatter(
                    x=[coord['x'] for coord in coords_2],
                    y=[coord['y'] for coord in coords_2],
                    mode='markers',
                    marker=dict(size=6, color='rgba(255, 127, 14, 0.8)'),
                    name='Dataset 2',
                    showlegend=False
                ),
                row=1, col=2
            )

            # 设置布局
            fig.update_layout(
                title='t-SNE对比可视化',
                width=config.get('width', 1600),
                height=config.get('height', 800),
                showlegend=False
            )

            return fig

        except Exception as e:
            logger.error(f"Failed to create comparison figure: {str(e)}")
            raise

    async def _export_image(self, fig: go.Figure, filename: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """导出图像文件"""
        try:
            format_type = config['format']
            file_extension = self.supported_formats[format_type]['extension']

            # 生成文件路径
            file_path = self.export_dir / f"{filename}{file_extension}"

            # 导出图像
            if format_type == 'png':
                fig.write_image(str(file_path), format='png', width=config['width'], height=config['height'])
            elif format_type == 'svg':
                fig.write_image(str(file_path), format='svg')
            elif format_type == 'pdf':
                fig.write_image(str(file_path), format='pdf', width=config['width'], height=config['height'])

            # 获取文件大小
            file_size = file_path.stat().st_size

            file_info = {
                'file_id': str(uuid.uuid4()),
                'filename': f"{filename}{file_extension}",
                'file_path': str(file_path),
                'file_size_bytes': file_size,
                'format': format_type,
                'created_at': datetime.now().isoformat()
            }

            return file_info

        except Exception as e:
            logger.error(f"Failed to export image: {str(e)}")
            raise

    def _validate_export_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证导出配置"""
        validated_config = {
            'format': config.get('format', 'png'),
            'width': config.get('width', 1200),
            'height': config.get('height', 800),
            'dpi': config.get('dpi', 300),
            'background_color': config.get('background_color', 'white'),
            'filename': config.get('filename', ''),
            'marker_size': config.get('marker_size', 8),
            'marker_color': config.get('marker_color', '#1f77b4'),
            'opacity': config.get('opacity', 0.8)
        }

        # 验证格式
        if validated_config['format'] not in self.supported_formats:
            validated_config['format'] = 'png'

        # 验证尺寸
        validated_config['width'] = max(400, min(4000, validated_config['width']))
        validated_config['height'] = max(300, min(3000, validated_config['height']))

        # 验证DPI
        validated_config['dpi'] = max(72, min(600, validated_config['dpi']))

        return validated_config

    def _generate_filename(self, pipeline_id: str, config: Dict[str, Any]) -> str:
        """生成文件名"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = config.get('filename', f"mof_tsne_{pipeline_id[:8]}_{timestamp}")
        return base_name

    def _generate_comparison_filename(self, pipeline_id_1: str, pipeline_id_2: str, config: Dict[str, Any]) -> str:
        """生成对比文件名"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = config.get('filename', f"mof_tsne_comparison_{pipeline_id_1[:8]}_vs_{pipeline_id_2[:8]}_{timestamp}")
        return base_name

    def _create_download_record(self, file_info: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """创建下载记录"""
        expiry_time = datetime.now() + timedelta(hours=self.file_expiry_hours)

        download_record = {
            'success': True,
            'data': {
                'file_id': file_info['file_id'],
                'filename': file_info['filename'],
                'file_size_bytes': file_info['file_size_bytes'],
                'download_url': f"/api/downloads/{file_info['file_id']}",
                'format': config.get('format', 'png'),
                'expires_at': expiry_time.isoformat()
            },
            'timestamp': datetime.now().isoformat()
        }

        return download_record

    async def _get_coordinates_dataframe(self, pipeline_id: str) -> Optional[pd.DataFrame]:
        """获取坐标数据框"""
        try:
            coordinates_file = Path("results") / pipeline_id / "tsne_coordinates.csv"
            if not coordinates_file.exists():
                return None

            return pd.read_csv(coordinates_file)
        except Exception as e:
            logger.error(f"Failed to load coordinates: {str(e)}")
            return None

    async def _get_processing_metadata(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """获取处理元数据"""
        try:
            metadata_file = Path("results") / pipeline_id / "processing_metadata.json"
            if not metadata_file.exists():
                return None

            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load processing metadata: {str(e)}")
            return None

    def _create_data_summary(self, metadata: Dict[str, Any], coordinates_df: pd.DataFrame) -> Dict[str, Any]:
        """创建数据摘要"""
        summary = {
            'pipeline_info': {
                'pipeline_id': metadata.get('pipeline_id', ''),
                'timestamp': metadata.get('timestamp', ''),
                'total_samples': metadata.get('total_samples', 0)
            },
            'processing_summary': {
                'preprocessing_time_ms': metadata.get('preprocessing', {}).get('processing_time_ms', 0),
                'pca_time_ms': metadata.get('pca', {}).get('processing_time_ms', 0),
                'tsne_time_ms': metadata.get('tsne', {}).get('processing_time_ms', 0),
                'total_time_ms': sum([
                    metadata.get('preprocessing', {}).get('processing_time_ms', 0),
                    metadata.get('pca', {}).get('processing_time_ms', 0),
                    metadata.get('tsne', {}).get('processing_time_ms', 0)
                ])
            },
            'algorithm_configs': {
                'preprocessing': metadata.get('preprocessing', {}).get('config', {}),
                'pca': metadata.get('pca', {}).get('config', {}),
                'tsne': metadata.get('tsne', {}).get('config', {})
            }
        }

        # 添加坐标统计信息
        if coordinates_df is not None:
            summary['coordinate_statistics'] = {
                'x_range': {'min': float(coordinates_df['x'].min()), 'max': float(coordinates_df['x'].max())},
                'y_range': {'min': float(coordinates_df['y'].min()), 'max': float(coordinates_df['y'].max())},
                'center_point': {
                    'x': float(coordinates_df['x'].mean()),
                    'y': float(coordinates_df['y'].mean())
                }
            }

        return summary

    async def _export_csv_summary(self, summary_data: Dict[str, Any], pipeline_id: str) -> Dict[str, Any]:
        """导出CSV格式的数据摘要"""
        filename = f"data_summary_{pipeline_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file_path = self.export_dir / filename

        # 将摘要数据转换为CSV格式
        # 这里简化处理，实际应用中可能需要更复杂的数据转换
        summary_df = pd.DataFrame([summary_data])
        summary_df.to_csv(file_path, index=False)

        file_size = file_path.stat().st_size

        return {
            'file_id': str(uuid.uuid4()),
            'filename': filename,
            'file_path': str(file_path),
            'file_size_bytes': file_size,
            'format': 'csv'
        }

    async def _export_json_summary(self, summary_data: Dict[str, Any], pipeline_id: str) -> Dict[str, Any]:
        """导出JSON格式的数据摘要"""
        filename = f"data_summary_{pipeline_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        file_path = self.export_dir / filename

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)

        file_size = file_path.stat().st_size

        return {
            'file_id': str(uuid.uuid4()),
            'filename': filename,
            'file_path': str(file_path),
            'file_size_bytes': file_size,
            'format': 'json'
        }

    async def _export_excel_summary(self, summary_data: Dict[str, Any], pipeline_id: str) -> Dict[str, Any]:
        """导出Excel格式的数据摘要"""
        filename = f"data_summary_{pipeline_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        file_path = self.export_dir / filename

        # 使用pandas导出Excel
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # 写入摘要信息
            summary_df = pd.DataFrame([summary_data])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

        file_size = file_path.stat().st_size

        return {
            'file_id': str(uuid.uuid4()),
            'filename': filename,
            'file_path': str(file_path),
            'file_size_bytes': file_size,
            'format': 'excel'
        }

    def get_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """获取文件信息"""
        # 遍历导出目录查找文件
        for file_path in self.export_dir.glob("*"):
            if file_path.is_file():
                # 这里可以根据文件ID匹配找到对应的文件
                # 简化处理，实际应用中需要维护文件索引
                pass

        return None

    def cleanup_expired_files(self) -> int:
        """清理过期文件"""
        cleanup_count = 0
        current_time = datetime.now()

        for file_path in self.export_dir.glob("*"):
            if file_path.is_file():
                try:
                    file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_age.total_seconds() > self.file_expiry_hours * 3600:
                        file_path.unlink()
                        cleanup_count += 1
                        logger.info(f"Cleaned up expired file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup file {file_path}: {str(e)}")

        logger.info(f"Cleanup completed. Removed {cleanup_count} expired files")
        return cleanup_count