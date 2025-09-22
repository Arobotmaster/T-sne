"""
算法基础类

遵循SDD Constitution的Library-First原则，
定义算法接口和基础功能。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import logging
from dataclasses import dataclass
from datetime import datetime


@dataclass
class AlgorithmResult:
    """算法结果数据类"""
    data: np.ndarray
    metadata: Dict[str, Any]
    processing_time_ms: int
    algorithm_version: str
    success: bool = True
    error_message: Optional[str] = None


class BaseAlgorithm(ABC):
    """算法基类"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化算法

        Args:
            config: 算法配置参数
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.version = "1.0.0"
        self.is_fitted = False
        self.fitted_data_shape = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'BaseAlgorithm':
        """
        训练算法

        Args:
            X: 训练数据
            y: 可选的目标变量

        Returns:
            self: 训练后的算法实例
        """
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        转换数据

        Args:
            X: 输入数据

        Returns:
            np.ndarray: 转换后的数据
        """
        pass

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> AlgorithmResult:
        """
        训练并转换数据

        Args:
            X: 输入数据
            y: 可选的目标变量

        Returns:
            AlgorithmResult: 算法结果
        """
        start_time = datetime.now()

        try:
            self.logger.info(f"开始执行 {self.__class__.__name__}")
            self.logger.info(f"输入数据形状: {X.shape}")

            # 训练算法
            self.fit(X, y)

            # 转换数据
            transformed_data = self.transform(X)

            # 计算处理时间
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)

            # 生成元数据
            metadata = self._generate_metadata(X, transformed_data)

            self.logger.info(f"算法执行完成")
            self.logger.info(f"输出数据形状: {transformed_data.shape}")
            self.logger.info(f"处理时间: {processing_time}ms")

            return AlgorithmResult(
                data=transformed_data,
                metadata=metadata,
                processing_time_ms=processing_time,
                algorithm_version=self.version,
                success=True
            )

        except Exception as e:
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            error_msg = f"算法执行失败: {str(e)}"
            self.logger.error(error_msg)

            return AlgorithmResult(
                data=np.array([]),
                metadata={},
                processing_time_ms=processing_time,
                algorithm_version=self.version,
                success=False,
                error_message=error_msg
            )

    @abstractmethod
    def _generate_metadata(self, X: np.ndarray, transformed_data: np.ndarray) -> Dict[str, Any]:
        """
        生成算法元数据

        Args:
            X: 原始数据
            transformed_data: 转换后的数据

        Returns:
            Dict[str, Any]: 元数据字典
        """
        pass

    def validate_input(self, X: np.ndarray) -> bool:
        """
        验证输入数据

        Args:
            X: 输入数据

        Returns:
            bool: 验证是否通过
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("输入数据必须是numpy数组")

        if X.size == 0:
            raise ValueError("输入数据不能为空")

        if not np.isfinite(X).all():
            raise ValueError("输入数据包含无限或NaN值")

        return True

    def get_config_summary(self) -> Dict[str, Any]:
        """
        获取配置摘要

        Returns:
            Dict[str, Any]: 配置摘要
        """
        return {
            "algorithm_name": self.__class__.__name__,
            "version": self.version,
            "config": self.config,
            "is_fitted": self.is_fitted
        }


class DimensionalityReductionAlgorithm(BaseAlgorithm):
    """降维算法基类"""

    @abstractmethod
    def get_explained_variance(self) -> Optional[np.ndarray]:
        """获取解释方差"""
        pass

    @abstractmethod
    def get_components(self) -> Optional[np.ndarray]:
        """获取主成分"""
        pass


class PreprocessingAlgorithm(BaseAlgorithm):
    """预处理算法基类"""

    @abstractmethod
    def get_feature_names_out(self) -> List[str]:
        """获取输出特征名称"""
        pass


class ClusteringAlgorithm(BaseAlgorithm):
    """聚类算法基类"""

    @abstractmethod
    def get_labels(self) -> Optional[np.ndarray]:
        """获取聚类标签"""
        pass

    @abstractmethod
    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """获取聚类中心"""
        pass


def validate_algorithm_config(config: Dict[str, Any], required_params: List[str]) -> None:
    """
    验证算法配置

    Args:
        config: 配置字典
        required_params: 必需参数列表
    """
    missing_params = [param for param in required_params if param not in config]
    if missing_params:
        raise ValueError(f"缺少必需的算法参数: {missing_params}")


def log_algorithm_performance(logger: logging.Logger, algorithm_name: str,
                              input_shape: tuple, output_shape: tuple,
                              processing_time_ms: int) -> None:
    """
    记录算法性能信息

    Args:
        logger: 日志记录器
        algorithm_name: 算法名称
        input_shape: 输入数据形状
        output_shape: 输出数据形状
        processing_time_ms: 处理时间(毫秒)
    """
    logger.info(f"算法性能统计 - {algorithm_name}:")
    logger.info(f"  输入形状: {input_shape}")
    logger.info(f"  输出形状: {output_shape}")
    logger.info(f"  处理时间: {processing_time_ms}ms")

    if input_shape[1] > 0:
        compression_ratio = output_shape[1] / input_shape[1]
        logger.info(f"  压缩比: {compression_ratio:.3f}")

    if processing_time_ms > 0:
        samples_per_second = (input_shape[0] * 1000) / processing_time_ms
        logger.info(f"  处理速度: {samples_per_second:.1f} 样本/秒")