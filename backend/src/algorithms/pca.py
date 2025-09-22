"""
PCA降维算法实现

遵循SDD Constitution的Library-First原则，
实现独立可测试的PCA算法库。
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging

from .base import DimensionalityReductionAlgorithm, AlgorithmResult, validate_algorithm_config, log_algorithm_performance


class PCAProcessor(DimensionalityReductionAlgorithm):
    """PCA降维处理器

    遵循SDD Constitution原则：
    - Library-First: 独立可重用的算法库
    - CLI Interface: 支持命令行调用
    - Scientific Observability: 详细的科学计算日志
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化PCA处理器

        Args:
            config: PCA配置参数
                - n_components: 降维后的维度或方差保留率
                - whiten: 是否白化
                - svd_solver: SVD求解器
                - random_state: 随机种子
        """
        super().__init__(config)
        self.pca = None
        self.scaler = StandardScaler()
        self.explained_variance_ratio_ = None
        self.cumulative_variance_ratio_ = None

        # 验证配置
        validate_algorithm_config(config, ['n_components'])

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'PCAProcessor':
        """
        训练PCA模型

        Args:
            X: 输入特征矩阵 (n_samples, n_features)
            y: 未使用

        Returns:
            self: 训练后的PCA处理器
        """
        self.validate_input(X)

        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)

        # 创建PCA模型（健壮化 n_components）
        requested_nc = self.config.get('n_components', None)
        eff_nc = requested_nc
        n_samples, n_features = X.shape
        max_nc = min(n_samples, n_features)
        # clamp when int and beyond bound
        if isinstance(eff_nc, int):
            if eff_nc <= 0:
                eff_nc = max_nc
            elif eff_nc > max_nc:
                self.logger.warning(
                    f"n_components={eff_nc} 超出上限，自动调整为 {max_nc} (min(n_samples, n_features))"
                )
                eff_nc = max_nc
        # float in (0,1] is allowed by sklearn as variance retention; leave as-is

        pca_params = {
            'n_components': eff_nc,
            'whiten': self.config.get('whiten', False),
            'svd_solver': self.config.get('svd_solver', 'auto')
        }

        if 'random_state' in self.config:
            pca_params['random_state'] = self.config['random_state']

        self.pca = PCA(**pca_params)

        # 训练PCA模型
        self.pca.fit(X_scaled)

        # 保存方差信息
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

        self.is_fitted = True
        self.fitted_data_shape = X.shape

        self.logger.info(f"PCA训练完成")
        self.logger.info(f"主成分数量: {self.pca.n_components_}")
        self.logger.info(f"累积解释方差: {self.cumulative_variance_ratio_[-1]:.3f}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        应用PCA降维

        Args:
            X: 输入特征矩阵

        Returns:
            np.ndarray: 降维后的数据
        """
        if not self.is_fitted:
            raise ValueError("PCA模型未训练，请先调用fit方法")

        self.validate_input(X)

        # 数据标准化
        X_scaled = self.scaler.transform(X)

        # 应用PCA降维
        X_transformed = self.pca.transform(X_scaled)

        return X_transformed

    def _generate_metadata(self, X: np.ndarray, transformed_data: np.ndarray) -> Dict[str, Any]:
        """
        生成PCA元数据

        Args:
            X: 原始数据
            transformed_data: 降维后的数据

        Returns:
            Dict[str, Any]: 元数据字典
        """
        metadata = {
            'algorithm': 'PCA',
            'version': self.version,
            'input_shape': X.shape,
            'output_shape': transformed_data.shape,
            'n_components': self.pca.n_components_,
            'explained_variance_ratio': self.explained_variance_ratio_.tolist(),
            'cumulative_variance_ratio': self.cumulative_variance_ratio_.tolist(),
            'total_variance_retained': float(self.cumulative_variance_ratio_[-1]),
            'config': self.config
        }

        # 添加主成分信息
        if hasattr(self.pca, 'components_'):
            metadata['components_shape'] = self.pca.components_.shape

        # 添加均值信息
        if hasattr(self.scaler, 'mean_'):
            metadata['feature_means'] = self.scaler.mean_.tolist()
            metadata['feature_stds'] = self.scaler.scale_.tolist()

        return metadata

    def get_explained_variance(self) -> Optional[np.ndarray]:
        """获取解释方差"""
        return self.explained_variance_ratio_

    def get_components(self) -> Optional[np.ndarray]:
        """获取主成分"""
        if self.pca is not None:
            return self.pca.components_
        return None

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """获取特征重要性（绝对值）"""
        if self.pca is not None:
            return np.abs(self.pca.components_).mean(axis=0)
        return None

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        反向变换到原始空间

        Args:
            X_transformed: 降维后的数据

        Returns:
            np.ndarray: 重构到原始空间的数据
        """
        if not self.is_fitted:
            raise ValueError("PCA模型未训练")

        # 反向PCA变换
        X_reconstructed = self.pca.inverse_transform(X_transformed)

        # 反向标准化
        X_original = self.scaler.inverse_transform(X_reconstructed)

        return X_original

    def get_optimal_components(self, variance_threshold: float = 0.95) -> int:
        """
        获取达到指定方差阈值所需的主成分数量

        Args:
            variance_threshold: 方差阈值

        Returns:
            int: 最优主成分数量
        """
        if self.cumulative_variance_ratio_ is None:
            raise ValueError("PCA模型未训练")

        optimal_idx = np.argmax(self.cumulative_variance_ratio_ >= variance_threshold)
        return optimal_idx + 1

    @staticmethod
    def recommend_parameters(X: np.ndarray) -> Dict[str, Any]:
        """
        根据数据特征推荐PCA参数

        Args:
            X: 输入数据

        Returns:
            Dict[str, Any]: 推荐的参数
        """
        n_samples, n_features = X.shape

        # 基于数据维度推荐组件数量
        if n_features <= 50:
            n_components = min(n_features, 20)
        elif n_features <= 100:
            n_components = min(n_features, 50)
        else:
            n_components = min(n_features, 100)

        return {
            'n_components': n_components,
            'whiten': False,
            'svd_solver': 'auto',
            'random_state': 42
        }


def create_pca_pipeline(config: Dict[str, Any]) -> PCAProcessor:
    """
    创建PCA处理管道

    Args:
        config: PCA配置

    Returns:
        PCAProcessor: 配置好的PCA处理器
    """
    return PCAProcessor(config)


# CLI支持函数
def run_pca_cli(X: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    运行PCA算法（CLI接口）

    Args:
        X: 输入数据
        config: PCA配置

    Returns:
        Dict[str, Any]: 处理结果
    """
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # 创建PCA处理器
        pca = create_pca_pipeline(config)

        # 执行PCA
        result = pca.fit_transform(X)

        if result.success:
            logger.info("PCA降维成功完成")
            log_algorithm_performance(
                logger, "PCA",
                X.shape, result.data.shape,
                result.processing_time_ms
            )

            return {
                'success': True,
                'data': result.data.tolist(),
                'metadata': result.metadata
            }
        else:
            logger.error(f"PCA降维失败: {result.error_message}")
            return {
                'success': False,
                'error': result.error_message
            }

    except Exception as e:
        error_msg = f"PCA处理异常: {str(e)}"
        logger.error(error_msg)
        return {
            'success': False,
            'error': error_msg
        }
