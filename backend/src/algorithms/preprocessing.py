"""
数据预处理算法实现

遵循SDD Constitution的Library-First原则，
实现独立可测试的数据预处理算法库。
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import logging
from scipy import stats

from .base import PreprocessingAlgorithm, AlgorithmResult, validate_algorithm_config, log_algorithm_performance


class DataPreprocessor(PreprocessingAlgorithm):
    """数据预处理器

    遵循SDD Constitution原则：
    - Library-First: 独立可重用的算法库
    - CLI Interface: 支持命令行调用
    - Scientific Observability: 详细的科学计算日志
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据预处理器

        Args:
            config: 预处理配置参数
                - missing_value_strategy: 缺失值处理策略
                - scaling_method: 标准化方法
                - outlier_detection: 是否检测异常值
                - outlier_threshold: 异常值阈值
                - feature_selection: 是否特征选择
        """
        super().__init__(config)
        self.imputer = None
        self.scaler = None
        self.outlier_mask_ = None
        self.feature_names_ = None
        self.n_features_ = None
        self.n_samples_ = None
        self.missing_stats_ = None
        self.outlier_stats_ = None

        # 验证配置
        validate_algorithm_config(config, ['missing_value_strategy'])

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'DataPreprocessor':
        """
        训练数据预处理器

        Args:
            X: 输入特征矩阵 (n_samples, n_features)
            y: 未使用

        Returns:
            self: 训练后的数据预处理器
        """
        self.validate_input(X)

        self.n_samples_, self.n_features_ = X.shape
        self.logger.info(f"开始数据预处理训练")
        self.logger.info(f"数据形状: {X.shape}")

        # 分析缺失值
        self._analyze_missing_values(X)

        # 处理缺失值
        self._fit_imputer(X)

        # 检测异常值
        if self.config.get('outlier_detection', True):
            self._detect_outliers(X)

        # 训练标准化器
        self._fit_scaler(X)

        self.is_fitted = True
        self.fitted_data_shape = X.shape

        self.logger.info("数据预处理训练完成")
        self.logger.info(f"处理后的特征数量: {self.n_features_}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        应用数据预处理

        Args:
            X: 输入特征矩阵

        Returns:
            np.ndarray: 预处理后的数据
        """
        if not self.is_fitted:
            raise ValueError("数据预处理器未训练，请先调用fit方法")

        self.validate_input(X)

        X_processed = X.copy()

        # 处理缺失值
        if self.imputer is not None:
            X_processed = self.imputer.transform(X_processed)
            self.logger.info("缺失值处理完成")

        # 数据标准化
        if self.scaler is not None:
            X_processed = self.scaler.transform(X_processed)
            self.logger.info("数据标准化完成")

        # 处理异常值
        if self.outlier_mask_ is not None:
            # 对于异常值，使用边界值替换
            X_processed = self._handle_outliers(X_processed)
            self.logger.info("异常值处理完成")

        return X_processed

    def _analyze_missing_values(self, X: np.ndarray) -> None:
        """分析缺失值统计信息"""
        missing_per_feature = np.isnan(X).sum(axis=0)
        missing_ratio = missing_per_feature / len(X)

        self.missing_stats_ = {
            'total_missing': int(missing_per_feature.sum()),
            'missing_per_feature': missing_per_feature.tolist(),
            'missing_ratio': missing_ratio.tolist(),
            'features_with_missing': int(np.sum(missing_ratio > 0)),
            'max_missing_ratio': float(np.max(missing_ratio))
        }

        self.logger.info(f"缺失值统计: 总计{self.missing_stats_['total_missing']}个")
        self.logger.info(f"有缺失值的特征: {self.missing_stats_['features_with_missing']}个")
        self.logger.info(f"最大缺失率: {self.missing_stats_['max_missing_ratio']:.3f}")

        # 检查缺失率是否过高
        if self.missing_stats_['max_missing_ratio'] > 0.5:
            self.logger.warning(f"存在特征缺失率过高(>{self.missing_stats_['max_missing_ratio']:.2%})")

    def _fit_imputer(self, X: np.ndarray) -> None:
        """训练缺失值填充器"""
        strategy = self.config.get('missing_value_strategy', 'mean')

        if np.isnan(X).any():  # 只有存在缺失值时才需要填充
            if strategy == 'drop':
                # 如果策略是drop，则不需要imputer
                self.imputer = None
            elif strategy in ['mean', 'median', 'mode']:
                self.imputer = SimpleImputer(strategy=strategy)
                self.imputer.fit(X)
            elif strategy == 'knn':
                self.imputer = KNNImputer(n_neighbors=5)
                self.imputer.fit(X)
            elif strategy == 'iterative':
                self.imputer = IterativeImputer(max_iter=10, random_state=42)
                self.imputer.fit(X)
            else:
                raise ValueError(f"不支持的缺失值处理策略: {strategy}")

            self.logger.info(f"缺失值填充器训练完成，策略: {strategy}")
        else:
            self.imputer = None
            self.logger.info("数据无缺失值，跳过缺失值处理")

    def _detect_outliers(self, X: np.ndarray) -> None:
        """检测异常值"""
        threshold = self.config.get('outlier_threshold', 3.0)
        method = self.config.get('outlier_method', 'zscore')

        outliers_per_feature = []
        total_outliers = 0

        for i in range(X.shape[1]):
            feature_data = X[:, i]
            non_nan_data = feature_data[~np.isnan(feature_data)]

            if len(non_nan_data) == 0:
                outliers_per_feature.append(0)
                continue

            if method == 'zscore':
                z_scores = np.abs(stats.zscore(non_nan_data))
                outliers = np.sum(z_scores > threshold)
            elif method == 'iqr':
                Q1 = np.percentile(non_nan_data, 25)
                Q3 = np.percentile(non_nan_data, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = np.sum((non_nan_data < lower_bound) | (non_nan_data > upper_bound))
            else:
                raise ValueError(f"不支持的异常值检测方法: {method}")

            outliers_per_feature.append(outliers)
            total_outliers += outliers

        self.outlier_stats_ = {
            'total_outliers': total_outliers,
            'outliers_per_feature': outliers_per_feature,
            'outlier_ratio': total_outliers / (X.shape[0] * X.shape[1]),
            'method': method,
            'threshold': threshold
        }

        # 创建异常值掩码
        self.outlier_mask_ = np.zeros_like(X, dtype=bool)
        for i in range(X.shape[1]):
            feature_data = X[:, i]
            non_nan_mask = ~np.isnan(feature_data)

            if method == 'zscore':
                z_scores = np.abs(stats.zscore(feature_data[non_nan_mask]))
                self.outlier_mask_[non_nan_mask, i] = z_scores > threshold
            elif method == 'iqr':
                Q1 = np.percentile(feature_data[non_nan_mask], 25)
                Q3 = np.percentile(feature_data[non_nan_mask], 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.outlier_mask_[non_nan_mask, i] = (
                    (feature_data[non_nan_mask] < lower_bound) |
                    (feature_data[non_nan_mask] > upper_bound)
                )

        self.logger.info(f"异常值检测完成，方法: {method}")
        self.logger.info(f"检测到异常值: {total_outliers}个 ({self.outlier_stats_['outlier_ratio']:.2%})")

    def _handle_outliers(self, X: np.ndarray) -> np.ndarray:
        """处理异常值"""
        if self.outlier_mask_ is None:
            return X

        X_handled = X.copy()
        method = self.config.get('outlier_method', 'zscore')

        for i in range(X.shape[1]):
            feature_data = X[:, i]
            outlier_indices = np.where(self.outlier_mask_[:, i])[0]

            if len(outlier_indices) > 0:
                non_nan_data = feature_data[~np.isnan(feature_data)]
                non_outlier_data = feature_data[~self.outlier_mask_[:, i] & ~np.isnan(feature_data)]

                if len(non_outlier_data) > 0:
                    if method == 'zscore':
                        # 使用中位数替换异常值
                        median_val = np.median(non_outlier_data)
                        X_handled[outlier_indices, i] = median_val
                    elif method == 'iqr':
                        # 使用边界值替换异常值
                        Q1 = np.percentile(non_outlier_data, 25)
                        Q3 = np.percentile(non_outlier_data, 75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR

                        # 替换为边界值
                        X_handled[outlier_indices, i] = np.where(
                            feature_data[outlier_indices] < lower_bound,
                            lower_bound,
                            upper_bound
                        )

        return X_handled

    def _fit_scaler(self, X: np.ndarray) -> None:
        """训练数据标准化器"""
        method = self.config.get('scaling_method', 'standard')

        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        elif method == 'none':
            self.scaler = None
        else:
            raise ValueError(f"不支持的标准化方法: {method}")

        if self.scaler is not None:
            self.scaler.fit(X)
            self.logger.info(f"标准化器训练完成，方法: {method}")
        else:
            self.logger.info("跳过数据标准化")

    def _generate_metadata(self, X: np.ndarray, transformed_data: np.ndarray) -> Dict[str, Any]:
        """生成预处理元数据"""
        metadata = {
            'algorithm': 'DataPreprocessor',
            'version': self.version,
            'input_shape': X.shape,
            'output_shape': transformed_data.shape,
            'config': self.config,
            'missing_stats': self.missing_stats_,
            'outlier_stats': self.outlier_stats_
        }

        # 添加数据质量指标
        metadata['data_quality'] = {
            'input_nan_ratio': float(np.isnan(X).mean()),
            'output_nan_ratio': float(np.isnan(transformed_data).mean()),
            'input_range': {
                'min': float(np.nanmin(X)),
                'max': float(np.nanmax(X))
            },
            'output_range': {
                'min': float(np.nanmin(transformed_data)),
                'max': float(np.nanmax(transformed_data))
            }
        }

        return metadata

    def get_feature_names_out(self) -> List[str]:
        """获取输出特征名称"""
        if self.feature_names_ is not None:
            return self.feature_names_
        return [f"feature_{i}" for i in range(self.n_features_)]

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """获取特征重要性（基于方差）"""
        if not self.is_fitted:
            return None

        # 使用方差作为特征重要性指标
        if hasattr(self.scaler, 'var_'):
            return self.scaler.var_
        elif self.n_features_ is not None:
            # 如果没有方差信息，返回等权重
            return np.ones(self.n_features_)
        return None

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """反向变换到原始空间"""
        if not self.is_fitted:
            raise ValueError("数据预处理器未训练")

        X_original = X_transformed.copy()

        # 反向标准化
        if self.scaler is not None:
            X_original = self.scaler.inverse_transform(X_original)

        # 注意：缺失值填充和异常值处理无法完全反向
        return X_original

    @staticmethod
    def recommend_parameters(X: np.ndarray) -> Dict[str, Any]:
        """根据数据特征推荐预处理参数"""
        n_samples, n_features = X.shape

        # 分析缺失值情况
        missing_ratio = np.isnan(X).mean()
        has_missing = missing_ratio > 0

        # 分析异常值情况
        outlier_ratio = 0
        for i in range(n_features):
            feature_data = X[:, i]
            non_nan_data = feature_data[~np.isnan(feature_data)]
            if len(non_nan_data) > 0:
                z_scores = np.abs(stats.zscore(non_nan_data))
                outlier_ratio += np.sum(z_scores > 3) / len(non_nan_data)
        outlier_ratio /= n_features

        # 推荐缺失值处理策略
        if missing_ratio < 0.05:
            missing_strategy = 'mean'
        elif missing_ratio < 0.2:
            missing_strategy = 'median'
        elif missing_ratio < 0.4:
            missing_strategy = 'knn'
        else:
            missing_strategy = 'iterative'

        # 推荐标准化方法
        if outlier_ratio > 0.1:  # 异常值较多时使用robust scaling
            scaling_method = 'robust'
        else:
            scaling_method = 'standard'

        return {
            'missing_value_strategy': missing_strategy,
            'scaling_method': scaling_method,
            'outlier_detection': outlier_ratio > 0.05,  # 异常值比例>5%时检测
            'outlier_method': 'zscore',
            'outlier_threshold': 3.0
        }

    def get_data_quality_report(self) -> Dict[str, Any]:
        """获取数据质量报告"""
        if not self.is_fitted:
            raise ValueError("数据预处理器未训练")

        report = {
            'samples_processed': self.n_samples_,
            'features_processed': self.n_features_,
            'missing_value_treatment': self.config.get('missing_value_strategy'),
            'scaling_method': self.config.get('scaling_method'),
            'outlier_detection': self.config.get('outlier_detection', False)
        }

        if self.missing_stats_:
            report.update({
                'missing_values_found': self.missing_stats_['total_missing'],
                'missing_ratio': self.missing_stats_['max_missing_ratio']
            })

        if self.outlier_stats_:
            report.update({
                'outliers_found': self.outlier_stats_['total_outliers'],
                'outlier_ratio': self.outlier_stats_['outlier_ratio']
            })

        return report


def create_preprocessing_pipeline(config: Dict[str, Any]) -> DataPreprocessor:
    """创建数据预处理管道"""
    return DataPreprocessor(config)


# CLI支持函数
def run_preprocessing_cli(X: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """运行数据预处理算法（CLI接口）"""
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # 创建数据预处理器
        preprocessor = create_preprocessing_pipeline(config)

        # 执行数据预处理
        result = preprocessor.fit_transform(X)

        if result.success:
            logger.info("数据预处理成功完成")
            log_algorithm_performance(
                logger, "DataPreprocessor",
                X.shape, result.data.shape,
                result.processing_time_ms
            )

            # 获取数据质量报告
            quality_report = preprocessor.get_data_quality_report()

            return {
                'success': True,
                'data': result.data.tolist(),
                'metadata': result.metadata,
                'quality_report': quality_report
            }
        else:
            logger.error(f"数据预处理失败: {result.error_message}")
            return {
                'success': False,
                'error': result.error_message
            }

    except Exception as e:
        error_msg = f"数据预处理异常: {str(e)}"
        logger.error(error_msg)
        return {
            'success': False,
            'error': error_msg
        }