"""
t-SNE降维算法实现

遵循SDD Constitution的Library-First原则，
实现独立可测试的t-SNE算法库。
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.manifold import TSNE
import logging
import time

from .base import DimensionalityReductionAlgorithm, AlgorithmResult, validate_algorithm_config, log_algorithm_performance


class TSNEProcessor(DimensionalityReductionAlgorithm):
    """t-SNE降维处理器

    遵循SDD Constitution原则：
    - Library-First: 独立可重用的算法库
    - CLI Interface: 支持命令行调用
    - Scientific Observability: 详细的科学计算日志
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化t-SNE处理器

        Args:
            config: t-SNE配置参数
                - perplexity: 困惑度
                - n_components: 输出维度
                - learning_rate: 学习率
                - n_iter: 迭代次数
                - random_state: 随机种子
                - metric: 距离度量
        """
        super().__init__(config)
        self.tsne = None
        self.kl_divergence_ = None
        self.n_iter_ = None

        # 验证配置
        validate_algorithm_config(config, ['perplexity'])

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'TSNEProcessor':
        """
        训练t-SNE模型

        Args:
            X: 输入特征矩阵 (n_samples, n_features)
            y: 未使用

        Returns:
            self: 训练后的t-SNE处理器
        """
        self.validate_input(X)

        # 验证数据量与perplexity的关系
        n_samples = X.shape[0]
        perplexity = self.config['perplexity']
        if perplexity >= n_samples / 5:
            raise ValueError(f"Perplexity ({perplexity}) 过大，建议小于样本数的1/5 ({n_samples/5:.1f})")

        # 创建t-SNE模型（兼容不同sklearn版本的参数名）
        tsne_params = {
            'n_components': self.config.get('n_components', 2),
            'perplexity': perplexity,
            'learning_rate': self.config.get('learning_rate', 200),
            'n_iter': self.config.get('n_iter', 1000),
            'random_state': self.config.get('random_state', 42),
            'metric': self.config.get('metric', 'euclidean'),
            'angle': self.config.get('angle', 0.5),
            'early_exaggeration': self.config.get('early_exaggeration', 12.0)
        }
        # 过滤/映射到实际支持的参数
        try:
            import inspect
            sig = inspect.signature(TSNE.__init__)
            valid = set(sig.parameters.keys()) - { 'self' }
        except Exception:
            valid = None

        final_params = {}
        for k, v in tsne_params.items():
            if valid is None or k in valid:
                final_params[k] = v
            elif k == 'n_iter' and 'max_iter' in valid:
                final_params['max_iter'] = v

        self.tsne = TSNE(**final_params)

        self.logger.info(f"开始t-SNE训练")
        self.logger.info(f"数据形状: {X.shape}")
        self.logger.info(f"参数: perplexity={perplexity}, learning_rate={tsne_params['learning_rate']}")

        # 训练t-SNE模型
        start_time = time.time()
        self.tsne.fit(X)
        training_time = time.time() - start_time

        # 保存训练信息
        self.kl_divergence_ = self.tsne.kl_divergence_
        self.n_iter_ = self.tsne.n_iter_

        self.is_fitted = True
        self.fitted_data_shape = X.shape

        self.logger.info(f"t-SNE训练完成")
        self.logger.info(f"训练时间: {training_time:.2f}秒")
        self.logger.info(f"KL散度: {self.kl_divergence_:.4f}")
        self.logger.info(f"实际迭代次数: {self.n_iter_}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        应用t-SNE降维
        注意：t-SNE的transform方法实际上会重新运行整个算法

        Args:
            X: 输入特征矩阵

        Returns:
            np.ndarray: 降维后的数据
        """
        if not self.is_fitted:
            raise ValueError("t-SNE模型未训练，请先调用fit方法")

        self.validate_input(X)

        # t-SNE的transform会重新运行整个算法
        self.logger.info("重新运行t-SNE降维...")
        transformed_data = self.tsne.fit_transform(X)

        return transformed_data

    def _generate_metadata(self, X: np.ndarray, transformed_data: np.ndarray) -> Dict[str, Any]:
        """
        生成t-SNE元数据

        Args:
            X: 原始数据
            transformed_data: 降维后的数据

        Returns:
            Dict[str, Any]: 元数据字典
        """
        metadata = {
            'algorithm': 't-SNE',
            'version': self.version,
            'input_shape': X.shape,
            'output_shape': transformed_data.shape,
            'kl_divergence': float(self.kl_divergence_),
            'n_iter': self.n_iter_,
            'config': self.config
        }

        # 添加坐标统计信息
        metadata['coordinate_stats'] = {
            'x_mean': float(np.mean(transformed_data[:, 0])),
            'x_std': float(np.std(transformed_data[:, 0])),
            'y_mean': float(np.mean(transformed_data[:, 1])),
            'y_std': float(np.std(transformed_data[:, 1])),
            'x_range': [float(np.min(transformed_data[:, 0])), float(np.max(transformed_data[:, 0]))],
            'y_range': [float(np.min(transformed_data[:, 1])), float(np.max(transformed_data[:, 1]))]
        }

        # 添加距离统计
        if transformed_data.shape[1] >= 2:
            distances = np.sqrt(np.sum(np.diff(transformed_data, axis=0)**2, axis=1))
            metadata['distance_stats'] = {
                'mean_distance': float(np.mean(distances)),
                'std_distance': float(np.std(distances)),
                'min_distance': float(np.min(distances)),
                'max_distance': float(np.max(distances))
            }

        return metadata

    def get_explained_variance(self) -> Optional[np.ndarray]:
        """t-SNE不提供解释方差"""
        return None

    def get_components(self) -> Optional[np.ndarray]:
        """t-SNE不提供主成分"""
        return None

    def get_kl_divergence(self) -> Optional[float]:
        """获取KL散度"""
        return self.kl_divergence_

    @staticmethod
    def recommend_parameters(X: np.ndarray) -> Dict[str, Any]:
        """
        根据数据特征推荐t-SNE参数

        Args:
            X: 输入数据

        Returns:
            Dict[str, Any]: 推荐的参数
        """
        n_samples, n_features = X.shape

        # 基于样本数推荐perplexity
        if n_samples < 100:
            perplexity = max(5, n_samples // 10)
        elif n_samples < 1000:
            perplexity = 30
        elif n_samples < 10000:
            perplexity = 50
        else:
            perplexity = min(100, n_samples // 50)

        # 基于样本数推荐迭代次数
        if n_samples < 1000:
            n_iter = 1000
        elif n_samples < 5000:
            n_iter = 1500
        else:
            n_iter = 2000

        return {
            'perplexity': perplexity,
            'n_components': 2,
            'learning_rate': 200,
            'n_iter': n_iter,
            'random_state': 42,
            'metric': 'euclidean',
            'angle': 0.5,
            'early_exaggeration': 12.0
        }

    def optimize_perplexity(self, X: np.ndarray, perplexity_values: list = None) -> Dict[str, Any]:
        """
        优化perplexity参数

        Args:
            X: 输入数据
            perplexity_values: 要测试的perplexity值列表

        Returns:
            Dict[str, Any]: 优化结果
        """
        if perplexity_values is None:
            perplexity_values = [10, 20, 30, 40, 50]

        results = []

        for perplexity in perplexity_values:
            try:
                test_config = self.config.copy()
                test_config['perplexity'] = perplexity

                test_tsne = TSNEProcessor(test_config)
                result = test_tsne.fit_transform(X)

                if result.success:
                    kl_divergence = result.metadata['kl_divergence']
                    results.append({
                        'perplexity': perplexity,
                        'kl_divergence': kl_divergence,
                        'success': True
                    })

            except Exception as e:
                self.logger.warning(f"Perplexity {perplexity} 测试失败: {str(e)}")
                results.append({
                    'perplexity': perplexity,
                    'kl_divergence': float('inf'),
                    'success': False,
                    'error': str(e)
                })

        # 选择KL散度最小的perplexity
        successful_results = [r for r in results if r['success']]
        if successful_results:
            best_result = min(successful_results, key=lambda x: x['kl_divergence'])
            return {
                'best_perplexity': best_result['perplexity'],
                'best_kl_divergence': best_result['kl_divergence'],
                'all_results': results
            }
        else:
            return {
                'error': '所有perplexity值测试失败',
                'all_results': results
            }


def create_tsne_pipeline(config: Dict[str, Any]) -> TSNEProcessor:
    """
    创建t-SNE处理管道

    Args:
        config: t-SNE配置

    Returns:
        TSNEProcessor: 配置好的t-SNE处理器
    """
    return TSNEProcessor(config)


# CLI支持函数
def run_tsne_cli(X: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    运行t-SNE算法（CLI接口）

    Args:
        X: 输入数据
        config: t-SNE配置

    Returns:
        Dict[str, Any]: 处理结果
    """
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # 创建t-SNE处理器
        tsne = create_tsne_pipeline(config)

        # 执行t-SNE
        result = tsne.fit_transform(X)

        if result.success:
            logger.info("t-SNE降维成功完成")
            log_algorithm_performance(
                logger, "t-SNE",
                X.shape, result.data.shape,
                result.processing_time_ms
            )

            return {
                'success': True,
                'data': result.data.tolist(),
                'metadata': result.metadata
            }
        else:
            logger.error(f"t-SNE降维失败: {result.error_message}")
            return {
                'success': False,
                'error': result.error_message
            }

    except Exception as e:
        error_msg = f"t-SNE处理异常: {str(e)}"
        logger.error(error_msg)
        return {
            'success': False,
            'error': error_msg
        }
