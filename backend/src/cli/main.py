#!/usr/bin/env python3
"""
MOF数据t-SNE交互式可视化 - CLI主程序

符合SDD Constitution的CLI Interface原则，
提供统一的命令行接口访问所有算法功能。
"""

import argparse
import sys
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.algorithms.pca import PCAProcessor
from src.algorithms.tsne import TSNEProcessor
from src.algorithms.preprocessing import Preprocessor
from src.config.logging_config import setup_logging

logger = logging.getLogger(__name__)


def setup_argument_parser() -> argparse.ArgumentParser:
    """设置命令行参数解析器"""

    parser = argparse.ArgumentParser(
        prog='mof-visualization',
        description='MOF数据t-SNE交互式可视化命令行工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # PCA降维
  mof-visualization pca --input data.csv --output pca_result.json --n-components 10

  # t-SNE降维
  mof-visualization tsne --input data.csv --output tsne_result.json --perplexity 30

  # 数据预处理
  mof-visualization preprocess --input data.csv --output clean_data.csv --strategy median

  # 批处理多个文件
  mof-visualization batch --input-dir ./data/ --output-dir ./results/ --config batch_config.json
        """
    )

    # 全局参数
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='启用详细输出')
    parser.add_argument('--log-file', type=str,
                       help='日志文件路径')
    parser.add_argument('--config', type=str,
                       help='配置文件路径 (JSON格式)')

    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # PCA命令
    pca_parser = subparsers.add_parser('pca', help='PCA降维处理')
    pca_parser.add_argument('--input', '-i', required=True,
                           help='输入CSV文件路径')
    pca_parser.add_argument('--output', '-o', required=True,
                           help='输出JSON文件路径')
    pca_parser.add_argument('--n-components', type=int, default=10,
                           help='主成分数量 (默认: 10)')
    pca_parser.add_argument('--variance-retention', type=float,
                           help='方差保留比例 (0-1), 与n-components互斥')
    pca_parser.add_argument('--whiten', action='store_true',
                           help='启用白化处理')
    pca_parser.add_argument('--random-state', type=int, default=42,
                           help='随机种子 (默认: 42)')

    # t-SNE命令
    tsne_parser = subparsers.add_parser('tsne', help='t-SNE降维处理')
    tsne_parser.add_argument('--input', '-i', required=True,
                            help='输入CSV文件路径')
    tsne_parser.add_argument('--output', '-o', required=True,
                            help='输出JSON文件路径')
    tsne_parser.add_argument('--perplexity', type=float, default=30.0,
                            help='perplexity参数 (默认: 30)')
    tsne_parser.add_argument('--n-components', type=int, default=2,
                            help='输出维度 (默认: 2)')
    tsne_parser.add_argument('--learning-rate', type=float, default=200.0,
                            help='学习率 (默认: 200)')
    tsne_parser.add_argument('--n-iter', type=int, default=1000,
                            help='迭代次数 (默认: 1000)')
    tsne_parser.add_argument('--pca-components', type=int, default=50,
                            help='PCA预处理维度 (默认: 50)')
    tsne_parser.add_argument('--random-state', type=int, default=42,
                            help='随机种子 (默认: 42)')

    # 预处理命令
    preprocess_parser = subparsers.add_parser('preprocess', help='数据预处理')
    preprocess_parser.add_argument('--input', '-i', required=True,
                                 help='输入CSV文件路径')
    preprocess_parser.add_argument('--output', '-o', required=True,
                                 help='输出CSV文件路径')
    preprocess_parser.add_argument('--missing-strategy',
                                 choices=['mean', 'median', 'mode', 'drop'],
                                 default='median',
                                 help='缺失值处理策略 (默认: median)')
    preprocess_parser.add_argument('--scaling-method',
                                 choices=['standard', 'minmax', 'robust', 'none'],
                                 default='standard',
                                 help='数据缩放方法 (默认: standard)')
    preprocess_parser.add_argument('--outlier-detection', action='store_true',
                                 help='启用异常值检测')
    preprocess_parser.add_argument('--outlier-threshold', type=float, default=3.0,
                                 help='异常值阈值 (默认: 3.0)')
    preprocess_parser.add_argument('--feature-selection', action='store_true',
                                 help='启用特征选择')
    preprocess_parser.add_argument('--correlation-threshold', type=float, default=0.95,
                                 help='特征选择相关系数阈值 (默认: 0.95)')

    # 批处理命令
    batch_parser = subparsers.add_parser('batch', help='批处理多个文件')
    batch_parser.add_argument('--input-dir', '-i', required=True,
                             help='输入目录路径')
    batch_parser.add_argument('--output-dir', '-o', required=True,
                             help='输出目录路径')
    batch_parser.add_argument('--config', '-c', required=True,
                             help='批处理配置文件路径 (JSON)')
    batch_parser.add_argument('--parallel', type=int, default=1,
                             help='并行处理数量 (默认: 1)')

    return parser


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """加载配置文件"""
    if not config_path:
        return {}

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return {}


def handle_pca_command(args) -> int:
    """处理PCA命令"""
    logger.info("开始PCA降维处理")

    try:
        from src.utils.io import load_csv, save_json

        # 加载数据
        data, metadata = load_csv(args.input)
        logger.info(f"加载数据: {metadata['n_samples']} 样本, {metadata['n_features']} 特征")

        # 配置PCA参数
        config = {
            'n_components': args.n_components,
            'whiten': args.whiten,
            'random_state': args.random_state
        }

        if args.variance_retention:
            config['variance_retention'] = args.variance_retention
            del config['n_components']

        # 执行PCA
        processor = PCAProcessor(config)
        result, pca_metadata = processor.fit_transform(data)

        # 保存结果
        output_data = {
            'coordinates': result.tolist(),
            'metadata': {
                'input_shape': data.shape,
                'output_shape': result.shape,
                'algorithm': 'PCA',
                'config': config,
                **pca_metadata
            }
        }

        save_json(output_data, args.output)
        logger.info(f"PCA处理完成，结果保存至: {args.output}")

        return 0

    except Exception as e:
        logger.error(f"PCA处理失败: {e}")
        return 1


def handle_tsne_command(args) -> int:
    """处理t-SNE命令"""
    logger.info("开始t-SNE降维处理")

    try:
        from src.utils.io import load_csv, save_json

        # 加载数据
        data, metadata = load_csv(args.input)
        logger.info(f"加载数据: {metadata['n_samples']} 样本, {metadata['n_features']} 特征")

        # 配置t-SNE参数
        config = {
            'n_components': args.n_components,
            'perplexity': args.perplexity,
            'learning_rate': args.learning_rate,
            'n_iter': args.n_iter,
            'random_state': args.random_state
        }

        # 执行t-SNE
        processor = TSNEProcessor(config)
        result, tsne_metadata = processor.fit_transform(data)

        # 保存结果
        output_data = {
            'coordinates': result.tolist(),
            'metadata': {
                'input_shape': data.shape,
                'output_shape': result.shape,
                'algorithm': 't-SNE',
                'config': config,
                **tsne_metadata
            }
        }

        save_json(output_data, args.output)
        logger.info(f"t-SNE处理完成，结果保存至: {args.output}")

        return 0

    except Exception as e:
        logger.error(f"t-SNE处理失败: {e}")
        return 1


def handle_preprocess_command(args) -> int:
    """处理预处理命令"""
    logger.info("开始数据预处理")

    try:
        from src.utils.io import load_csv, save_csv

        # 加载数据
        data, metadata = load_csv(args.input)
        logger.info(f"加载数据: {metadata['n_samples']} 样本, {metadata['n_features']} 特征")

        # 配置预处理参数
        config = {
            'missing_value_strategy': args.missing_strategy,
            'scaling_method': args.scaling_method,
            'outlier_detection': args.outlier_detection,
            'outlier_threshold': args.outlier_threshold,
            'feature_selection': args.feature_selection,
            'correlation_threshold': args.correlation_threshold
        }

        # 执行预处理
        processor = Preprocessor(config)
        result, preprocess_metadata = processor.fit_transform(data)

        # 保存结果
        save_csv(result, args.output)
        logger.info(f"数据预处理完成，结果保存至: {args.output}")

        return 0

    except Exception as e:
        logger.error(f"数据预处理失败: {e}")
        return 1


def handle_batch_command(args) -> int:
    """处理批处理命令"""
    logger.info("开始批处理")

    try:
        from src.utils.io import load_csv, save_json
        import glob
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # 加载配置
        with open(args.config, 'r', encoding='utf-8') as f:
            batch_config = json.load(f)

        # 获取输入文件
        input_pattern = f"{args.input_dir}/*.csv"
        input_files = glob.glob(input_pattern)

        if not input_files:
            logger.error(f"在目录 {args.input_dir} 中未找到CSV文件")
            return 1

        logger.info(f"找到 {len(input_files)} 个文件待处理")

        # 创建输出目录
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 处理函数
        def process_file(input_file):
            try:
                filename = Path(input_file).stem
                output_file = output_dir / f"{filename}_result.json"

                # 加载数据
                data, metadata = load_csv(input_file)

                # 根据配置选择处理方式
                if batch_config.get('algorithm') == 'pca':
                    processor = PCAProcessor(batch_config.get('pca_config', {}))
                    result, algo_metadata = processor.fit_transform(data)
                elif batch_config.get('algorithm') == 'tsne':
                    processor = TSNEProcessor(batch_config.get('tsne_config', {}))
                    result, algo_metadata = processor.fit_transform(data)
                else:
                    raise ValueError(f"不支持的算法: {batch_config.get('algorithm')}")

                # 保存结果
                output_data = {
                    'coordinates': result.tolist(),
                    'metadata': {
                        'input_shape': data.shape,
                        'output_shape': result.shape,
                        'algorithm': batch_config.get('algorithm'),
                        'input_file': input_file,
                        **algo_metadata
                    }
                }

                save_json(output_data, str(output_file))
                return input_file, None

            except Exception as e:
                return input_file, str(e)

        # 并行处理
        results = []
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = [executor.submit(process_file, file) for file in input_files]

            for future in as_completed(futures):
                input_file, error = future.result()
                if error:
                    logger.error(f"处理文件 {input_file} 失败: {error}")
                else:
                    logger.info(f"成功处理文件: {input_file}")
                    results.append(input_file)

        logger.info(f"批处理完成，成功处理 {len(results)} 个文件")
        return 0

    except Exception as e:
        logger.error(f"批处理失败: {e}")
        return 1


def main():
    """主函数"""
    parser = setup_argument_parser()
    args = parser.parse_args()

    # 设置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level, log_file=args.log_file)

    # 加载配置
    config = load_config(args.config)

    # 检查命令
    if not args.command:
        parser.print_help()
        return 1

    logger.info(f"执行命令: {args.command}")

    # 执行相应命令
    if args.command == 'pca':
        return handle_pca_command(args)
    elif args.command == 'tsne':
        return handle_tsne_command(args)
    elif args.command == 'preprocess':
        return handle_preprocess_command(args)
    elif args.command == 'batch':
        return handle_batch_command(args)
    else:
        logger.error(f"未知命令: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())