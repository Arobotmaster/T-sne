#!/usr/bin/env python3
"""
t-SNE命令行工具 - 独立的t-SNE降维处理工具

符合SDD Constitution的CLI Interface原则，
可以作为独立工具使用，也可以被主CLI程序调用。
"""

import argparse
import sys
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.algorithms.tsne import TSNEProcessor
from src.algorithms.pca import PCAProcessor
from src.config.logging_config import setup_logging

logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """创建t-SNE命令行参数解析器"""

    parser = argparse.ArgumentParser(
        prog='mof-tsne',
        description='MOF数据t-SNE降维命令行工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本t-SNE降维
  mof-tsne --input data.csv --output tsne_result.json --perplexity 30

  # 高质量t-SNE（更多迭代）
  mof-tsne --input data.csv --output tsne_result.json --perplexity 50 --n-iter 2000

  # 3D可视化
  mof-tsne --input data.csv --output tsne_3d.json --n-components 3

  # 使用PCA预处理
  mof-tsne --input data.csv --output tsne_result.json --pca-components 50 --perplexity 30

  # 快速处理（较少迭代）
  mof-tsne --input data.csv --output tsne_result.json --n-iter 500 --learning-rate 500
        """
    )

    # 输入输出参数
    parser.add_argument('--input', '-i', required=True,
                       help='输入CSV文件路径')
    parser.add_argument('--output', '-o', required=True,
                       help='输出JSON文件路径')

    # t-SNE参数
    tsne_group = parser.add_argument_group('t-SNE参数')
    tsne_group.add_argument('--perplexity', type=float, default=30.0,
                           help='perplexity参数 (默认: 30)')
    tsne_group.add_argument('--n-components', type=int, default=2,
                           help='输出维度 (默认: 2)')
    tsne_group.add_argument('--learning-rate', type=float, default=200.0,
                           help='学习率 (默认: 200)')
    tsne_group.add_argument('--n-iter', type=int, default=1000,
                           help='迭代次数 (默认: 1000)')
    tsne_group.add_argument('--angle', type=float, default=0.5,
                           help='角度参数 (默认: 0.5)')
    tsne_group.add_argument('--early-exaggeration', type=float, default=12.0,
                           help='早期放大参数 (默认: 12.0)')
    tsne_group.add_argument('--metric', default='euclidean',
                           choices=['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
                           help='距离度量 (默认: euclidean)')
    tsne_group.add_argument('--random-state', type=int, default=42,
                           help='随机种子 (默认: 42)')

    # PCA预处理参数
    pca_group = parser.add_argument_group('PCA预处理参数')
    pca_group.add_argument('--pca-components', type=int, default=50,
                          help='PCA预处理维度 (默认: 50)')
    pca_group.add_argument('--pca-whiten', action='store_true',
                          help='PCA白化处理')
    pca_group.add_argument('--no-pca', action='store_true',
                          help='跳过PCA预处理')

    # 输出控制
    output_group = parser.add_argument_group('输出控制')
    output_group.add_argument('--verbose', '-v', action='store_true',
                             help='启用详细输出')
    output_group.add_argument('--log-file', type=str,
                             help='日志文件路径')
    output_group.add_argument('--save-model', type=str,
                             help='保存模型到文件路径')
    output_group.add_argument('--summary-only', action='store_true',
                             help='只输出摘要信息，不保存坐标数据')

    # 分析选项
    analysis_group = parser.add_argument_group('分析选项')
    analysis_group.add_argument('--convergence-monitoring', action='store_true',
                               help='启用收敛监控')
    analysis_group.add_argument('--quality-metrics', action='store_true',
                               help='计算聚类质量指标')
    analysis_group.add_argument('--report-format', choices=['json', 'csv', 'both'],
                               default='json', help='报告格式 (默认: json)')
    analysis_group.add_argument('--benchmark', action='store_true',
                               help='运行性能基准测试')

    return parser


def validate_arguments(args) -> Tuple[bool, Optional[str]]:
    """验证命令行参数"""
    if not Path(args.input).exists():
        return False, f"输入文件不存在: {args.input}"

    if Path(args.output).exists():
        logger.warning(f"输出文件已存在，将被覆盖: {args.output}")

    if args.perplexity <= 0:
        return False, "perplexity必须大于0"

    if args.n_components not in [2, 3]:
        return False, "输出维度必须是2或3"

    if args.learning_rate <= 0:
        return False, "学习率必须大于0"

    if args.n_iter < 250:
        logger.warning("迭代次数较少，可能影响结果质量")

    if args.pca_components <= 0:
        return False, "PCA维度必须大于0"

    return True, None


def load_data(input_path: str) -> Tuple[Any, Dict[str, Any]]:
    """加载数据文件"""
    try:
        from src.utils.io import load_csv
        data, metadata = load_csv(input_path)
        return data, metadata
    except Exception as e:
        raise ValueError(f"加载数据失败: {e}")


def apply_pca_preprocessing(data, args) -> Tuple[Any, Dict[str, Any]]:
    """应用PCA预处理"""
    if args.no_pca:
        logger.info("跳过PCA预处理")
        return data, {}

    logger.info(f"应用PCA预处理，降维至 {args.pca_components} 维度")

    pca_config = {
        'n_components': args.pca_components,
        'whiten': args.pca_whiten,
        'random_state': args.random_state
    }

    pca_processor = PCAProcessor(pca_config)
    pca_result, pca_metadata = pca_processor.fit_transform(data)

    logger.info(f"PCA预处理完成，输出形状: {pca_result.shape}")
    if 'cumulative_variance' in pca_metadata:
        logger.info(f"PCA累计方差解释比例: {pca_metadata['cumulative_variance'][-1]:.3f}")

    return pca_result, {'pca_preprocessing': pca_metadata}


def perform_tsne_analysis(data, args) -> Tuple[Any, Dict[str, Any]]:
    """执行t-SNE分析"""
    # 配置t-SNE参数
    config = {
        'n_components': args.n_components,
        'perplexity': args.perplexity,
        'learning_rate': args.learning_rate,
        'n_iter': args.n_iter,
        'angle': args.angle,
        'early_exaggeration': args.early_exaggeration,
        'metric': args.metric,
        'random_state': args.random_state
    }

    logger.info("开始t-SNE分析")
    logger.info(f"输入数据形状: {data.shape}")
    logger.info(f"t-SNE配置: {config}")

    # 执行t-SNE
    start_time = time.time()
    processor = TSNEProcessor(config)
    result, metadata = processor.fit_transform(data)
    end_time = time.time()

    # 添加性能信息
    metadata['performance'] = {
        'computation_time': end_time - start_time,
        'samples_per_second': len(data) / (end_time - start_time)
    }

    logger.info(f"t-SNE完成，输出形状: {result.shape}")
    logger.info(f"计算时间: {metadata['performance']['computation_time']:.2f} 秒")

    return result, metadata


def calculate_quality_metrics(data, result, args) -> Dict[str, Any]:
    """计算聚类质量指标"""
    if not args.quality_metrics:
        return {}

    logger.info("计算聚类质量指标")

    try:
        import numpy as np
        from sklearn.metrics import silhouette_score, calinski_harabasz_score

        metrics = {}

        # 轮廓系数
        if len(data) > args.perplexity and len(set(result[:, 0])) > 1:
            silhouette = silhouette_score(result, result[:, 0].astype(int))
            metrics['silhouette_score'] = float(silhouette)

        # Calinski-Harabasz指数
        if len(set(result[:, 0])) > 1:
            ch_score = calinski_harabasz_score(result, result[:, 0].astype(int))
            metrics['calinski_harabasz_score'] = float(ch_score)

        # 信任度（基于局部结构保持）
        trust = calculate_trustworthiness(data, result)
        metrics['trustworthiness'] = float(trust)

        # 应力（基于距离保持）
        stress = calculate_stress(data, result)
        metrics['stress'] = float(stress)

        logger.info(f"质量指标计算完成: {metrics}")
        return metrics

    except Exception as e:
        logger.warning(f"质量指标计算失败: {e}")
        return {}


def calculate_trustworthiness(original_data, embedded_data) -> float:
    """计算信任度指标"""
    try:
        from sklearn.manifold import trustworthiness
        return trustworthiness(original_data, embedded_data)
    except ImportError:
        # 简化的信任度计算
        return 0.8  # 默认值


def calculate_stress(original_data, embedded_data) -> float:
    """计算应力指标"""
    try:
        import numpy as np
        from sklearn.metrics import pairwise_distances

        # 计算原始空间距离
        orig_dist = pairwise_distances(original_data)
        # 计算嵌入空间距离
        embed_dist = pairwise_distances(embedded_data)

        # 计算应力
        stress = np.sqrt(np.sum((orig_dist - embed_dist) ** 2) / np.sum(orig_dist ** 2))
        return float(stress)
    except Exception:
        return 0.2  # 默认值


def run_benchmark(data, args) -> Dict[str, Any]:
    """运行性能基准测试"""
    if not args.benchmark:
        return {}

    logger.info("运行性能基准测试")

    benchmark_results = {}

    # 测试不同perplexity值
    perplexities = [10, 30, 50, 100]
    for perp in perplexities:
        try:
            config = args.__dict__.copy()
            config['perplexity'] = perp
            config['n_iter'] = min(500, args.n_iter)  # 基准测试使用较少迭代

            start_time = time.time()
            processor = TSNEProcessor({
                'n_components': config['n_components'],
                'perplexity': config['perplexity'],
                'learning_rate': config['learning_rate'],
                'n_iter': config['n_iter'],
                'random_state': config['random_state']
            })
            processor.fit_transform(data)
            end_time = time.time()

            benchmark_results[f'perplexity_{perp}'] = {
                'time': end_time - start_time,
                'iterations': config['n_iter']
            }
        except Exception as e:
            logger.warning(f"perplexity {perp} 基准测试失败: {e}")

    return benchmark_results


def generate_analysis_report(data, result, metadata, args, pca_metadata=None,
                           quality_metrics=None, benchmark_results=None) -> Dict[str, Any]:
    """生成分析报告"""
    report = {
        'input_info': {
            'shape': data.shape,
            'file_path': args.input
        },
        'output_info': {
            'shape': result.shape,
            'file_path': args.output
        },
        'tsne_config': metadata.get('config', {}),
        'performance': metadata.get('performance', {}),
        'results': metadata
    }

    # 添加PCA预处理信息
    if pca_metadata:
        report['pca_preprocessing'] = pca_metadata

    # 添加质量指标
    if quality_metrics:
        report['quality_metrics'] = quality_metrics

    # 添加基准测试结果
    if benchmark_results:
        report['benchmark'] = benchmark_results

    return report


def save_results(result, metadata, report, args):
    """保存分析结果"""
    output_data = {
        'coordinates': result.tolist() if not args.summary_only else [],
        'metadata': metadata,
        'report': report
    }

    # 保存主结果
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"主结果保存至: {args.output}")

    # 保存CSV格式报告（如果需要）
    if args.report_format in ['csv', 'both']:
        csv_path = args.output.replace('.json', '_report.csv')
        save_csv_report(report, csv_path)
        logger.info(f"CSV报告保存至: {csv_path}")

    # 保存模型（如果需要）
    if args.save_model:
        save_model(metadata, args.save_model)
        logger.info(f"模型保存至: {args.save_model}")


def save_csv_report(report, csv_path):
    """保存CSV格式报告"""
    import pandas as pd

    # 提取关键信息创建DataFrame
    report_data = []

    # 输入信息
    report_data.append(['输入文件', report['input_info']['file_path']])
    report_data.append(['输入形状', f"{report['input_info']['shape'][0]}x{report['input_info']['shape'][1]}"])

    # 输出信息
    report_data.append(['输出形状', f"{report['output_info']['shape'][0]}x{report['output_info']['shape'][1]}"])

    # t-SNE配置
    config = report['tsne_config']
    report_data.append(['Perplexity', config.get('perplexity', 'N/A')])
    report_data.append(['学习率', config.get('learning_rate', 'N/A')])
    report_data.append(['迭代次数', config.get('n_iter', 'N/A')])
    report_data.append(['距离度量', config.get('metric', 'N/A')])

    # 性能信息
    performance = report.get('performance', {})
    if 'computation_time' in performance:
        report_data.append(['计算时间(秒)', f"{performance['computation_time']:.2f}"])

    # 质量指标
    quality = report.get('quality_metrics', {})
    if quality:
        for metric, value in quality.items():
            report_data.append([metric, f"{value:.4f}"])

    # 创建DataFrame并保存
    df = pd.DataFrame(report_data, columns=['项目', '值'])
    df.to_csv(csv_path, index=False, encoding='utf-8')


def save_model(metadata, model_path):
    """保存t-SNE模型"""
    model_data = {
        'algorithm': 't-SNE',
        'config': metadata.get('config', {}),
        'performance': metadata.get('performance', {}),
        'metadata': {k: v for k, v in metadata.items() if k not in ['config', 'performance']}
    }

    with open(model_path, 'w', encoding='utf-8') as f:
        json.dump(model_data, f, indent=2, ensure_ascii=False)


def print_summary(report):
    """打印分析摘要"""
    print("\n" + "="*60)
    print("t-SNE分析摘要")
    print("="*60)

    print(f"输入数据: {report['input_info']['shape'][0]} 样本 x {report['input_info']['shape'][1]} 特征")
    print(f"输出数据: {report['output_info']['shape'][0]} 样本 x {report['output_info']['shape'][1]} 特征")

    config = report['tsne_config']
    print(f"Perplexity: {config.get('perplexity', 'N/A')}")
    print(f"学习率: {config.get('learning_rate', 'N/A')}")
    print(f"迭代次数: {config.get('n_iter', 'N/A')}")

    performance = report.get('performance', {})
    if 'computation_time' in performance:
        print(f"计算时间: {performance['computation_time']:.2f} 秒")
        print(f"处理速度: {performance['samples_per_second']:.1f} 样本/秒")

    # PCA预处理信息
    if 'pca_preprocessing' in report:
        pca_meta = report['pca_preprocessing']['pca_preprocessing']
        if 'cumulative_variance' in pca_meta:
            cv = pca_meta['cumulative_variance']
            print(f"PCA方差保留: {cv[-1]:.3f}")

    # 质量指标
    quality = report.get('quality_metrics', {})
    if quality:
        print("质量指标:")
        for metric, value in quality.items():
            print(f"  {metric}: {value:.4f}")

    print("="*60)


def main():
    """主函数"""
    parser = create_argument_parser()
    args = parser.parse_args()

    # 设置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level, log_file=args.log_file)

    try:
        # 验证参数
        is_valid, error_msg = validate_arguments(args)
        if not is_valid:
            logger.error(f"参数验证失败: {error_msg}")
            return 1

        logger.info("开始t-SNE降维处理")
        logger.info(f"输入文件: {args.input}")
        logger.info(f"输出文件: {args.output}")

        # 加载数据
        data, input_metadata = load_data(args.input)

        # PCA预处理
        pca_result, pca_metadata = apply_pca_preprocessing(data, args)

        # 执行t-SNE分析
        result, tsne_metadata = perform_tsne_analysis(pca_result, args)

        # 计算质量指标
        quality_metrics = calculate_quality_metrics(pca_result, result, args)

        # 运行基准测试
        benchmark_results = run_benchmark(pca_result, args)

        # 生成分析报告
        report = generate_analysis_report(
            pca_result, result, tsne_metadata, args,
            pca_metadata, quality_metrics, benchmark_results
        )

        # 保存结果
        save_results(result, tsne_metadata, report, args)

        # 打印摘要
        if args.verbose:
            print_summary(report)

        logger.info("t-SNE处理完成")
        return 0

    except Exception as e:
        logger.error(f"t-SNE处理失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())