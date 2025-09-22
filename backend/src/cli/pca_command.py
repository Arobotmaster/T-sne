#!/usr/bin/env python3
"""
PCA命令行工具 - 独立的PCA降维处理工具

符合SDD Constitution的CLI Interface原则，
可以作为独立工具使用，也可以被主CLI程序调用。
"""

import argparse
import sys
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.algorithms.pca import PCAProcessor
from src.config.logging_config import setup_logging

logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """创建PCA命令行参数解析器"""

    parser = argparse.ArgumentParser(
        prog='mof-pca',
        description='MOF数据PCA降维命令行工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本PCA降维
  mof-pca --input data.csv --output pca_result.json --n-components 10

  # 按方差保留比例降维
  mof-pca --input data.csv --output pca_result.json --variance-retention 0.95

  # 启用白化处理
  mof-pca --input data.csv --output pca_result.json --n-components 15 --whiten

  # 详细输出模式
  mof-pca --input data.csv --output pca_result.json --n-components 10 --verbose
        """
    )

    # 输入输出参数
    parser.add_argument('--input', '-i', required=True,
                       help='输入CSV文件路径')
    parser.add_argument('--output', '-o', required=True,
                       help='输出JSON文件路径')

    # PCA参数
    pca_group = parser.add_argument_group('PCA参数')
    pca_group.add_argument('--n-components', type=int, default=10,
                          help='主成分数量 (默认: 10)')
    pca_group.add_argument('--variance-retention', type=float,
                          help='方差保留比例 (0-1), 与n-components互斥')
    pca_group.add_argument('--whiten', action='store_true',
                          help='启用白化处理')
    pca_group.add_argument('--random-state', type=int, default=42,
                          help='随机种子 (默认: 42)')

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
    analysis_group.add_argument('--feature-importance', action='store_true',
                               help='输出特征重要性分析')
    analysis_group.add_argument('--cross-validation', type=int, default=5,
                               help='交叉验证折数 (默认: 5)')
    analysis_group.add_argument('--report-format', choices=['json', 'csv', 'both'],
                               default='json', help='报告格式 (默认: json)')

    return parser


def validate_arguments(args) -> Tuple[bool, Optional[str]]:
    """验证命令行参数"""
    if not Path(args.input).exists():
        return False, f"输入文件不存在: {args.input}"

    if Path(args.output).exists():
        logger.warning(f"输出文件已存在，将被覆盖: {args.output}")

    if args.variance_retention is not None:
        if not 0 < args.variance_retention <= 1:
            return False, "方差保留比例必须在0到1之间"

        if args.n_components != 10:
            return False, "variance-retention和n-components不能同时使用"

    if args.n_components <= 0:
        return False, "主成分数量必须大于0"

    return True, None


def load_data(input_path: str) -> Tuple[Any, Dict[str, Any]]:
    """加载数据文件"""
    try:
        from src.utils.io import load_csv
        data, metadata = load_csv(input_path)
        return data, metadata
    except Exception as e:
        raise ValueError(f"加载数据失败: {e}")


def perform_pca_analysis(data, args) -> Tuple[Any, Dict[str, Any]]:
    """执行PCA分析"""
    # 配置PCA参数
    config = {
        'n_components': args.n_components,
        'whiten': args.whiten,
        'random_state': args.random_state
    }

    if args.variance_retention:
        config['variance_retention'] = args.variance_retention
        del config['n_components']

    logger.info("开始PCA分析")
    logger.info(f"输入数据形状: {data.shape}")
    logger.info(f"PCA配置: {config}")

    # 执行PCA
    processor = PCAProcessor(config)
    result, metadata = processor.fit_transform(data)

    logger.info(f"PCA完成，输出形状: {result.shape}")
    if 'cumulative_variance' in metadata:
        logger.info(f"累计方差解释比例: {metadata['cumulative_variance'][-1]:.3f}")

    return result, metadata


def generate_analysis_report(data, result, metadata, args) -> Dict[str, Any]:
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
        'pca_config': metadata.get('config', {}),
        'results': metadata
    }

    # 添加特征重要性分析
    if args.feature_importance and 'feature_importance' in metadata:
        report['feature_importance'] = metadata['feature_importance']

    # 添加交叉验证结果
    if args.cross_validation > 1:
        try:
            cv_results = perform_cross_validation(data, args)
            report['cross_validation'] = cv_results
        except Exception as e:
            logger.warning(f"交叉验证失败: {e}")

    return report


def perform_cross_validation(data, args) -> Dict[str, Any]:
    """执行交叉验证分析"""
    from sklearn.model_selection import cross_val_score
    from sklearn.decomposition import PCA
    import numpy as np

    # 准备交叉验证
    pca = PCA(
        n_components=args.n_components if not args.variance_retention else None,
        whiten=args.whiten,
        random_state=args.random_state
    )

    if args.variance_retention:
        pca.set_params(n_components=args.variance_retention)

    # 使用重构误差作为评估指标
    # 这里使用负的MSE作为分数（越大越好）
    scores = cross_val_score(pca, data, cv=args.cross_validation,
                            scoring='neg_mean_squared_error')

    return {
        'cv_scores': scores.tolist(),
        'mean_score': float(np.mean(scores)),
        'std_score': float(np.std(scores)),
        'cv_folds': args.cross_validation
    }


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

    # PCA配置
    config = report['pca_config']
    report_data.append(['随机种子', config.get('random_state', 'N/A')])
    report_data.append(['白化处理', str(config.get('whiten', False))])

    # 结果
    results = report['results']
    if 'explained_variance_ratio' in results:
        evr = results['explained_variance_ratio']
        report_data.append(['解释方差比例', f"[{', '.join(f'{x:.4f}' for x in evr[:5])}]..."])

    if 'cumulative_variance' in results:
        cv = results['cumulative_variance']
        report_data.append(['累计方差比例', f"[{', '.join(f'{x:.4f}' for x in cv[:5])}]..."])

    # 特征重要性
    if 'feature_importance' in report:
        fi = report['feature_importance']
        for i, importance in enumerate(fi[:10]):  # 只显示前10个
            report_data.append([f'特征{i+1}重要性', f'{importance:.4f}'])

    # 创建DataFrame并保存
    df = pd.DataFrame(report_data, columns=['项目', '值'])
    df.to_csv(csv_path, index=False, encoding='utf-8')


def save_model(metadata, model_path):
    """保存PCA模型"""
    # 这里可以添加模型序列化逻辑
    # 目前只保存配置和元数据
    model_data = {
        'algorithm': 'PCA',
        'config': metadata.get('config', {}),
        'metadata': {k: v for k, v in metadata.items() if k != 'config'}
    }

    with open(model_path, 'w', encoding='utf-8') as f:
        json.dump(model_data, f, indent=2, ensure_ascii=False)


def print_summary(report):
    """打印分析摘要"""
    print("\n" + "="*60)
    print("PCA分析摘要")
    print("="*60)

    print(f"输入数据: {report['input_info']['shape'][0]} 样本 x {report['input_info']['shape'][1]} 特征")
    print(f"输出数据: {report['output_info']['shape'][0]} 样本 x {report['output_info']['shape'][1]} 特征")

    results = report['results']
    if 'cumulative_variance' in results:
        cv = results['cumulative_variance']
        print(f"累计方差解释比例: {cv[-1]:.3f}")

    if 'cross_validation' in report:
        cv = report['cross_validation']
        print(f"交叉验证分数: {cv['mean_score']:.4f} ± {cv['std_score']:.4f}")

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

        logger.info("开始PCA降维处理")
        logger.info(f"输入文件: {args.input}")
        logger.info(f"输出文件: {args.output}")

        # 加载数据
        data, input_metadata = load_data(args.input)

        # 执行PCA分析
        result, pca_metadata = perform_pca_analysis(data, args)

        # 生成分析报告
        report = generate_analysis_report(data, result, pca_metadata, args)

        # 保存结果
        save_results(result, pca_metadata, report, args)

        # 打印摘要
        if args.verbose:
            print_summary(report)

        logger.info("PCA处理完成")
        return 0

    except Exception as e:
        logger.error(f"PCA处理失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())