#!/usr/bin/env python3
"""
批处理命令行工具 - 批量处理多个数据文件

符合SDD Constitution的CLI Interface原则，
支持并行处理、错误恢复和详细的进度跟踪。
"""

import argparse
import sys
import logging
import json
import time
import glob
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from dataclasses import dataclass, asdict
from enum import Enum

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.algorithms.pca import PCAProcessor
from src.algorithms.tsne import TSNEProcessor
from src.algorithms.preprocessing import Preprocessor
from src.config.logging_config import setup_logging

logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """处理状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class BatchTask:
    """批处理任务"""
    input_file: str
    output_file: str
    algorithm: str
    config: Dict[str, Any]
    status: ProcessingStatus = ProcessingStatus.PENDING
    error_message: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    processing_time: Optional[float] = None


@dataclass
class BatchResult:
    """批处理结果"""
    total_files: int
    successful_files: int
    failed_files: int
    skipped_files: int
    total_processing_time: float
    average_processing_time: float
    tasks: List[BatchTask]
    summary: Dict[str, Any]


def create_argument_parser() -> argparse.ArgumentParser:
    """创建批处理命令行参数解析器"""

    parser = argparse.ArgumentParser(
        prog='mof-batch',
        description='MOF数据批处理命令行工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本批处理
  mof-batch --input-dir ./data/ --output-dir ./results/ --config batch_config.json

  # 并行处理
  mof-batch --input-dir ./data/ --output-dir ./results/ --config batch_config.json --parallel 4

  # 使用进程池（CPU密集型任务）
  mof-batch --input-dir ./data/ --output-dir ./results/ --config batch_config.json --parallel 4 --use-processes

  # 重新运行失败的文件
  mof-batch --input-dir ./data/ --output-dir ./results/ --config batch_config.json --retry-failed

  # 只处理特定模式的文件
  mof-batch --input-dir ./data/ --output-dir ./results/ --config batch_config.json --file-pattern "*.csv"

配置文件示例 (batch_config.json):
{
  "algorithm": "pca",
  "pca_config": {
    "n_components": 10,
    "whiten": true,
    "random_state": 42
  },
  "output_format": "json",
  "include_metadata": true,
  "max_file_size_mb": 100
}
        """
    )

    # 输入输出参数
    parser.add_argument('--input-dir', '-i', required=True,
                       help='输入目录路径')
    parser.add_argument('--output-dir', '-o', required=True,
                       help='输出目录路径')
    parser.add_argument('--config', '-c', required=True,
                       help='批处理配置文件路径 (JSON)')

    # 文件处理参数
    file_group = parser.add_argument_group('文件处理参数')
    file_group.add_argument('--file-pattern', default='*.csv',
                           help='文件匹配模式 (默认: *.csv)')
    file_group.add_argument('--recursive', '-r', action='store_true',
                           help='递归搜索子目录')
    file_group.add_argument('--max-file-size', type=int, default=1024,
                           help='最大文件大小 (MB, 默认: 1024)')
    file_group.add_argument('--skip-existing', action='store_true',
                           help='跳过已存在的输出文件')
    file_group.add_argument('--retry-failed', action='store_true',
                           help='重新运行失败的文件')

    # 并行处理参数
    parallel_group = parser.add_argument_group('并行处理参数')
    parallel_group.add_argument('--parallel', type=int, default=1,
                               help='并行处理数量 (默认: 1)')
    parallel_group.add_argument('--use-processes', action='store_true',
                               help='使用进程池而不是线程池')
    parallel_group.add_argument('--chunk-size', type=int, default=1,
                               help='每个任务的块大小 (默认: 1)')

    # 输出控制
    output_group = parser.add_argument_group('输出控制')
    output_group.add_argument('--verbose', '-v', action='store_true',
                             help='启用详细输出')
    output_group.add_argument('--log-file', type=str,
                             help='日志文件路径')
    output_group.add_argument('--progress-interval', type=int, default=10,
                             help='进度更新间隔 (秒, 默认: 10)')
    output_group.add_argument('--save-summary', action='store_true',
                             help='保存处理摘要')
    output_group.add_argument('--output-format', choices=['json', 'csv', 'both'],
                             default='json', help='输出格式 (默认: json)')

    # 错误处理
    error_group = parser.add_argument_group('错误处理')
    error_group.add_argument('--continue-on-error', action='store_true',
                            help='遇到错误时继续处理')
    error_group.add_argument('--max-retries', type=int, default=0,
                            help='最大重试次数 (默认: 0)')
    error_group.add_argument('--retry-delay', type=float, default=1.0,
                            help='重试延迟 (秒, 默认: 1.0)')

    return parser


def load_batch_config(config_path: str) -> Dict[str, Any]:
    """加载批处理配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 验证必需字段
        required_fields = ['algorithm']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"配置文件缺少必需字段: {field}")

        # 设置默认值
        config.setdefault('output_format', 'json')
        config.setdefault('include_metadata', True)
        config.setdefault('max_file_size_mb', 100)

        return config

    except Exception as e:
        raise ValueError(f"加载配置文件失败: {e}")


def discover_input_files(input_dir: str, file_pattern: str, recursive: bool = False) -> List[str]:
    """发现输入文件"""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"输入目录不存在: {input_dir}")

    if recursive:
        pattern = f"**/{file_pattern}"
        files = list(input_path.rglob(pattern))
    else:
        pattern = file_pattern
        files = list(input_path.glob(pattern))

    # 过滤并排序
    files = [str(f) for f in files if f.is_file()]
    files.sort()

    logger.info(f"发现 {len(files)} 个输入文件")
    return files


def filter_files_by_size(files: List[str], max_size_mb: int) -> List[str]:
    """根据文件大小过滤文件"""
    filtered_files = []
    for file_path in files:
        try:
            size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            if size_mb <= max_size_mb:
                filtered_files.append(file_path)
            else:
                logger.warning(f"跳过过大文件: {file_path} ({size_mb:.1f}MB > {max_size_mb}MB)")
        except Exception as e:
            logger.warning(f"无法获取文件大小: {file_path} ({e})")

    logger.info(f"文件大小过滤后剩余 {len(filtered_files)} 个文件")
    return filtered_files


def create_batch_tasks(files: List[str], output_dir: str, config: Dict[str, Any]) -> List[BatchTask]:
    """创建批处理任务"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tasks = []
    for input_file in files:
        # 生成输出文件名
        input_name = Path(input_file).stem
        output_ext = config.get('output_format', 'json')
        output_file = output_path / f"{input_name}_result.{output_ext}"

        task = BatchTask(
            input_file=input_file,
            output_file=str(output_file),
            algorithm=config['algorithm'],
            config=config.get(f"{config['algorithm']}_config", {})
        )

        tasks.append(task)

    return tasks


def filter_existing_tasks(tasks: List[BatchTask]) -> List[BatchTask]:
    """过滤已存在的任务"""
    filtered_tasks = []
    for task in tasks:
        if Path(task.output_file).exists():
            task.status = ProcessingStatus.SKIPPED
            logger.info(f"跳过已存在的输出文件: {task.output_file}")
        else:
            filtered_tasks.append(task)

    return filtered_tasks


def process_single_task(task: BatchTask) -> BatchTask:
    """处理单个任务"""
    import traceback

    retry_count = 0
    max_retries = getattr(process_single_task, 'max_retries', 0)
    retry_delay = getattr(process_single_task, 'retry_delay', 1.0)

    while retry_count <= max_retries:
        try:
            task.status = ProcessingStatus.RUNNING
            task.start_time = time.time()

            logger.info(f"开始处理: {task.input_file}")

            # 加载数据
            from src.utils.io import load_csv
            data, metadata = load_csv(task.input_file)

            # 根据算法选择处理器
            if task.algorithm == 'pca':
                processor = PCAProcessor(task.config)
            elif task.algorithm == 'tsne':
                processor = TSNEProcessor(task.config)
            elif task.algorithm == 'preprocess':
                processor = Preprocessor(task.config)
            else:
                raise ValueError(f"不支持的算法: {task.algorithm}")

            # 执行处理
            result, algo_metadata = processor.fit_transform(data)

            # 准备输出数据
            output_data = {
                'coordinates': result.tolist() if hasattr(result, 'tolist') else result,
                'metadata': {
                    'input_file': task.input_file,
                    'algorithm': task.algorithm,
                    'config': task.config,
                    'input_shape': data.shape,
                    'output_shape': result.shape if hasattr(result, 'shape') else (len(result),),
                    **metadata
                }
            }

            # 保存结果
            if task.output_file.endswith('.json'):
                with open(task.output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
            elif task.output_file.endswith('.csv'):
                import pandas as pd
                # 保存坐标数据
                pd.DataFrame(result).to_csv(task.output_file, index=False)

            task.status = ProcessingStatus.COMPLETED
            task.end_time = time.time()
            task.processing_time = task.end_time - task.start_time

            logger.info(f"处理完成: {task.input_file} ({task.processing_time:.2f}秒)")
            return task

        except Exception as e:
            retry_count += 1
            task.error_message = str(e)

            if retry_count <= max_retries:
                logger.warning(f"处理失败 (尝试 {retry_count}/{max_retries}): {task.input_file} - {e}")
                if retry_delay > 0:
                    time.sleep(retry_delay)
            else:
                task.status = ProcessingStatus.FAILED
                task.end_time = time.time()
                if task.start_time:
                    task.processing_time = task.end_time - task.start_time

                logger.error(f"处理失败: {task.input_file} - {e}")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(traceback.format_exc())

                return task


def process_tasks_parallel(tasks: List[BatchTask], parallel_count: int,
                          use_processes: bool = False, continue_on_error: bool = False) -> BatchResult:
    """并行处理任务"""
    # 设置重试参数
    process_single_task.max_retries = getattr(process_single_task, 'max_retries', 0)
    process_single_task.retry_delay = getattr(process_single_task, 'retry_delay', 1.0)

    # 选择执行器
    executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

    # 筛选待处理任务
    pending_tasks = [task for task in tasks if task.status == ProcessingStatus.PENDING]

    logger.info(f"开始并行处理 {len(pending_tasks)} 个任务，并行度: {parallel_count}")

    start_time = time.time()
    completed_tasks = []
    failed_tasks = []

    with executor_class(max_workers=parallel_count) as executor:
        # 提交任务
        future_to_task = {executor.submit(process_single_task, task): task for task in pending_tasks}

        # 处理完成的任务
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result_task = future.result()
                completed_tasks.append(result_task)

                if result_task.status == ProcessingStatus.FAILED:
                    failed_tasks.append(result_task)
                    if not continue_on_error:
                        # 取消所有未完成的任务
                        for f in future_to_task:
                            f.cancel()
                        break

            except Exception as e:
                logger.error(f"任务执行异常: {task.input_file} - {e}")
                if not continue_on_error:
                    # 取消所有未完成的任务
                    for f in future_to_task:
                        f.cancel()
                    break

    end_time = time.time()

    # 计算统计信息
    successful_tasks = [t for t in completed_tasks if t.status == ProcessingStatus.COMPLETED]
    skipped_tasks = [t for t in tasks if t.status == ProcessingStatus.SKIPPED]

    total_processing_time = sum(t.processing_time or 0 for t in successful_tasks)
    average_processing_time = total_processing_time / len(successful_tasks) if successful_tasks else 0

    return BatchResult(
        total_files=len(tasks),
        successful_files=len(successful_tasks),
        failed_files=len(failed_tasks),
        skipped_files=len(skipped_tasks),
        total_processing_time=total_processing_time,
        average_processing_time=average_processing_time,
        tasks=tasks,
        summary={
            'start_time': start_time,
            'end_time': end_time,
            'total_time': end_time - start_time,
            'parallel_count': parallel_count,
            'use_processes': use_processes
        }
    )


def save_batch_summary(result: BatchResult, output_dir: str):
    """保存批处理摘要"""
    summary_path = Path(output_dir) / 'batch_summary.json'

    summary_data = {
        'statistics': {
            'total_files': result.total_files,
            'successful_files': result.successful_files,
            'failed_files': result.failed_files,
            'skipped_files': result.skipped_files,
            'success_rate': result.successful_files / result.total_files if result.total_files > 0 else 0,
            'total_processing_time': result.total_processing_time,
            'average_processing_time': result.average_processing_time
        },
        'summary': result.summary,
        'failed_tasks': [
            {
                'input_file': task.input_file,
                'error_message': task.error_message,
                'processing_time': task.processing_time
            }
            for task in result.tasks if task.status == ProcessingStatus.FAILED
        ],
        'performance_metrics': {
            'throughput': result.total_files / result.summary['total_time'],
            'efficiency': result.total_processing_time / result.summary['total_time']
        }
    }

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    logger.info(f"批处理摘要保存至: {summary_path}")


def print_progress_report(result: BatchResult):
    """打印进度报告"""
    print("\n" + "="*60)
    print("批处理进度报告")
    print("="*60)

    print(f"总文件数: {result.total_files}")
    print(f"成功处理: {result.successful_files}")
    print(f"处理失败: {result.failed_files}")
    print(f"跳过文件: {result.skipped_files}")

    if result.total_files > 0:
        success_rate = result.successful_files / result.total_files * 100
        print(f"成功率: {success_rate:.1f}%")

    print(f"总处理时间: {result.total_processing_time:.2f} 秒")
    print(f"平均处理时间: {result.average_processing_time:.2f} 秒/文件")
    print(f"总运行时间: {result.summary['total_time']:.2f} 秒")

    if result.summary['total_time'] > 0:
        throughput = result.total_files / result.summary['total_time']
        print(f"处理速度: {throughput:.2f} 文件/秒")

    if result.failed_files > 0:
        print("\n失败的文件:")
        for task in result.tasks:
            if task.status == ProcessingStatus.FAILED:
                print(f"  - {task.input_file}: {task.error_message}")

    print("="*60)


def main():
    """主函数"""
    parser = create_argument_parser()
    args = parser.parse_args()

    # 设置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level, log_file=args.log_file)

    try:
        logger.info("开始批处理")

        # 设置重试参数
        process_single_task.max_retries = args.max_retries
        process_single_task.retry_delay = args.retry_delay

        # 加载配置
        config = load_batch_config(args.config)
        logger.info(f"加载配置: {config}")

        # 发现输入文件
        files = discover_input_files(args.input_dir, args.file_pattern, args.recursive)

        # 过滤文件大小
        files = filter_files_by_size(files, args.max_file_size)

        if not files:
            logger.error("没有找到符合条件的输入文件")
            return 1

        # 创建任务
        tasks = create_batch_tasks(files, args.output_dir, config)

        # 过滤已存在的文件
        if args.skip_existing:
            tasks = filter_existing_tasks(tasks)

        logger.info(f"创建 {len(tasks)} 个处理任务")

        # 并行处理
        result = process_tasks_parallel(
            tasks, args.parallel, args.use_processes, args.continue_on_error
        )

        # 保存摘要
        if args.save_summary:
            save_batch_summary(result, args.output_dir)

        # 打印报告
        if args.verbose:
            print_progress_report(result)

        logger.info("批处理完成")
        return 0 if result.failed_files == 0 else 1

    except Exception as e:
        logger.error(f"批处理失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())