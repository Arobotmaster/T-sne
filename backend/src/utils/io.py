"""
IO工具函数 - 文件读写和数据格式转换

符合SDD Constitution的工具函数，
提供统一的文件读写接口。
"""

import pandas as pd
import numpy as np
import json
import csv
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


def load_csv(file_path: str, encoding: str = 'utf-8') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    加载CSV文件

    Args:
        file_path: CSV文件路径
        encoding: 文件编码

    Returns:
        Tuple: (数据DataFrame, 元数据字典)
    """
    try:
        logger.info(f"加载CSV文件: {file_path}")

        # 检测文件编码
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read(1024)
            encoding = 'utf-8'
        except UnicodeDecodeError:
            encoding = 'gbk'
            logger.info(f"使用编码: {encoding}")

        # 读取CSV文件
        data = pd.read_csv(file_path, encoding=encoding)

        # 生成元数据
        metadata = {
            'file_path': file_path,
            'encoding': encoding,
            'n_samples': len(data),
            'n_features': len(data.columns),
            'column_names': data.columns.tolist(),
            'data_types': data.dtypes.astype(str).to_dict(),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
            'missing_values': data.isnull().sum().to_dict()
        }

        logger.info(f"成功加载数据: {metadata['n_samples']} 样本, {metadata['n_features']} 特征")

        return data, metadata

    except Exception as e:
        logger.error(f"加载CSV文件失败: {e}")
        raise ValueError(f"无法加载CSV文件 {file_path}: {e}")


def save_csv(data: Union[pd.DataFrame, np.ndarray], file_path: str, encoding: str = 'utf-8'):
    """
    保存CSV文件

    Args:
        data: 要保存的数据 (DataFrame或numpy数组)
        file_path: 输出文件路径
        encoding: 文件编码
    """
    try:
        logger.info(f"保存CSV文件: {file_path}")

        # 创建输出目录
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 转换为DataFrame
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            data = pd.DataFrame(data)

        # 保存CSV
        data.to_csv(file_path, index=False, encoding=encoding)

        logger.info(f"成功保存CSV文件: {file_path} ({len(data)} 行, {len(data.columns)} 列)")

    except Exception as e:
        logger.error(f"保存CSV文件失败: {e}")
        raise ValueError(f"无法保存CSV文件 {file_path}: {e}")


def load_json(file_path: str, encoding: str = 'utf-8') -> Dict[str, Any]:
    """
    加载JSON文件

    Args:
        file_path: JSON文件路径
        encoding: 文件编码

    Returns:
        JSON数据字典
    """
    try:
        logger.info(f"加载JSON文件: {file_path}")

        with open(file_path, 'r', encoding=encoding) as f:
            data = json.load(f)

        logger.info(f"成功加载JSON文件: {file_path}")
        return data

    except Exception as e:
        logger.error(f"加载JSON文件失败: {e}")
        raise ValueError(f"无法加载JSON文件 {file_path}: {e}")


def save_json(data: Dict[str, Any], file_path: str, encoding: str = 'utf-8', indent: int = 2):
    """
    保存JSON文件

    Args:
        data: 要保存的数据字典
        file_path: 输出文件路径
        encoding: 文件编码
        indent: JSON缩进格式
    """
    try:
        logger.info(f"保存JSON文件: {file_path}")

        # 创建输出目录
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存JSON
        with open(file_path, 'w', encoding=encoding) as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, cls=NumpyJSONEncoder)

        logger.info(f"成功保存JSON文件: {file_path}")

    except Exception as e:
        logger.error(f"保存JSON文件失败: {e}")
        raise ValueError(f"无法保存JSON文件 {file_path}: {e}")


class NumpyJSONEncoder(json.JSONEncoder):
    """支持numpy数据类型的JSON编码器"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return super().default(obj)


def load_data(file_path: str) -> Tuple[Any, Dict[str, Any]]:
    """
    根据文件扩展名自动加载数据

    Args:
        file_path: 数据文件路径

    Returns:
        Tuple: (数据, 元数据)
    """
    file_ext = Path(file_path).suffix.lower()

    if file_ext == '.csv':
        return load_csv(file_path)
    elif file_ext == '.json':
        data = load_json(file_path)
        return data, {'file_path': file_path, 'format': 'json'}
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}")


def save_data(data: Any, file_path: str, **kwargs):
    """
    根据文件扩展名自动保存数据

    Args:
        data: 要保存的数据
        file_path: 输出文件路径
        **kwargs: 保存参数
    """
    file_ext = Path(file_path).suffix.lower()

    if file_ext == '.csv':
        save_csv(data, file_path, **kwargs)
    elif file_ext == '.json':
        save_json(data, file_path, **kwargs)
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}")


def validate_file_path(file_path: str, must_exist: bool = True) -> bool:
    """
    验证文件路径

    Args:
        file_path: 文件路径
        must_exist: 文件是否必须存在

    Returns:
        验证结果
    """
    path = Path(file_path)

    if must_exist:
        return path.exists() and path.is_file()
    else:
        # 检查父目录是否存在
        return path.parent.exists()


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    获取文件信息

    Args:
        file_path: 文件路径

    Returns:
        文件信息字典
    """
    path = Path(file_path)

    if not path.exists():
        return {'exists': False}

    stat = path.stat()

    return {
        'exists': True,
        'size_bytes': stat.st_size,
        'size_mb': stat.st_size / (1024 * 1024),
        'modified_time': stat.st_mtime,
        'is_file': path.is_file(),
        'is_dir': path.is_dir(),
        'extension': path.suffix.lower(),
        'name': path.name,
        'stem': path.stem
    }


def create_sample_csv(output_path: str, n_samples: int = 100, n_features: int = 10):
    """
    创建示例CSV文件用于测试

    Args:
        output_path: 输出文件路径
        n_samples: 样本数量
        n_features: 特征数量
    """
    try:
        logger.info(f"创建示例CSV文件: {output_path}")

        np.random.seed(42)

        # 生成随机数据
        data = np.random.randn(n_samples, n_features)

        # 添加一些有意义的特征名
        feature_names = [f'feature_{i+1}' for i in range(n_features)]
        df = pd.DataFrame(data, columns=feature_names)

        # 添加一些分类特征
        df['category'] = np.random.choice(['A', 'B', 'C'], n_samples)

        # 保存CSV
        save_csv(df, output_path)

        logger.info(f"示例CSV文件创建完成: {output_path}")

    except Exception as e:
        logger.error(f"创建示例CSV文件失败: {e}")
        raise


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    清理DataFrame数据

    Args:
        df: 输入DataFrame

    Returns:
        清理后的DataFrame
    """
    logger.info("清理DataFrame数据")

    # 移除全为空的行
    df = df.dropna(how='all')

    # 移除全为空的列
    df = df.dropna(axis=1, how='all')

    # 重置索引
    df = df.reset_index(drop=True)

    logger.info(f"数据清理完成: {df.shape}")
    return df


def merge_dataframes(dfs: list, merge_method: str = 'concat') -> pd.DataFrame:
    """
    合并多个DataFrame

    Args:
        dfs: DataFrame列表
        merge_method: 合并方法 ('concat', 'merge', 'join')

    Returns:
        合并后的DataFrame
    """
    if not dfs:
        raise ValueError("DataFrame列表不能为空")

    if len(dfs) == 1:
        return dfs[0]

    logger.info(f"合并 {len(dfs)} 个DataFrame，方法: {merge_method}")

    try:
        if merge_method == 'concat':
            result = pd.concat(dfs, ignore_index=True)
        elif merge_method == 'merge':
            result = dfs[0]
            for df in dfs[1:]:
                result = pd.merge(result, df)
        elif merge_method == 'join':
            result = dfs[0]
            for df in dfs[1:]:
                result = result.join(df)
        else:
            raise ValueError(f"不支持的合并方法: {merge_method}")

        logger.info(f"DataFrame合并完成: {result.shape}")
        return result

    except Exception as e:
        logger.error(f"DataFrame合并失败: {e}")
        raise


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    # 创建示例数据
    create_sample_csv('test_data.csv', 50, 5)

    # 测试加载
    data, metadata = load_csv('test_data.csv')
    print(f"加载数据: {metadata}")

    # 测试保存
    save_json(metadata, 'test_metadata.json')

    print("IO工具函数测试完成")