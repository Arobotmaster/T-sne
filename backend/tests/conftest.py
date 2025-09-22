"""
pytest配置和共享fixture

提供测试环境的统一配置和常用fixture
"""

import sys
import os
from pathlib import Path
import pytest
import tempfile
import shutil
from unittest.mock import Mock

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import settings, Settings
from src.config.logging_config import setup_logging


@pytest.fixture(scope="session")
def test_settings():
    """测试配置fixture"""
    # 创建临时目录用于测试
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 创建测试配置
        test_config = {
            "APP_NAME": "MOF数据t-SNE可视化测试",
            "DEBUG": True,
            "ENVIRONMENT": "testing",
            "DATABASE_URL": f"sqlite:///{temp_path}/test.db",
            "REDIS_URL": "redis://localhost:6379/1",
            "LOG_LEVEL": "DEBUG",
            "LOG_FILE": f"{temp_path}/test.log",
            "UPLOAD_DIRECTORY": f"{temp_path}/uploads",
            "TEMP_DIRECTORY": f"{temp_path}/temp",
            "EXPORT_DIRECTORY": f"{temp_path}/exports",
            "MAX_UPLOAD_SIZE": 10485760,  # 10MB for testing
            "ALLOWED_ORIGINS": ["http://localhost:3000", "http://localhost:8000"],
            "MAX_CONCURRENT_PROCESSES": 2,
            "PROCESSING_TIMEOUT": 300,
        }

        # 创建测试专用设置
        settings_dict = Settings(**test_config)

        # 确保目录存在
        settings_dict.ensure_directories()

        yield settings_dict


@pytest.fixture(scope="session")
def test_logger(test_settings):
    """测试日志fixture"""
    return setup_logging(test_settings.LOG_LEVEL)


@pytest.fixture
def sample_csv_data():
    """示例CSV数据fixture"""
    return """sample_id,feature_1,feature_2,feature_3,feature_4,category
sample_001,1.2,3.4,5.6,7.8,MOF_A
sample_002,2.1,4.3,6.5,8.7,MOF_B
sample_003,1.5,3.2,5.1,7.2,MOF_A
sample_004,2.8,4.1,6.8,8.1,MOF_C
sample_005,1.1,3.7,5.9,7.5,MOF_B
sample_006,2.3,4.5,6.2,8.9,MOF_A
sample_007,1.8,3.1,5.4,7.1,MOF_C
sample_008,2.6,4.8,6.7,8.3,MOF_B"""


@pytest.fixture
def temp_csv_file(sample_csv_data):
    """临时CSV文件fixture"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(sample_csv_data)
        csv_path = f.name

    yield csv_path

    # 清理
    os.unlink(csv_path)


@pytest.fixture
def mock_upload_file():
    """模拟上传文件fixture"""
    from io import BytesIO

    content = b"sample_id,feature_1,feature_2,category\nsample_001,1.2,3.4,MOF_A\n"
    mock_file = Mock()
    mock_file.filename = "test_data.csv"
    mock_file.file = BytesIO(content)
    mock_file.size = len(content)
    mock_file.content_type = "text/csv"

    return mock_file


@pytest.fixture
def sample_numpy_data():
    """示例NumPy数据fixture"""
    import numpy as np

    np.random.seed(42)  # 确保可重现性
    return np.random.rand(20, 10)  # 20个样本，10个特征


@pytest.fixture
def sample_metadata():
    """示例元数据fixture"""
    return {
        'n_samples': 20,
        'n_features': 10,
        'column_names': [f'feature_{i}' for i in range(10)],
        'category_column': 'category',
        'data_quality_score': 0.95,
        'missing_values': 0,
        'encoding': 'utf-8'
    }


@pytest.fixture
def sample_tsne_config():
    """示例t-SNE配置fixture"""
    return {
        'perplexity': 30,
        'n_components': 2,
        'random_state': 42,
        'learning_rate': 200,
        'n_iter': 1000,
        'early_exaggeration': 12,
        'metric': 'euclidean'
    }


@pytest.fixture
def sample_pca_config():
    """示例PCA配置fixture"""
    return {
        'n_components': 10,
        'random_state': 42,
        'whiten': False,
        'svd_solver': 'auto'
    }


@pytest.fixture
def sample_processing_result():
    """示例处理结果fixture"""
    import numpy as np

    np.random.seed(42)
    coordinates = np.random.rand(20, 2)

    return {
        'pipeline_id': 'test_pipeline_001',
        'coordinates': coordinates.tolist(),
        'config': {
            'preprocessing': {
                'missing_value_strategy': 'mean',
                'scaling_method': 'standard'
            },
            'pca': {'n_components': 10},
            'tsne': {'perplexity': 30}
        },
        'metadata': {
            'original_shape': [20, 10],
            'pca_shape': [20, 10],
            'final_shape': [20, 2],
            'processing_time': 1.23,
            'explained_variance': [0.45, 0.32, 0.15, 0.08]
        }
    }


@pytest.fixture
def test_client():
    """FastAPI测试客户端fixture"""
    # 这个fixture将在FastAPI应用实现后工作
    # 现在返回一个Mock对象避免测试失败
    return Mock()


# 自定义标记
def pytest_configure(config):
    """配置pytest自定义标记"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "contract: marks tests as contract tests"
    )


# 测试运行前和运行后的钩子
def pytest_sessionstart(session):
    """测试会话开始时的钩子"""
    print(f"\n=== MOF t-SNE可视化测试会话开始 ===")
    print(f"项目根目录: {project_root}")
    print(f"Python路径: {sys.path[:3]}...")  # 只显示前3个路径


def pytest_sessionfinish(session, exitstatus):
    """测试会束时的钩子"""
    print(f"\n=== 测试会话结束，退出码: {exitstatus} ===")


def pytest_runtest_setup(item):
    """每个测试运行前的设置"""
    # 为慢速测试添加标记
    if "slow" in item.keywords:
        print(f"\n[慢速测试] {item.name}")


def pytest_runtest_teardown(item, nextitem):
    """每个测试运行后的清理"""
    pass