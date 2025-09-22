"""
数据验证器测试 - 符合SDD Constitution的Test-First原则
测试DataValidator的各种数据验证功能
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os
from pathlib import Path

from src.utils.validation import (
    DataValidator,
    DataValidationLevel,
    ValidationError,
)


class TestDataValidator:
    """数据验证器测试类"""

    def setup_method(self):
        """测试前的设置"""
        self.validator = DataValidator(DataValidationLevel.STRICT)

    def test_initialization(self):
        """测试初始化"""
        assert self.validator.validation_level == DataValidationLevel.STRICT
        assert hasattr(self.validator, 'logger')
        assert hasattr(self.validator, 'error_handler')
        assert self.validator.validation_stats["total_checks"] == 0

    def test_validation_error_exception(self):
        """测试ValidationError异常"""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("test_field", "Test validation message", "test_value")

        assert exc_info.value.field == "test_field"
        assert exc_info.value.message == "Test validation message"
        assert exc_info.value.value == "test_value"

    def create_test_csv(self, content, filename="test.csv"):
        """创建测试CSV文件的辅助方法"""
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, filename)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return file_path

    def test_validate_file_upload_success(self):
        """测试文件上传验证成功"""
        # 创建有效的测试CSV
        csv_content = """mofid,category,feature1,feature2,feature3
1,A,1.0,2.0,3.0
2,B,4.0,5.0,6.0
3,A,7.0,8.0,9.0"""

        file_path = self.create_test_csv(csv_content)

        result = self.validator.validate_file_upload(file_path)

        assert result["is_valid"] is True
        assert result["file_size"] > 0
        assert result["encoding"] == "utf-8"
        assert result["format"] == "csv"

    def test_validate_file_upload_too_large(self):
        """测试文件过大验证"""
        # 创建大文件内容
        large_content = "mofid,category,feature1\n" + "1,A,1.0\n" * 1000000

        file_path = self.create_test_csv(large_content)

        result = self.validator.validate_file_upload(file_path)

        assert result["is_valid"] is False
        assert "size_exceeded" in result["errors"]

    def test_validate_file_upload_wrong_format(self):
        """测试文件格式错误验证"""
        # 创建非CSV文件
        file_path = self.create_test_csv("not a csv file", "test.txt")

        result = self.validator.validate_file_upload(file_path)

        assert result["is_valid"] is False
        assert "invalid_format" in result["errors"]

    def test_validate_mof_data_structure_success(self):
        """测试MOF数据结构验证成功"""
        # 创建有效的MOF数据
        df = pd.DataFrame({
            'mofid': [1, 2, 3, 4, 5],
            'category': ['A', 'B', 'A', 'B', 'C'],
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'feature3': [100.0, 200.0, 300.0, 400.0, 500.0]
        })

        result = self.validator.validate_mof_data_structure(df)

        assert result["is_valid"] is True
        assert result["quality_score"] >= 0.8
        assert "missing_columns" not in result["errors"]

    def test_validate_mof_data_structure_missing_columns(self):
        """测试MOF数据结构缺失必需列"""
        # 缺少category列的数据
        df = pd.DataFrame({
            'mofid': [1, 2, 3],
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [10.0, 20.0, 30.0]
        })

        result = self.validator.validate_mof_data_structure(df)

        assert result["is_valid"] is False
        assert "missing_columns" in result["errors"]
        assert result["quality_score"] < 0.8

    def test_validate_mof_data_structure_insufficient_samples(self):
        """测试样本数量不足"""
        # 样本数量过少
        df = pd.DataFrame({
            'mofid': [1],
            'category': ['A'],
            'feature1': [1.0]
        })

        result = self.validator.validate_mof_data_structure(df)

        assert result["is_valid"] is False
        assert "insufficient_samples" in result["errors"]

    def test_validate_mof_data_structure_insufficient_features(self):
        """测试特征数量不足"""
        # 特征数量过少
        df = pd.DataFrame({
            'mofid': [1, 2, 3],
            'category': ['A', 'B', 'A']
        })

        result = self.validator.validate_mof_data_structure(df)

        assert result["is_valid"] is False
        assert "insufficient_features" in result["errors"]

    def test_detect_column_types(self):
        """测试列类型检测"""
        df = pd.DataFrame({
            'mofid': [1, 2, 3],
            'category': ['A', 'B', 'C'],
            'numeric_col': [1.0, 2.0, 3.0],
            'text_col': ['text1', 'text2', 'text3']
        })

        result = self.validator._detect_column_types(df)

        assert result['mofid'] == 'identifier'
        assert result['category'] == 'categorical'
        assert result['numeric_col'] == 'numeric'
        assert result['text_col'] == 'text'

    def test_check_required_columns(self):
        """检查必需列"""
        # 完整的必需列
        df_complete = pd.DataFrame({
            'mofid': [1, 2, 3],
            'category': ['A', 'B', 'C'],
            'feature1': [1.0, 2.0, 3.0]
        })

        result_complete = self.validator._check_required_columns(df_complete)
        assert len(result_complete) == 0

        # 缺少必需列
        df_missing = pd.DataFrame({
            'mofid': [1, 2, 3],
            'feature1': [1.0, 2.0, 3.0]
        })

        result_missing = self.validator._check_required_columns(df_missing)
        assert 'category' in result_missing

    def test_check_duplicates(self):
        """测试重复数据检查"""
        # 无重复数据
        df_no_duplicates = pd.DataFrame({
            'mofid': [1, 2, 3],
            'category': ['A', 'B', 'C']
        })

        result_no_duplicates = self.validator._check_duplicates(df_no_duplicates)
        assert result_no_duplicates["has_duplicates"] is False
        assert result_no_duplicates["duplicate_count"] == 0

        # 有重复数据
        df_with_duplicates = pd.DataFrame({
            'mofid': [1, 1, 2, 2],
            'category': ['A', 'A', 'B', 'B']
        })

        result_with_duplicates = self.validator._check_duplicates(df_with_duplicates)
        assert result_with_duplicates["has_duplicates"] is True
        assert result_with_duplicates["duplicate_count"] == 2

    def test_check_outliers(self):
        """测试异常值检测"""
        # 包含异常值的数据
        df_with_outliers = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 1000.0],  # 1000.0是异常值
            'feature2': [10.0, 20.0, 30.0, 40.0]
        })

        result = self.validator._check_outliers(df_with_outliers)

        assert "outliers" in result
        assert len(result["outliers"]) > 0

    def test_calculate_quality_score(self):
        """测试质量分数计算"""
        validation_result = {
            "missing_columns": [],
            "duplicates": {"has_duplicates": False},
            "outliers": {"outlier_count": 0},
            "sample_count": 100,
            "feature_count": 10
        }

        score = self.validator._calculate_quality_score(validation_result)

        assert 0 <= score <= 1
        assert score > 0.8  # 高质量数据

    def test_clean_missing_values_drop_strategy(self):
        """测试缺失值清理 - 删除策略"""
        df = pd.DataFrame({
            'mofid': [1, 2, 3, 4],
            'category': ['A', 'B', None, 'C'],
            'feature1': [1.0, None, 3.0, 4.0]
        })

        result = self.validator.clean_missing_values(df, strategy="drop")

        assert len(result) < len(df)  # 应该删除了包含缺失值的行

    def test_clean_missing_values_fill_strategy(self):
        """测试缺失值清理 - 填充策略"""
        df = pd.DataFrame({
            'mofid': [1, 2, 3, 4],
            'category': ['A', 'B', None, 'C'],
            'feature1': [1.0, None, 3.0, 4.0]
        })

        result = self.validator.clean_missing_values(df, strategy="fill")

        assert len(result) == len(df)  # 行数不变
        assert result.isnull().sum().sum() == 0  # 无缺失值

    def test_normalize_data_standard(self):
        """测试数据标准化 - 标准化方法"""
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': [10.0, 20.0, 30.0, 40.0],
            'category': ['A', 'B', 'A', 'B']  # 非数值列应该保持不变
        })

        result = self.validator.normalize_data(df, method="standard")

        # 验证数值列被标准化
        assert abs(result['feature1'].mean()) < 1e-10  # 均值接近0
        assert abs(result['feature1'].std() - 1.0) < 1e-10  # 标准差接近1

        # 验证非数值列保持不变
        assert result['category'].equals(df['category'])

    def test_normalize_data_minmax(self):
        """测试数据标准化 - 最小-最大标准化"""
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': [10.0, 20.0, 30.0, 40.0]
        })

        result = self.validator.normalize_data(df, method="minmax")

        # 验证数据被缩放到[0,1]区间
        assert result['feature1'].min() >= 0
        assert result['feature1'].max() <= 1

    def test_remove_duplicates(self):
        """测试重复数据删除"""
        df = pd.DataFrame({
            'mofid': [1, 1, 2, 2, 3],
            'category': ['A', 'A', 'B', 'B', 'C']
        })

        result = self.validator.remove_duplicates(df)

        assert len(result) < len(df)
        assert len(result['mofid'].unique()) == len(result)

    def test_preprocess_pipeline_high_quality(self):
        """测试数据预处理流水线 - 高质量数据"""
        df = pd.DataFrame({
            'mofid': list(range(100)),
            'category': ['A', 'B'] * 50,
            'feature1': np.random.random(100),
            'feature2': np.random.random(100)
        })

        result = self.validator.preprocess_pipeline(df)

        # 高质量数据应该返回原样
        assert result.equals(df)

    def test_preprocess_pipeline_low_quality(self):
        """测试数据预处理流水线 - 低质量数据"""
        df = pd.DataFrame({
            'mofid': [1, 1, 2, 2],  # 重复数据
            'category': ['A', 'A', None, 'B'],  # 缺失值
            'feature1': [1.0, 2.0, 1000.0, 4.0]  # 异常值
        })

        original_shape = df.shape
        result = self.validator.preprocess_pipeline(df)

        # 低质量数据应该被清理
        assert result.shape[0] <= original_shape[0]

    def test_validation_statistics(self):
        """测试验证统计"""
        initial_checks = self.validator.validation_stats["total_checks"]

        # 执行一些验证操作
        df = pd.DataFrame({'mofid': [1, 2, 3], 'category': ['A', 'B', 'C']})
        self.validator.validate_mof_data_structure(df)

        assert self.validator.validation_stats["total_checks"] > initial_checks

    @patch('backend.src.utils.validation.ScientificLogger')
    def test_logging_integration(self, mock_logger_class):
        """测试日志记录集成"""
        mock_logger = Mock()
        mock_logger_class.return_value = mock_logger

        validator = DataValidator(DataValidationLevel.BASIC)

        # 执行验证操作
        df = pd.DataFrame({'mofid': [1, 2, 3], 'category': ['A', 'B', 'C']})
        validator.validate_mof_data_structure(df)

        # 验证日志记录被调用
        mock_logger.log_info.assert_called()

    def test_different_validation_levels(self):
        """测试不同验证级别"""
        # 测试基本级别
        basic_validator = DataValidator(DataValidationLevel.BASIC)
        df = pd.DataFrame({'mofid': [1, 2, 3], 'category': ['A', 'B', 'C']})

        basic_result = basic_validator.validate_mof_data_structure(df)

        # 测试科学级别
        scientific_validator = DataValidator(DataValidationLevel.SCIENTIFIC)
        scientific_result = scientific_validator.validate_mof_data_structure(df)

        # 科学级别的验证应该更严格
        assert hasattr(scientific_validator, 'validation_level')
        assert scientific_validator.validation_level == DataValidationLevel.SCIENTIFIC