"""
输入验证和数据清理工具 - 符合SDD Constitution的Library-First原则
提供数据验证、清理、转换功能，支持MOF数据处理需求
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from enum import Enum

from .scientific_logging import ScientificLogger
from .error_handler import ErrorHandler, ErrorCategory


class ValidationError(Exception):
    """验证错误异常"""

    def __init__(self, field: str, message: str, value: Any = None):
        self.field = field
        self.message = message
        self.value = value
        super().__init__(f"Validation failed for field '{field}': {message}")


class DataValidationLevel(Enum):
    """数据验证级别"""

    BASIC = "basic"  # 基本格式验证
    STRICT = "strict"  # 严格数据质量验证
    SCIENTIFIC = "scientific"  # 科学数据验证


class DataValidator:
    """数据验证器 - 支持MOF数据的各种验证需求"""

    def __init__(
        self, validation_level: DataValidationLevel = DataValidationLevel.STRICT
    ):
        self.validation_level = validation_level
        self.logger = ScientificLogger("data_validation", log_dir="logs")
        self.error_handler = ErrorHandler(log_dir="logs")

        # 验证统计
        self.validation_stats = {
            "total_checks": 0,
            "passed_checks": 0,
            "failed_checks": 0,
            "warnings": 0,
        }

    def validate_file_upload(self, file_path: str) -> Dict[str, Any]:
        """验证文件上传"""
        self._increment_stat("total_checks")

        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "file_info": {},
        }

        try:
            # 检查文件存在性
            if not os.path.exists(file_path):
                raise ValidationError("file_path", f"File does not exist: {file_path}")

            # 检查文件大小
            file_size = os.path.getsize(file_path)
            max_size = 100 * 1024 * 1024  # 100MB

            if file_size > max_size:
                error_msg = f"File size {file_size} bytes exceeds maximum allowed size {max_size} bytes"
                validation_result["errors"].append(
                    {"field": "file_size", "message": error_msg, "severity": "error"}
                )
                validation_result["is_valid"] = False
                self._increment_stat("failed_checks")

            # 检查文件格式
            file_ext = Path(file_path).suffix.lower()
            if file_ext != ".csv":
                error_msg = (
                    f"Unsupported file format: {file_ext}. Only CSV files are allowed."
                )
                validation_result["errors"].append(
                    {"field": "file_format", "message": error_msg, "severity": "error"}
                )
                validation_result["is_valid"] = False
                self._increment_stat("failed_checks")

            # 检查文件编码
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    f.read(1024)  # 读取前1KB检测编码
                encoding = "utf-8"
            except UnicodeDecodeError:
                try:
                    with open(file_path, "r", encoding="gbk") as f:
                        f.read(1024)
                    encoding = "gbk"
                    validation_result["warnings"].append(
                        {
                            "field": "encoding",
                            "message": "File uses GBK encoding, UTF-8 is recommended",
                            "severity": "warning",
                        }
                    )
                    self._increment_stat("warnings")
                except:
                    error_msg = "Unable to detect file encoding. Supported encodings: UTF-8, GBK"
                    validation_result["errors"].append(
                        {"field": "encoding", "message": error_msg, "severity": "error"}
                    )
                    validation_result["is_valid"] = False
                    self._increment_stat("failed_checks")

            # 获取文件信息
            validation_result["file_info"] = {
                "file_size": file_size,
                "file_extension": file_ext,
                "encoding": encoding,
                "file_name": Path(file_path).name,
            }

            if validation_result["is_valid"]:
                self._increment_stat("passed_checks")

            # 记录验证结果
            self.logger.log_metric(
                metric_name="file_validation",
                metric_value=1.0 if validation_result["is_valid"] else 0.0,
                metric_unit="success_rate",
                context={
                    "file_size": file_size,
                    "file_extension": file_ext,
                    "validation_level": self.validation_level.value,
                },
            )

            return validation_result

        except Exception as e:
            self.error_handler.handle_error(
                error=e,
                category=ErrorCategory.VALIDATION_ERROR,
                context={
                    "file_path": file_path,
                    "validation_level": self.validation_level.value,
                },
            )
            self._increment_stat("failed_checks")
            raise

    def validate_mof_data_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """验证MOF数据结构"""
        self._increment_stat("total_checks")

        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "data_quality": {},
        }

        try:
            # 检查数据框是否为空
            if df.empty:
                error_msg = "DataFrame is empty"
                validation_result["errors"].append(
                    {"field": "dataframe", "message": error_msg, "severity": "error"}
                )
                validation_result["is_valid"] = False
                self._increment_stat("failed_checks")
                return validation_result

            # 检查最小行数
            if len(df) < 10:
                warning_msg = f"Dataset has only {len(df)} rows, which may be insufficient for meaningful analysis"
                validation_result["warnings"].append(
                    {
                        "field": "row_count",
                        "message": warning_msg,
                        "severity": "warning",
                    }
                )
                self._increment_stat("warnings")

            # 检查列名
            required_columns = ["mofid", "category"]  # 基本必需列
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                error_msg = f"Missing required columns: {missing_columns}"
                validation_result["errors"].append(
                    {"field": "columns", "message": error_msg, "severity": "error"}
                )
                validation_result["is_valid"] = False
                self._increment_stat("failed_checks")

            # 检查数值列
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) < 2:
                error_msg = f"At least 2 numeric columns required for analysis, found {len(numeric_columns)}"
                validation_result["errors"].append(
                    {
                        "field": "numeric_columns",
                        "message": error_msg,
                        "severity": "error",
                    }
                )
                validation_result["is_valid"] = False
                self._increment_stat("failed_checks")

            # 检查数据质量
            quality_metrics = self._calculate_data_quality(df)
            validation_result["data_quality"] = quality_metrics

            # 严格模式下的额外检查
            if self.validation_level in [
                DataValidationLevel.STRICT,
                DataValidationLevel.SCIENTIFIC,
            ]:
                self._strict_validation(df, validation_result)

            if validation_result["is_valid"]:
                self._increment_stat("passed_checks")

            # 记录验证结果
            self.logger.log_metric(
                metric_name="data_structure_validation",
                metric_value=1.0 if validation_result["is_valid"] else 0.0,
                metric_unit="success_rate",
                context={
                    "rows": len(df),
                    "columns": len(df.columns),
                    "numeric_columns": len(numeric_columns),
                    "validation_level": self.validation_level.value,
                },
            )

            return validation_result

        except Exception as e:
            self.error_handler.handle_error(
                error=e,
                category=ErrorCategory.VALIDATION_ERROR,
                context={"data_shape": df.shape if hasattr(df, "shape") else None},
            )
            self._increment_stat("failed_checks")
            raise

    def validate_tsne_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """验证t-SNE参数"""
        self._increment_stat("total_checks")

        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "normalized_params": {},
        }

        try:
            # Perplexity验证
            if "perplexity" in params:
                perplexity = params["perplexity"]
                if (
                    not isinstance(perplexity, (int, float))
                    or perplexity < 5
                    or perplexity > 50
                ):
                    error_msg = f"Perplexity must be between 5 and 50, got {perplexity}"
                    validation_result["errors"].append(
                        {
                            "field": "perplexity",
                            "message": error_msg,
                            "severity": "error",
                        }
                    )
                    validation_result["is_valid"] = False
                else:
                    validation_result["normalized_params"]["perplexity"] = float(
                        perplexity
                    )

            # 学习率验证
            if "learning_rate" in params:
                learning_rate = params["learning_rate"]
                if (
                    not isinstance(learning_rate, (int, float))
                    or learning_rate < 10
                    or learning_rate > 1000
                ):
                    error_msg = f"Learning rate must be between 10 and 1000, got {learning_rate}"
                    validation_result["errors"].append(
                        {
                            "field": "learning_rate",
                            "message": error_msg,
                            "severity": "error",
                        }
                    )
                    validation_result["is_valid"] = False
                else:
                    validation_result["normalized_params"]["learning_rate"] = float(
                        learning_rate
                    )

            # 迭代次数验证
            if "n_iter" in params:
                n_iter = params["n_iter"]
                if not isinstance(n_iter, int) or n_iter < 250 or n_iter > 5000:
                    error_msg = f"n_iter must be between 250 and 5000, got {n_iter}"
                    validation_result["errors"].append(
                        {"field": "n_iter", "message": error_msg, "severity": "error"}
                    )
                    validation_result["is_valid"] = False
                else:
                    validation_result["normalized_params"]["n_iter"] = int(n_iter)

            # 随机种子验证
            if "random_state" in params:
                random_state = params["random_state"]
                if random_state is not None:
                    if not isinstance(random_state, int):
                        error_msg = f"random_state must be an integer or None, got {type(random_state)}"
                        validation_result["errors"].append(
                            {
                                "field": "random_state",
                                "message": error_msg,
                                "severity": "error",
                            }
                        )
                        validation_result["is_valid"] = False
                    else:
                        validation_result["normalized_params"]["random_state"] = int(
                            random_state
                        )

            # 距离度量验证
            if "metric" in params:
                metric = params["metric"]
                valid_metrics = ["euclidean", "manhattan", "chebyshev", "minkowski"]
                if metric not in valid_metrics:
                    error_msg = f"metric must be one of {valid_metrics}, got {metric}"
                    validation_result["errors"].append(
                        {"field": "metric", "message": error_msg, "severity": "error"}
                    )
                    validation_result["is_valid"] = False
                else:
                    validation_result["normalized_params"]["metric"] = metric

            if validation_result["is_valid"]:
                self._increment_stat("passed_checks")

            return validation_result

        except Exception as e:
            self.error_handler.handle_error(
                error=e,
                category=ErrorCategory.VALIDATION_ERROR,
                context={"parameters": params},
            )
            self._increment_stat("failed_checks")
            raise

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理DataFrame数据"""
        self._increment_stat("total_checks")

        try:
            original_shape = df.shape
            cleaned_df = df.copy()

            # 移除完全重复的行
            cleaned_df = cleaned_df.drop_duplicates()
            duplicates_removed = original_shape[0] - cleaned_df.shape[0]

            # 处理缺失值
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype in ["object", "category"]:
                    # 分类列用众数填充
                    mode_value = cleaned_df[col].mode()
                    if len(mode_value) > 0:
                        cleaned_df[col] = cleaned_df[col].fillna(mode_value[0])
                else:
                    # 数值列用中位数填充
                    median_value = cleaned_df[col].median()
                    cleaned_df[col] = cleaned_df[col].fillna(median_value)

            # 清理字符串列
            for col in cleaned_df.select_dtypes(include=["object"]).columns:
                cleaned_df[col] = cleaned_df[col].astype(str).str.strip()

            # 记录清理结果
            cleaning_stats = {
                "original_shape": original_shape,
                "cleaned_shape": cleaned_df.shape,
                "duplicates_removed": duplicates_removed,
                "missing_values_filled": (
                    df.isnull().sum().sum() - cleaned_df.isnull().sum().sum()
                ),
            }

            self.logger.log_metric(
                metric_name="data_cleaning",
                metric_value=duplicates_removed,
                metric_unit="rows_removed",
                context=cleaning_stats,
            )

            self._increment_stat("passed_checks")
            return cleaned_df

        except Exception as e:
            self.error_handler.handle_error(
                error=e,
                category=ErrorCategory.DATA_ERROR,
                context={"data_shape": df.shape if hasattr(df, "shape") else None},
            )
            self._increment_stat("failed_checks")
            raise

    def _calculate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算数据质量指标"""
        quality_metrics = {
            "completeness": 1.0
            - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])),
            "uniqueness": (
                1.0 - (df.duplicated().sum() / len(df)) if len(df) > 0 else 1.0
            ),
            "consistency": self._calculate_consistency_score(df),
            "accuracy": self._calculate_accuracy_score(df),
        }

        return quality_metrics

    def _calculate_consistency_score(self, df: pd.DataFrame) -> float:
        """计算数据一致性分数"""
        # 检查数据类型一致性
        consistency_score = 1.0

        # 检查数值列的一致性
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() / len(df) > 0.3:  # 缺失值超过30%
                consistency_score *= 0.9

        # 检查分类列的一致性
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.8:  # 唯一值过多，可能是数据错误
                consistency_score *= 0.95

        return consistency_score

    def _calculate_accuracy_score(self, df: pd.DataFrame) -> float:
        """计算数据准确性分数"""
        accuracy_score = 1.0

        # 检查数值列的合理性
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # 检查异常值（使用IQR方法）
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_ratio = outliers / len(df)

            if outlier_ratio > 0.1:  # 异常值超过10%
                accuracy_score *= 0.9

        return accuracy_score

    def _strict_validation(self, df: pd.DataFrame, validation_result: Dict[str, Any]):
        """严格模式验证"""
        # 检查mofid格式
        if "mofid" in df.columns:
            invalid_mofids = (
                df["mofid"]
                .apply(lambda x: not isinstance(x, str) or len(str(x)) < 3)
                .sum()
            )
            if invalid_mofids > 0:
                warning_msg = f"Found {invalid_mofids} invalid mofid values"
                validation_result["warnings"].append(
                    {"field": "mofid", "message": warning_msg, "severity": "warning"}
                )
                self._increment_stat("warnings")

        # 检查分类列的唯一值数量
        if "category" in df.columns:
            unique_categories = df["category"].nunique()
            if unique_categories > 10:
                warning_msg = f"Found {unique_categories} unique categories, which may be too many for effective visualization"
                validation_result["warnings"].append(
                    {"field": "category", "message": warning_msg, "severity": "warning"}
                )
                self._increment_stat("warnings")

        # 科学模式下的额外检查
        if self.validation_level == DataValidationLevel.SCIENTIFIC:
            # 检查数值列的相关性
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.95:
                            high_corr_pairs.append(
                                (corr_matrix.columns[i], corr_matrix.columns[j])
                            )

                if high_corr_pairs:
                    warning_msg = (
                        f"Found highly correlated feature pairs: {high_corr_pairs}"
                    )
                    validation_result["warnings"].append(
                        {
                            "field": "feature_correlation",
                            "message": warning_msg,
                            "severity": "warning",
                        }
                    )
                    self._increment_stat("warnings")

    def _increment_stat(self, stat_name: str):
        """递增统计计数"""
        if stat_name in self.validation_stats:
            self.validation_stats[stat_name] += 1

    def get_validation_statistics(self) -> Dict[str, Any]:
        """获取验证统计信息"""
        total = self.validation_stats["total_checks"]
        passed = self.validation_stats["passed_checks"]

        return {
            **self.validation_stats,
            "success_rate": passed / total if total > 0 else 0.0,
            "validation_level": self.validation_level.value,
        }


# 便捷验证函数
def validate_file_safely(
    file_path: str, validator: Optional[DataValidator] = None
) -> Dict[str, Any]:
    """安全验证文件的便捷函数"""
    if validator is None:
        validator = DataValidator()

    return validator.validate_file_upload(file_path)


def validate_dataframe_safely(
    df: pd.DataFrame, validator: Optional[DataValidator] = None
) -> Dict[str, Any]:
    """安全验证DataFrame的便捷函数"""
    if validator is None:
        validator = DataValidator()

    return validator.validate_mof_data_structure(df)


def clean_dataframe_safely(
    df: pd.DataFrame, validator: Optional[DataValidator] = None
) -> pd.DataFrame:
    """安全清理DataFrame的便捷函数"""
    if validator is None:
        validator = DataValidator()

    return validator.clean_dataframe(df)
