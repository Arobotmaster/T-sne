"""
契约测试：数据处理API

遵循SDD Constitution的Test-First原则
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

with patch('backend.main.app') as mock_app:
    from backend.main import app

class TestProcessingAPI:
    """数据处理API契约测试"""

    def setup_method(self):
        """设置测试环境"""
        self.client = TestClient(app)
        self.test_dataset_id = "test-dataset-123"

    def test_start_processing_success(self):
        """测试成功启动数据处理"""
        processing_config = {
            "tsne_config": {
                "perplexity": 30,
                "learning_rate": 200,
                "n_iter": 1000,
                "random_state": 42
            },
            "pca_config": {
                "n_components": 50,
                "whiten": False,
                "random_state": 42
            }
        }

        response = self.client.post(
            f"/api/datasets/{self.test_dataset_id}/process",
            json=processing_config
        )

        # 期望的状态码和响应结构
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert data["success"] == True
        assert "data" in data
        assert "pipeline_id" in data["data"]
        assert "status" in data["data"]

    def test_start_processing_missing_config(self):
        """测试缺少配置参数"""
        response = self.client.post(f"/api/datasets/{self.test_dataset_id}/process")

        # 期望返回422错误
        assert response.status_code == 422

    def test_start_processing_invalid_dataset(self):
        """测试对无效数据集启动处理"""
        processing_config = {
            "tsne_config": {"perplexity": 30},
            "pca_config": {"n_components": 50}
        }

        response = self.client.post(
            "/api/datasets/invalid-dataset/process",
            json=processing_config
        )

        # 期望返回404错误
        assert response.status_code == 404

    def test_start_processing_invalid_tsne_config(self):
        """测试无效的t-SNE配置"""
        invalid_config = {
            "tsne_config": {
                "perplexity": -1,  # 无效值
                "learning_rate": 200
            },
            "pca_config": {"n_components": 50}
        }

        response = self.client.post(
            f"/api/datasets/{self.test_dataset_id}/process",
            json=invalid_config
        )

        # 期望返回400错误
        assert response.status_code == 400

    def test_start_processing_invalid_pca_config(self):
        """测试无效的PCA配置"""
        invalid_config = {
            "tsne_config": {"perplexity": 30},
            "pca_config": {
                "n_components": -1  # 无效值
            }
        }

        response = self.client.post(
            f"/api/datasets/{self.test_dataset_id}/process",
            json=invalid_config
        )

        # 期望返回400错误
        assert response.status_code == 400

    def test_processing_response_schema(self):
        """测试处理响应的JSON Schema"""
        processing_config = {
            "tsne_config": {"perplexity": 30, "learning_rate": 200},
            "pca_config": {"n_components": 50}
        }

        response = self.client.post(
            f"/api/datasets/{self.test_dataset_id}/process",
            json=processing_config
        )

        assert response.status_code == 200
        data = response.json()

        # 验证响应结构
        required_fields = ["success", "data"]
        for field in required_fields:
            assert field in data

        # 验证data字段结构
        data_fields = ["pipeline_id", "status", "estimated_duration", "created_at"]
        for field in data_fields:
            assert field in data["data"]

        # 验证数据类型
        assert isinstance(data["success"], bool)
        assert isinstance(data["data"]["pipeline_id"], str)
        assert isinstance(data["data"]["status"], str)
        assert isinstance(data["data"]["estimated_duration"], (int, float))

    def test_processing_config_validation(self):
        """测试处理配置参数验证"""
        # 测试有效的perplexity范围
        valid_configs = [
            {"tsne_config": {"perplexity": 5}, "pca_config": {"n_components": 2}},
            {"tsne_config": {"perplexity": 50}, "pca_config": {"n_components": 100}},
            {"tsne_config": {"perplexity": 30}, "pca_config": {"n_components": 0.95}}
        ]

        for config in valid_configs:
            response = self.client.post(
                f"/api/datasets/{self.test_dataset_id}/process",
                json=config
            )
            # 应该接受有效配置
            assert response.status_code in [200, 404]  # 404是因为数据集不存在，但配置有效