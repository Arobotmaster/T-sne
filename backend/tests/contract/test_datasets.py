"""
契约测试：数据集管理API

遵循SDD Constitution的Test-First原则
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

with patch('backend.main.app') as mock_app:
    from backend.main import app

class TestDatasetsAPI:
    """数据集管理API契约测试"""

    def setup_method(self):
        """设置测试环境"""
        self.client = TestClient(app)
        self.test_dataset_id = "test-dataset-123"

    def test_get_dataset_status_success(self):
        """测试成功获取数据集状态"""
        response = self.client.get(f"/api/datasets/{self.test_dataset_id}/status")

        # 期望的状态码和响应结构
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert data["success"] == True
        assert "data" in data
        assert "status" in data["data"]
        assert "progress_percentage" in data["data"]

    def test_get_dataset_status_not_found(self):
        """测试获取不存在的数据集状态"""
        response = self.client.get("/api/datasets/nonexistent-dataset/status")

        # 期望返回404错误
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    def test_get_dataset_status_invalid_id(self):
        """测试使用无效ID获取状态"""
        response = self.client.get("/api/datasets/invalid-id-123/status")

        # 期望返回400错误
        assert response.status_code == 400

    def test_dataset_status_response_schema(self):
        """测试数据集状态响应的JSON Schema"""
        response = self.client.get(f"/api/datasets/{self.test_dataset_id}/status")

        assert response.status_code == 200
        data = response.json()

        # 验证响应结构
        required_fields = ["success", "data"]
        for field in required_fields:
            assert field in data

        # 验证data字段结构
        data_fields = ["status", "progress_percentage", "message", "updated_at"]
        for field in data_fields:
            assert field in data["data"]

        # 验证数据类型
        assert isinstance(data["success"], bool)
        assert isinstance(data["data"]["status"], str)
        assert isinstance(data["data"]["progress_percentage"], (int, float))
        assert isinstance(data["data"]["message"], str)

    def test_dataset_status_values(self):
        """测试数据集状态的可能值"""
        response = self.client.get(f"/api/datasets/{self.test_dataset_id}/status")

        assert response.status_code == 200
        data = response.json()

        # 验证状态值
        valid_statuses = ["pending", "processing", "completed", "failed"]
        assert data["data"]["status"] in valid_statuses

        # 验证进度百分比范围
        progress = data["data"]["progress_percentage"]
        assert 0 <= progress <= 100