"""
契约测试：处理流水线API

遵循SDD Constitution的Test-First原则
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

with patch('backend.main.app') as mock_app:
    from backend.main import app

class TestPipelinesAPI:
    """处理流水线API契约测试"""

    def setup_method(self):
        """设置测试环境"""
        self.client = TestClient(app)
        self.test_pipeline_id = "test-pipeline-123"

    def test_get_pipeline_progress_success(self):
        """测试成功获取流水线进度"""
        response = self.client.get(f"/api/pipelines/{self.test_pipeline_id}/progress")

        # 期望的状态码和响应结构
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert data["success"] == True
        assert "data" in data
        assert "progress_percentage" in data["data"]
        assert "current_step" in data["data"]

    def test_get_pipeline_progress_not_found(self):
        """测试获取不存在的流水线进度"""
        response = self.client.get("/api/pipelines/nonexistent-pipeline/progress")

        # 期望返回404错误
        assert response.status_code == 404

    def test_pipeline_progress_response_schema(self):
        """测试流水线进度响应的JSON Schema"""
        response = self.client.get(f"/api/pipelines/{self.test_pipeline_id}/progress")

        assert response.status_code == 200
        data = response.json()

        # 验证响应结构
        required_fields = ["success", "data"]
        for field in required_fields:
            assert field in data

        # 验证data字段结构
        data_fields = ["progress_percentage", "current_step", "steps_completed", "total_steps", "eta"]
        for field in data_fields:
            assert field in data["data"]

        # 验证数据类型
        assert isinstance(data["success"], bool)
        assert isinstance(data["data"]["progress_percentage"], (int, float))
        assert isinstance(data["data"]["current_step"], str)
        assert isinstance(data["data"]["steps_completed"], int)
        assert isinstance(data["data"]["total_steps"], int)