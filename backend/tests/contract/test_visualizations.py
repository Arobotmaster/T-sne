"""
契约测试：可视化API

遵循SDD Constitution的Test-First原则
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

with patch('backend.main.app') as mock_app:
    from backend.main import app

class TestVisualizationsAPI:
    """可视化API契约测试"""

    def setup_method(self):
        """设置测试环境"""
        self.client = TestClient(app)
        self.test_pipeline_id = "test-pipeline-123"

    def test_get_visualization_data_success(self):
        """测试成功获取可视化数据"""
        response = self.client.get(f"/api/visualizations/{self.test_pipeline_id}")

        # 期望的状态码和响应结构
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert data["success"] == True
        assert "data" in data
        assert "coordinates" in data["data"]
        assert "config" in data["data"]

    def test_get_visualization_data_not_found(self):
        """测试获取不存在的可视化数据"""
        response = self.client.get("/api/visualizations/nonexistent-pipeline")

        # 期望返回404错误
        assert response.status_code == 404

    def test_visualization_response_schema(self):
        """测试可视化响应的JSON Schema"""
        response = self.client.get(f"/api/visualizations/{self.test_pipeline_id}")

        assert response.status_code == 200
        data = response.json()

        # 验证响应结构
        required_fields = ["success", "data"]
        for field in required_fields:
            assert field in data

        # 验证data字段结构
        data_fields = ["coordinates", "config", "metadata", "total_samples"]
        for field in data_fields:
            assert field in data["data"]

        # 验证coordinates结构
        coordinates = data["data"]["coordinates"]
        assert isinstance(coordinates, list)
        if len(coordinates) > 0:
            coord = coordinates[0]
            coord_fields = ["sample_id", "x_coordinate", "y_coordinate", "category_id", "category_name"]
            for field in coord_fields:
                assert field in coord