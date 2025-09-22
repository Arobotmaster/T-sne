"""
WebSocket实时通信模块

实现处理流水线的实时状态更新功能，符合SDD Constitution原则：
- Library-First: 独立的WebSocket服务模块
- CLI Interface: 支持命令行测试
- Test-First: 完整的WebSocket测试覆盖
- Integration-First: 与处理流水线集成
- Scientific Observability: 详细的连接和消息日志
"""

import json
import asyncio
from typing import Dict, Any, Set
from datetime import datetime
import uuid
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from src.config.logging_config import get_logger

logger = get_logger(__name__)

# WebSocket连接管理器
class ConnectionManager:
    """WebSocket连接管理器"""

    def __init__(self):
        # 存储活跃连接: {pipeline_id: Set[WebSocket]}
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # 存储连接信息: {connection_id: {"pipeline_id": str, "connected_at": datetime}}
        self.connection_info: Dict[str, Dict[str, Any]] = {}
        logger.info("WebSocket连接管理器初始化完成")

    async def connect(self, websocket: WebSocket, pipeline_id: str):
        """建立WebSocket连接"""
        await websocket.accept()
        connection_id = str(uuid.uuid4())

        # 为pipeline_id创建连接集合（如果不存在）
        if pipeline_id not in self.active_connections:
            self.active_connections[pipeline_id] = set()

        # 添加连接
        self.active_connections[pipeline_id].add(websocket)

        # 记录连接信息
        self.connection_info[connection_id] = {
            "pipeline_id": pipeline_id,
            "connected_at": datetime.now().isoformat(),
            "websocket": websocket
        }

        logger.info(f"WebSocket连接建立: pipeline_id={pipeline_id}, connection_id={connection_id}")
        logger.info(f"当前活跃连接数: {len(self.connection_info)}")

        # 发送连接成功消息
        await self.send_personal_message({
            "type": "connection_established",
            "connection_id": connection_id,
            "pipeline_id": pipeline_id,
            "timestamp": datetime.now().isoformat(),
            "message": "WebSocket连接建立成功"
        }, websocket)

    def disconnect(self, websocket: WebSocket):
        """断开WebSocket连接"""
        # 查找并移除连接
        connection_id_to_remove = None
        pipeline_id_to_remove = None

        for conn_id, info in self.connection_info.items():
            if info["websocket"] == websocket:
                connection_id_to_remove = conn_id
                pipeline_id_to_remove = info["pipeline_id"]
                break

        if connection_id_to_remove:
            # 从连接信息中移除
            del self.connection_info[connection_id_to_remove]

            # 从活跃连接中移除
            if pipeline_id_to_remove in self.active_connections:
                self.active_connections[pipeline_id_to_remove].discard(websocket)

                # 如果该pipeline没有连接了，清理字典
                if not self.active_connections[pipeline_id_to_remove]:
                    del self.active_connections[pipeline_id_to_remove]

            logger.info(f"WebSocket连接断开: pipeline_id={pipeline_id_to_remove}, connection_id={connection_id_to_remove}")
            logger.info(f"当前活跃连接数: {len(self.connection_info)}")

    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """发送个人消息"""
        try:
            await websocket.send_text(json.dumps(message, ensure_ascii=False))
        except Exception as e:
            logger.error(f"发送个人消息失败: {str(e)}")
            self.disconnect(websocket)

    async def broadcast_to_pipeline(self, message: Dict[str, Any], pipeline_id: str):
        """向特定pipeline的所有连接广播消息"""
        if pipeline_id not in self.active_connections:
            logger.warning(f"Pipeline {pipeline_id} 没有活跃连接")
            return

        # 添加时间戳
        message["timestamp"] = datetime.now().isoformat()

        # 复制连接集合以避免在迭代时修改
        connections = self.active_connections[pipeline_id].copy()

        for connection in connections:
            try:
                await connection.send_text(json.dumps(message, ensure_ascii=False))
                logger.debug(f"消息已发送到pipeline {pipeline_id}: {message.get('type', 'unknown')}")
            except Exception as e:
                logger.error(f"广播消息失败: {str(e)}")
                self.disconnect(connection)

    async def get_pipeline_connections(self, pipeline_id: str) -> int:
        """获取指定pipeline的连接数"""
        return len(self.active_connections.get(pipeline_id, set()))

    def get_connection_stats(self) -> Dict[str, Any]:
        """获取连接统计信息"""
        return {
            "total_connections": len(self.connection_info),
            "active_pipelines": len(self.active_connections),
            "pipeline_connections": {
                pipeline_id: len(connections)
                for pipeline_id, connections in self.active_connections.items()
            },
            "timestamp": datetime.now().isoformat()
        }

# 全局连接管理器实例
manager = ConnectionManager()

# 模拟流水线状态（实际项目中应该从处理服务获取真实状态）
_pipeline_status_store = {}

async def simulate_pipeline_progress(pipeline_id: str):
    """模拟流水线进度更新"""
    if pipeline_id not in _pipeline_status_store:
        _pipeline_status_store[pipeline_id] = {
            "status": "running",
            "progress": 0,
            "current_step": "数据预处理",
            "start_time": datetime.now().isoformat()
        }

    pipeline = _pipeline_status_store[pipeline_id]

    # 模拟处理步骤
    steps = [
        {"name": "数据预处理", "duration": 5},
        {"name": "PCA降维", "duration": 10},
        {"name": "t-SNE降维", "duration": 15},
        {"name": "可视化生成", "duration": 8}
    ]

    for step in steps:
        if pipeline["status"] != "running":
            break

        pipeline["current_step"] = step["name"]

        # 发送步骤开始消息
        await manager.broadcast_to_pipeline({
            "type": "step_started",
            "pipeline_id": pipeline_id,
            "step_name": step["name"],
            "message": f"开始{step['name']}"
        }, pipeline_id)

        # 模拟步骤进度
        for i in range(1, 101):
            if pipeline["status"] != "running":
                break

            progress = int((len([s for s in steps if steps.index(s) < steps.index(step)]) + i/100) * 100 / len(steps))
            pipeline["progress"] = progress

            # 每20%发送一次进度更新
            if i % 20 == 0:
                await manager.broadcast_to_pipeline({
                    "type": "progress_update",
                    "pipeline_id": pipeline_id,
                    "progress": progress,
                    "current_step": step["name"],
                    "step_progress": i,
                    "message": f"{step['name']}进度: {i}%"
                }, pipeline_id)

            await asyncio.sleep(step["duration"] / 100)  # 模拟处理时间

        # 发送步骤完成消息
        await manager.broadcast_to_pipeline({
            "type": "step_completed",
            "pipeline_id": pipeline_id,
            "step_name": step["name"],
            "message": f"{step['name']}完成"
        }, pipeline_id)

    # 处理完成
    pipeline["status"] = "completed"
    pipeline["progress"] = 100

    await manager.broadcast_to_pipeline({
        "type": "pipeline_completed",
        "pipeline_id": pipeline_id,
        "progress": 100,
        "duration": sum(step["duration"] for step in steps),
        "message": "处理流水线完成"
    }, pipeline_id)

async def websocket_endpoint(websocket: WebSocket, pipeline_id: str):
    """WebSocket端点 - T048"""
    logger.info(f"WebSocket连接请求: pipeline_id={pipeline_id}")

    # 验证pipeline_id存在（实际项目中应该检查数据库）
    if not pipeline_id or len(pipeline_id) < 10:
        logger.warning(f"无效的pipeline_id: {pipeline_id}")
        await websocket.close(code=4000, reason="无效的pipeline_id")
        return

    # 建立连接
    await manager.connect(websocket, pipeline_id)

    try:
        # 如果流水线不存在，创建模拟状态
        if pipeline_id not in _pipeline_status_store:
            # 启动模拟处理任务
            asyncio.create_task(simulate_pipeline_progress(pipeline_id))

        # 保持连接并处理客户端消息
        while True:
            try:
                # 等待客户端消息
                data = await websocket.receive_text()
                message = json.loads(data)

                logger.debug(f"收到客户端消息: {message}")

                # 处理不同类型的消息
                msg_type = message.get("type")

                if msg_type == "ping":
                    # 响应ping消息
                    await manager.send_personal_message({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }, websocket)

                elif msg_type == "get_status":
                    # 获取当前状态
                    pipeline_status = _pipeline_status_store.get(pipeline_id, {})
                    await manager.send_personal_message({
                        "type": "status_response",
                        "pipeline_id": pipeline_id,
                        "status": pipeline_status
                    }, websocket)

                elif msg_type == "pause_pipeline":
                    # 暂停流水线
                    if pipeline_id in _pipeline_status_store:
                        _pipeline_status_store[pipeline_id]["status"] = "paused"
                        await manager.broadcast_to_pipeline({
                            "type": "pipeline_paused",
                            "pipeline_id": pipeline_id,
                            "message": "流水线已暂停"
                        }, pipeline_id)

                elif msg_type == "resume_pipeline":
                    # 恢复流水线
                    if pipeline_id in _pipeline_status_store:
                        _pipeline_status_store[pipeline_id]["status"] = "running"
                        await manager.broadcast_to_pipeline({
                            "type": "pipeline_resumed",
                            "pipeline_id": pipeline_id,
                            "message": "流水线已恢复"
                        }, pipeline_id)

                elif msg_type == "cancel_pipeline":
                    # 取消流水线
                    if pipeline_id in _pipeline_status_store:
                        _pipeline_status_store[pipeline_id]["status"] = "cancelled"
                        await manager.broadcast_to_pipeline({
                            "type": "pipeline_cancelled",
                            "pipeline_id": pipeline_id,
                            "message": "流水线已取消"
                        }, pipeline_id)

                else:
                    logger.warning(f"未知的消息类型: {msg_type}")
                    await manager.send_personal_message({
                        "type": "error",
                        "message": f"未知的消息类型: {msg_type}"
                    }, websocket)

            except json.JSONDecodeError:
                logger.error("JSON解析错误")
                await manager.send_personal_message({
                    "type": "error",
                    "message": "消息格式错误，需要有效的JSON"
                }, websocket)

            except Exception as e:
                logger.error(f"处理消息时出错: {str(e)}")
                await manager.send_personal_message({
                    "type": "error",
                    "message": f"处理消息时出错: {str(e)}"
                }, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"WebSocket连接断开: pipeline_id={pipeline_id}")

    except Exception as e:
        logger.error(f"WebSocket连接错误: {str(e)}")
        manager.disconnect(websocket)