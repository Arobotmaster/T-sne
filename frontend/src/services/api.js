/**
 * API服务层 - MOF数据t-SNE可视化应用
 * 提供与后端API通信的所有功能
 */

class APIService {
    constructor(baseURL = 'http://localhost:8000/api') {
        this.baseURL = baseURL;
        // 不在全局强制 Content-Type，避免 GET 触发预检（OPTIONS）
        this.defaultHeaders = {};
    }

    /**
     * 通用请求方法
     * @param {string} endpoint - API端点
     * @param {Object} options - 请求选项
     * @returns {Promise} 响应数据
     */
    async request(endpoint, options = {}) {
        // 支持绝对URL；否则拼接到 baseURL
        const url = /^https?:\/\//.test(endpoint) ? endpoint : `${this.baseURL}${endpoint}`;

        // 仅在有 body 且非 FormData 且未显式提供 Content-Type 时才自动加 JSON 头
        const headers = { ...this.defaultHeaders, ...(options.headers || {}) };
        const isFormData = typeof FormData !== 'undefined' && options.body instanceof FormData;
        if (options.body && !isFormData && !headers['Content-Type']) {
            headers['Content-Type'] = 'application/json';
        }

        const { method = options.method || (options.body ? 'POST' : 'GET'), body } = options;

        const config = {
            method,
            headers,
            body
        };

        try {
            const response = await fetch(url, config);

            // 处理非JSON响应（如文件下载）
            if (options.responseType === 'blob') {
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                return await response.blob();
            }

            // 处理JSON响应
            const data = await response.json().catch(() => ({}));

            if (!response.ok) {
                throw new Error(data.detail || `HTTP ${response.status}: ${response.statusText}`);
            }

            return data;
        } catch (error) {
            console.error(`API请求失败 [${endpoint}]:`, error);
            throw error;
        }
    }

    /**
     * 检查后端连接状态
     * @returns {Promise<Object>} 连接状态
     */
    async checkHealth() {
        // 直接请求后端根健康检查，避免 /api/v1 前缀导致 404/预检
        return this.request('http://localhost:8000/health');
    }

    /**
     * 上传CSV文件
     * @param {File} file - CSV文件对象
     * @returns {Promise<Object>} 上传结果
     */
    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        return this.request('/upload', {
            method: 'POST',
            body: formData,
            headers: {} // 让浏览器自动设置Content-Type
        });
    }

    /**
     * 获取数据集列表
     * @param {Object} params - 查询参数
     * @returns {Promise<Object>} 数据集列表
     */
    async getDatasets(params = {}) {
        const queryString = new URLSearchParams(params).toString();
        const base = '/datasets';
        const endpoint = queryString ? `${base}?${queryString}` : base;
        return this.request(endpoint);
    }

    /**
     * 获取数据集信息
     * @param {string} datasetId - 数据集ID
     * @returns {Promise<Object>} 数据集信息
     */
    async getDatasetInfo(datasetId) {
        return this.request(`/datasets/${datasetId}`);
    }

    /**
     * 获取数据集状态
     * @param {string} datasetId - 数据集ID
     * @returns {Promise<Object>} 数据集状态
     */
    async getDatasetStatus(datasetId) {
        return this.request(`/datasets/${datasetId}/status`);
    }

    /**
     * 获取数据集统计信息
     * @param {string} datasetId - 数据集ID
     * @returns {Promise<Object>} 统计信息
     */
    async getDatasetStatistics(datasetId) {
        return this.request(`/datasets/${datasetId}/statistics`);
    }

    /**
     * 获取数据集预览
     * @param {string} datasetId - 数据集ID
     * @param {Object} params - 预览参数
     * @returns {Promise<Object>} 预览数据
     */
    async getDatasetPreview(datasetId, params = {}) {
        const queryString = new URLSearchParams(params).toString();
        const base = `/datasets/${datasetId}/preview`;
        const endpoint = queryString ? `${base}?${queryString}` : base;
        return this.request(endpoint);
    }

    /**
     * 删除数据集
     * @param {string} datasetId - 数据集ID
     * @returns {Promise<Object>} 删除结果
     */
    async deleteDataset(datasetId) {
        return this.request(`/datasets/${datasetId}`, {
            method: 'DELETE'
        });
    }

    /**
     * 启动数据处理流水线
     * @param {string} datasetId - 数据集ID
     * @param {Object} config - 处理配置
     * @returns {Promise<Object>} 处理响应
     */
    async startProcessing(datasetId, config = {}) {
        const defaultConfig = {
            preprocessing_config: {
                missing_value_strategy: 'mean',
                scaling_method: 'standard',
                outlier_detection: true,
                outlier_threshold: 3.0
            },
            pca_config: {
                n_components: 50,
                whiten: false,
                random_state: 42
            },
            tsne_config: {
                perplexity: 30,
                n_components: 2,
                learning_rate: 200,
                n_iter: 1000,
                random_state: 42,
                metric: 'euclidean'
            }
        };

        const finalConfig = this.mergeConfig(defaultConfig, config);

        return this.request(`/datasets/${datasetId}/process`, {
            method: 'POST',
            body: JSON.stringify(finalConfig)
        });
    }

    /**
     * 自动调参（后端网格搜索）
     * @param {string} datasetId
     * @param {Object} options 例如 {perplexities:[...], metrics:[...], max_search_rows:5000}
     * @returns {Promise<Object>} { success, data: { recommended_config, results, ... } }
     */
    async autoTuneDataset(datasetId, options = {}) {
        return this.request(`/datasets/${datasetId}/auto-config`, {
            method: 'POST',
            body: JSON.stringify(options || {})
        });
    }

    /**
     * 获取处理流水线状态
     * @param {string} pipelineId - 流水线ID
     * @returns {Promise<Object>} 流水线状态
     */
    async getPipelineStatus(pipelineId) {
        return this.request(`/pipelines/${pipelineId}/status`);
    }

    /**
     * 从文件直接启动处理（开发/快速通道）
     * @param {Object} payload - { file_path, embedding, id_column, category_column, config? }
     * @returns {Promise<Object>} { success, task_id }
     */
    async startProcessingFromFile(payload) {
        return this.request(`/processing/from-file`, {
            method: 'POST',
            body: JSON.stringify(payload)
        });
    }

    /**
     * 获取处理任务状态（from-file）
     * @param {string} taskId - 任务ID
     * @returns {Promise<Object>} 任务状态
     */
    async getProcessingTaskStatus(taskId) {
        return this.request(`/processing/${taskId}/status`);
    }

    /**
     * 获取可视化数据
     * @param {string} pipelineId - 流水线ID
     * @returns {Promise<Object>} 可视化数据
     */
    async getVisualizationData(pipelineId) {
        return this.request(`/visualizations/${pipelineId}`);
    }

    /** 获取可视化/训练指标 */
    async getVisualizationMetrics(pipelineId) {
        return this.request(`/visualizations/${pipelineId}/metrics`);
    }

    /** 获取训练报告（参数+指标+调参结果） */
    async getTrainingReport(pipelineId) {
        return this.request(`/visualizations/${pipelineId}/report`);
    }

    /**
     * 更新可视化配置
     * @param {string} pipelineId - 流水线ID
     * @param {Object} config - 可视化配置
     * @returns {Promise<Object>} 更新后的配置
     */
    async updateVisualizationConfig(pipelineId, config) {
        return this.request(`/visualizations/${pipelineId}/config`, {
            method: 'PUT',
            body: JSON.stringify(config)
        });
    }

    /**
     * 导出可视化图像
     * @param {string} pipelineId - 流水线ID
     * @param {Object} exportConfig - 导出配置
     * @returns {Promise<Object>} 导出结果
     */
    async exportVisualization(pipelineId, exportConfig = {}) {
        const defaultConfig = {
            format: 'png',
            width: 1200,
            height: 800,
            dpi: 300,
            background_color: 'white',
            filename: 'mof-visualization'
        };

        const finalConfig = this.mergeConfig(defaultConfig, exportConfig);

        return this.request(`/visualizations/${pipelineId}/export`, {
            method: 'POST',
            body: JSON.stringify(finalConfig)
        });
    }

    /**
     * 下载导出的文件
     * @param {string} file_id - 文件ID
     * @returns {Promise<Blob>} 文件内容
     */
    async downloadFile(fileId) {
        return this.request(`/downloads/${fileId}`, {
            responseType: 'blob'
        });
    }

    /**
     * 获取对比可视化数据
     * @param {string} pipelineId - 流水线ID
     * @param {Object} comparisonConfig - 对比配置
     * @returns {Promise<Object>} 对比可视化数据
     */
    async getComparisonData(pipelineId, comparisonConfig = {}) {
        const defaultConfig = {
            show_original: true,
            show_filtered: true,
            original_opacity: 0.3,
            filtered_opacity: 1.0,
            color_scheme: 'viridis'
        };

        const finalConfig = this.mergeConfig(defaultConfig, comparisonConfig);

        return this.request(`/visualizations/comparison`, {
            method: 'POST',
            body: JSON.stringify({
                pipeline_id: pipelineId,
                ...finalConfig
            })
        });
    }

    /**
     * 更新对比可视化配置
     * @param {string} pipelineId - 流水线ID
     * @param {Object} config - 对比配置
     * @returns {Promise<Object>} 更新后的配置
     */
    async updateComparisonConfig(pipelineId, config) {
        return this.request(`/visualizations/${pipelineId}/comparison-config`, {
            method: 'PUT',
            body: JSON.stringify(config)
        });
    }

    /**
     * 合并配置对象
     * @param {Object} defaultConfig - 默认配置
     * @param {Object} userConfig - 用户配置
     * @returns {Object} 合并后的配置
     */
    mergeConfig(defaultConfig, userConfig) {
        const merged = { ...defaultConfig };

        for (const key in userConfig) {
            if (typeof userConfig[key] === 'object' && userConfig[key] !== null && !Array.isArray(userConfig[key])) {
                merged[key] = { ...merged[key], ...userConfig[key] };
            } else {
                merged[key] = userConfig[key];
            }
        }

        return merged;
    }

    /**
     * 验证API响应格式
     * @param {Object} response - API响应
     * @param {Array<string>} requiredFields - 必需字段
     * @returns {boolean} 验证结果
     */
    validateResponse(response, requiredFields = []) {
        if (!response || typeof response !== 'object') {
            return false;
        }

        return requiredFields.every(field => {
            return field in response && response[field] !== null && response[field] !== undefined;
        });
    }

    /**
     * 处理API错误
     * @param {Error} error - 错误对象
     * @returns {string} 用户友好的错误消息
     */
    handleApiError(error) {
        if (error.message.includes('HTTP 404')) {
            return '请求的资源不存在';
        } else if (error.message.includes('HTTP 400')) {
            return '请求参数错误';
        } else if (error.message.includes('HTTP 413')) {
            return '文件过大，请选择较小的文件';
        } else if (error.message.includes('HTTP 415')) {
            return '不支持的文件格式';
        } else if (error.message.includes('Failed to fetch')) {
            return '网络连接失败，请检查后端服务';
        } else {
            return `操作失败: ${error.message}`;
        }
    }

    /**
     * 创建WebSocket连接
     * @param {string} pipelineId - 流水线ID
     * @param {Function} onMessage - 消息处理函数
     * @returns {WebSocket} WebSocket实例
     */
    createWebSocket(pipelineId, onMessage) {
        // 从 baseURL 解析出后端主机端口，确保在静态端口(8001)运行时也能连到后端(8000)
        let apiHost = '';
        try {
            apiHost = new URL(this.baseURL).host;
        } catch (e) {
            apiHost = window.location.host; // 兜底
        }
        const wsUrl = `ws://${apiHost}/ws/pipelines/${pipelineId}`;
        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            console.log('WebSocket连接已建立');
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                onMessage(data);
            } catch (error) {
                console.error('WebSocket消息解析失败:', error);
            }
        };

        ws.onerror = (error) => {
            console.error('WebSocket错误:', error);
        };

        ws.onclose = () => {
            console.log('WebSocket连接已关闭');
        };

        return ws;
    }
}

// 导出API服务类
export default APIService;

// 创建全局API服务实例
window.apiService = new APIService();

// 导出一些有用的工具函数
export const APIUtils = {
    /**
     * 格式化文件大小
     * @param {number} bytes - 字节数
     * @returns {string} 格式化后的大小
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    /**
     * 格式化持续时间
     * @param {number} ms - 毫秒数
     * @returns {string} 格式化后的时间
     */
    formatDuration(ms) {
        if (ms < 1000) return `${ms}ms`;
        if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
        if (ms < 3600000) return `${(ms / 60000).toFixed(1)}min`;
        return `${(ms / 3600000).toFixed(1)}h`;
    },

    /**
     * 生成UUID
     * @returns {string} UUID字符串
     */
    generateUUID() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c == 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }
};
