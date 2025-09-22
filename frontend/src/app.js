/**
 * MOF数据t-SNE可视化应用主程序
 * 集成所有前端组件的核心应用程序
 */

import UploadComponent from './components/upload.js';
import ParametersComponent from './components/parameters.js';
import VisualizationComponent from './components/visualization.js';
import ExportComponent from './components/export.js';
import ComparisonComponent from './components/comparison.js';
import DatasetSelectorComponent from './components/dataset-selector.js';
import APIService from './services/api.js';

class MOFVisualizationApp {
    constructor() {
        this.apiService = new APIService();
        this.currentDataset = null;
        this.currentPipeline = null;
        this.components = {};
        this.currentView = 'upload';

        this.init();
    }

    init() {
        this.setupComponents();
        this.setupEventListeners();
        this.setupNavigation();
        this.checkBackendConnection();
    }

    setupComponents() {
        // 初始化所有组件
        this.components.upload = new UploadComponent('upload-section', {
            autoProcess: false
        });

        this.components.parameters = new ParametersComponent('parameters-container');
        this.components.visualization = new VisualizationComponent('visualization-section');
        this.components.export = new ExportComponent('export-section');
        this.components.comparison = new ComparisonComponent('comparison-section');
        this.components.datasetSelector = new DatasetSelectorComponent('dataset-selector-section', {
            multiSelect: true,
            showActions: true,
            showStatus: true
        });

        this.setupComponentEvents();

        // 开发快捷入口：从文件直接计算 E2/E1
        this.injectDevShortcuts();
    }

    injectDevShortcuts() {
        const host = document.getElementById('dataset-selector-section');
        if (!host) return;
        const panel = document.createElement('div');
        panel.className = 'card mb-3';
        panel.innerHTML = `
            <div class="card-header">
                <strong>开发快捷入口</strong>
            </div>
            <div class="card-body">
                <div class="d-flex gap-2 flex-wrap">
                    <button class="btn btn-sm btn-outline-primary" id="btn-run-e2-file">
                        计算 E2（4519，含类别）
                    </button>
                    <button class="btn btn-sm btn-outline-secondary" id="btn-run-e1-file">
                        计算 E1（12088，完整集）
                    </button>
                </div>
                <div class="form-text mt-2">使用 data/uploads/CSV_pld_filtered_category_4519.csv 与 CSV_full_12088.csv</div>
            </div>
        `;
        host.prepend(panel);

        const runE2 = () => this.runFromFile({
            file_path: 'data/uploads/CSV_pld_filtered_category_4519.csv',
            embedding: 'E2',
            id_column: 'sample_id',
            category_column: 'category'
        });
        const runE1 = () => this.runFromFile({
            file_path: 'data/uploads/CSV_full_12088.csv',
            embedding: 'E1',
            id_column: 'sample_id',
            category_column: 'category'
        });

        panel.querySelector('#btn-run-e2-file').addEventListener('click', runE2);
        panel.querySelector('#btn-run-e1-file').addEventListener('click', runE1);
    }

    async runFromFile(payload) {
        try {
            this.showNotification(`开始从文件计算 (${payload.embedding})...`, 'info');
            const resp = await this.apiService.startProcessingFromFile(payload);
            if (!resp || !resp.success) {
                this.showNotification('启动计算失败', 'error');
                return;
            }
            const taskId = resp.task_id;
            const poll = async () => {
                try {
                    const st = await this.apiService.getProcessingTaskStatus(taskId);
                    if (st.status === 'completed' && st.result && st.result.pipeline_id) {
                        const pipelineId = st.result.pipeline_id;
                        this.currentPipeline = { pipeline_id: pipelineId };
                        // 记录到嵌入表
                        if (!this.embeddings) this.embeddings = {};
                        this.embeddings[payload.embedding] = { pipelineId };
                        // 通知可视化组件更新可用嵌入并加载
                        if (this.components?.visualization?.setEmbeddings) {
                            await this.components.visualization.setEmbeddings(this.embeddings);
                        } else {
                            await this.loadVisualizationData(pipelineId);
                        }
                        this.switchToView('visualization');
                        this.showNotification('计算完成，已加载可视化', 'success');
                    } else if (st.status === 'failed') {
                        this.showNotification(`计算失败: ${st.error || '未知错误'}`, 'error');
                    } else {
                        setTimeout(poll, 2000);
                    }
                } catch (e) {
                    console.error('轮询任务状态失败', e);
                    setTimeout(poll, 3000);
                }
            };
            poll();
        } catch (error) {
            const msg = this.apiService.handleApiError(error);
            this.showNotification('计算启动失败: ' + msg, 'error');
        }
    }

    setupComponentEvents() {
        // 文件上传组件事件
        this.components.upload.on('uploadSuccess', (datasetInfo) => {
            this.currentDataset = datasetInfo;
            this.showNotification('文件上传成功！', 'success');
            this.switchToView('parameters');
        });

        this.components.upload.on('processData', (datasetInfo) => {
            this.currentDataset = datasetInfo;
            this.switchToView('parameters');
        });

        // 参数配置组件事件
        this.components.parameters.on('parameterChange', (config) => {
            console.log('参数配置更新:', config);
        });

        this.components.parameters.on('parametersApplied', async (config) => {
            await this.startProcessing(config);
        });

        // 可视化组件事件
        this.components.visualization.on('pointSelect', (selection) => {
            console.log('点选择:', selection);
        });

        this.components.visualization.on('exportRequest', (format) => {
            this.switchToView('export');
        });

        // 导出组件事件
        this.components.export.on('exportComplete', (result) => {
            this.showNotification('导出成功！', 'success');
        });

        // 对比组件事件
        this.components.comparison.on('datasetsSelected', (datasets) => {
            console.log('对比数据集选择:', datasets);
        });

        // 数据集选择器组件事件
        this.components.datasetSelector.on('datasetSelected', (datasetId) => {
            this.loadDataset(datasetId);
        });

        // 从数据集列表直接进入处理或查看
        this.components.datasetSelector.on('processDataset', async (datasetId) => {
            await this.loadDataset(datasetId);
            this.switchToView('parameters');
        });
        this.components.datasetSelector.on('viewDataset', async (datasetId) => {
            try {
                // 仅在已处理过时进入可视化；否则提示用户先“处理”
                const statusResp = await this.apiService.getDatasetStatus(datasetId);
                // 兼容不同返回结构：优先取 data/last_pipeline_id，其次 pipeline_id 或 recent_pipeline_id
                const statusData = statusResp.data || statusResp;
                const pipelineId = statusData?.last_pipeline_id || statusData?.pipeline_id || statusData?.recent_pipeline_id;

                if (pipelineId) {
                    await this.loadVisualizationData(pipelineId);
                    this.switchToView('visualization');
                } else {
                    this.showNotification('该数据集尚未处理，请点击“处理”进入参数配置后开始处理', 'info');
                }
            } catch (error) {
                const msg = this.apiService.handleApiError(error);
                this.showNotification('无法获取数据集状态：' + msg, 'warning');
            }
        });

        this.components.datasetSelector.on('selectionChanged', (selection) => {
            console.log('数据集选择变更:', selection);
        });

        this.components.datasetSelector.on('compareSelected', (datasets) => {
            this.loadComparisonDatasets(datasets);
            this.switchToView('comparison');
        });

        this.components.datasetSelector.on('uploadNewDataset', () => {
            this.switchToView('upload');
        });
    }

    setupEventListeners() {
        // 确保初始化只触发一次，避免遗漏或重复
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.initializeUI(), { once: true });
        } else {
            this.initializeUI();
        }

        // WebSocket 连接（如果后端支持）
        this.setupWebSocket();
    }

    setupNavigation() {
        const navLinks = document.querySelectorAll('.nav-link');
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const view = link.getAttribute('data-view');
                if (view) {
                    this.switchToView(view);
                }
            });
        });
    }

    switchToView(viewName) {
        // 隐藏所有视图
        const views = document.querySelectorAll('.view-section');
        views.forEach(view => {
            view.style.display = 'none';
        });

        // 显示目标视图
        const targetView = document.getElementById(`${viewName}-section`);
        if (targetView) {
            targetView.style.display = 'block';
            this.currentView = viewName;

            // 更新导航状态
            const navLinks = document.querySelectorAll('.nav-link');
            navLinks.forEach(link => {
                link.classList.remove('active');
                if (link.getAttribute('data-view') === viewName) {
                    link.classList.add('active');
                }
            });

            // 触发视图切换事件
            this.emit('viewChanged', viewName);
        }
    }

    async checkBackendConnection() {
        try {
            await this.apiService.checkHealth();
            this.showNotification('后端服务连接正常', 'info');
        } catch (error) {
            this.showNotification('后端服务连接异常，请检查服务是否启动', 'warning');
        }
    }

    async startProcessing(config) {
        if (!this.currentDataset) {
            this.showNotification('请先上传数据集', 'error');
            return;
        }

        try {
            this.showNotification('开始处理数据...', 'info');

            // 启动数据处理流水线
            let pipelineResponse;
            try {
                pipelineResponse = await this.apiService.startProcessing(this.currentDataset.dataset_id, config);
            } catch (err) {
                const msg = String(err?.message || '');
                // 若后端返回409“正在处理中”，自动强制取消并重启
                if (msg.includes('409') || msg.includes('正在处理中')) {
                    this.showNotification('检测到正在处理的任务，已为你取消并重启...', 'warning');
                    const forceConfig = { ...config, force: true };
                    pipelineResponse = await this.apiService.startProcessing(this.currentDataset.dataset_id, forceConfig);
                } else {
                    throw err;
                }
            }

            if (pipelineResponse.success) {
                this.currentPipeline = pipelineResponse.data;

                // 绑定导出组件所需上下文
                if (this.components.export) {
                    this.components.export.setPipeline(this.currentPipeline);
                    this.components.export.setVisualizationComponent(this.components.visualization);
                }

                // 监听处理进度
                this.monitorProcessing(this.currentPipeline.pipeline_id);

                // 切换到可视化视图
                this.switchToView('visualization');
            } else {
                this.showNotification('创建处理流水线失败: ' + pipelineResponse.detail, 'error');
            }
        } catch (error) {
            const errorMessage = this.apiService.handleApiError(error);
            this.showNotification('处理失败: ' + errorMessage, 'error');
        }
    }

    async monitorProcessing(pipelineId) {
        const checkStatus = async () => {
            try {
                const resp = await this.apiService.getPipelineStatus(pipelineId);
                // 兼容 {success, data} 和裸对象
                const status = resp?.data && resp.data.status ? resp.data : resp;

                if (status && status.status) {
                    if (status.status === 'completed') {
                        // 处理完成，加载数据
                        await this.loadVisualizationData(pipelineId);
                        // 拉取指标并提示
                        try {
                            const m = await this.apiService.getVisualizationMetrics(pipelineId);
                            const md = m?.data || {};
                            const parts = [];
                            if (md.tsne && (md.tsne.kl_divergence !== undefined)) parts.push(`KL=${Number(md.tsne.kl_divergence).toFixed(4)}`);
                            if (md.pca && (md.pca.total_variance_retained !== undefined)) parts.push(`PCA累计方差=${(Number(md.pca.total_variance_retained)*100).toFixed(1)}%`);
                            if (md.total_samples) parts.push(`样本数=${md.total_samples}`);
                            this.showNotification('数据处理完成！' + (parts.length? (' 指标：' + parts.join(' | ')) : ''), 'success');
                        } catch (_) {
                            this.showNotification('数据处理完成！', 'success');
                        }
                        return;
                    } else if (status.status === 'error' || status.status === 'failed') {
                        this.showNotification('数据处理出错: ' + status.error_message, 'error');
                        return;
                    } else {
                        // 继续监控
                        setTimeout(checkStatus, 2000);
                    }
                }
            } catch (error) {
                console.error('监控处理状态出错:', error);
            }
        };

        checkStatus();
    }

    async loadVisualizationData(pipelineId) {
        try {
            await this.components.visualization.loadData(pipelineId);
        } catch (error) {
            const errorMessage = this.apiService.handleApiError(error);
            this.showNotification('加载可视化数据失败: ' + errorMessage, 'error');
        }
    }

    async loadDataset(datasetId) {
        try {
            const resp = await this.apiService.getDatasetInfo(datasetId);
            // 后端该端点返回“裸对象”，不包 success；兼容 success 包装与裸对象两种结构
            const ds = resp?.data && resp.data.dataset_id ? resp.data : resp;
            if (ds && ds.dataset_id) {
                this.currentDataset = ds;
                this.showNotification('数据集加载成功', 'success');
                this.switchToView('parameters');
                return;
            }
            this.showNotification('加载数据集失败', 'error');
        } catch (error) {
            const errorMessage = this.apiService.handleApiError(error);
            this.showNotification('加载数据集失败: ' + errorMessage, 'error');
        }
    }

    async loadComparisonDatasets(datasetIds) {
        try {
            const promises = datasetIds.map(id => this.apiService.getDatasetInfo(id));
            const responses = await Promise.all(promises);

            const datasets = responses
                .filter(response => response.success)
                .map(response => response.data);

            if (datasets.length > 0) {
                this.components.comparison.loadDatasets(datasets);
                this.showNotification(`成功加载 ${datasets.length} 个数据集`, 'success');
            } else {
                this.showNotification('无法加载对比数据集', 'error');
            }
        } catch (error) {
            const errorMessage = this.apiService.handleApiError(error);
            this.showNotification('加载对比数据集失败: ' + errorMessage, 'error');
        }
    }

    setupWebSocket() {
        // 将在获取到 pipelineId 后，使用 APIService.createWebSocket(pipelineId, this.handleWebSocketMessage.bind(this))
        // 以订阅处理进度与通知。此处不建立无上下文的通用连接。
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'processing_update':
                this.updateProcessingStatus(data.data);
                break;
            case 'dataset_status':
                this.updateDatasetStatus(data.data);
                break;
            case 'notification':
                this.showNotification(data.message, data.level);
                break;
            default:
                console.log('未知的WebSocket消息类型:', data.type);
        }
    }

    updateProcessingStatus(status) {
        // 更新处理状态显示
        console.log('处理状态更新:', status);
    }

    updateDatasetStatus(status) {
        // 更新数据集状态
        console.log('数据集状态更新:', status);
    }

    showNotification(message, type = 'info') {
        // 创建通知元素
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; max-width: 400px;';
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        document.body.appendChild(notification);

        // 自动移除通知
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }

    initializeUI() {
        // 初始化用户界面
        this.switchToView('upload');

        // 加载数据集列表
        this.components.datasetSelector.refresh();

        // 设置工具提示
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }

    emit(event, data) {
        // 触发应用程序级别的事件
        console.log(`App Event: ${event}`, data);
    }

    // 公共方法
    getCurrentDataset() {
        return this.currentDataset;
    }

    getCurrentPipeline() {
        return this.currentPipeline;
    }

    getCurrentView() {
        return this.currentView;
    }

    refreshData() {
        // 刷新所有数据
        this.components.datasetSelector.refresh();
        if (this.currentPipeline) {
            this.loadVisualizationData(this.currentPipeline.pipeline_id);
        }
    }

    destroy() {
        // 清理资源
        Object.values(this.components).forEach(component => {
            if (typeof component.destroy === 'function') {
                component.destroy();
            }
        });

        if (this.websocket) {
            this.websocket.close();
        }
    }
}

// 全局应用实例
let app;

// 当DOM加载完成后初始化应用
document.addEventListener('DOMContentLoaded', () => {
    app = new MOFVisualizationApp();

    // 将应用实例暴露到全局作用域以便调试
    window.mofApp = app;
});

// 导出应用类
export default MOFVisualizationApp;
