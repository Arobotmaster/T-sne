/**
 * 数据集选择器组件 - MOF数据t-SNE可视化应用
 * 管理多个数据集的选择、筛选和操作
 */

import APIService from '../services/api.js';

class DatasetSelectorComponent {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Container element with id '${containerId}' not found`);
        }

        this.options = {
            multiSelect: false,
            showActions: true,
            showStatus: true,
            autoRefresh: true,
            refreshInterval: 30000, // 30秒
            ...options
        };

        this.datasets = [];
        this.selectedDatasets = [];
        this.apiService = new APIService();
        this.eventHandlers = {};
        this.refreshTimer = null;

        this.init();
    }

    init() {
        this.createInterface();
        this.setupEventListeners();
        this.loadDatasets();

        if (this.options.autoRefresh) {
            this.startAutoRefresh();
        }
    }

    createInterface() {
        this.container.innerHTML = `
            <div class="dataset-selector-container">
                <!-- 数据集管理面板 -->
                <div class="dataset-management">
                    <div class="row">
                        <div class="col-md-6">
                            <h5><i class="bi bi-collection"></i> 数据集管理</h5>
                        </div>
                        <div class="col-md-6 text-end">
                            <button type="button" class="btn btn-primary btn-sm" id="btn-refresh-datasets">
                                <i class="bi bi-arrow-clockwise"></i> 刷新
                            </button>
                            <button type="button" class="btn btn-success btn-sm" id="btn-upload-new">
                                <i class="bi bi-plus-circle"></i> 上传新数据集
                            </button>
                        </div>
                    </div>
                </div>

                <!-- 筛选和排序 -->
                <div class="row">
                    <div class="col-md-6">
                        <div class="dataset-filter">
                            <div class="input-group">
                                <span class="input-group-text"><i class="bi bi-search"></i></span>
                                <input type="text" class="form-control" id="filter-datasets"
                                       placeholder="搜索数据集名称、描述...">
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="dataset-sort">
                            <select class="form-select" id="sort-datasets">
                                <option value="created_desc">创建时间 (最新)</option>
                                <option value="created_asc">创建时间 (最早)</option>
                                <option value="name_asc">名称 (A-Z)</option>
                                <option value="name_desc">名称 (Z-A)</option>
                                <option value="samples_desc">样本数量 (多到少)</option>
                                <option value="samples_asc">样本数量 (少到多)</option>
                            </select>
                        </div>
                    </div>
                </div>

                <!-- 数据集统计 -->
                <div class="row mb-3">
                    <div class="col-md-12">
                        <div class="dataset-stats">
                            <div class="row">
                                <div class="col-md-3">
                                    <div class="stats-card text-center">
                                        <h6>总数据集</h6>
                                        <h3 id="total-datasets">0</h3>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="stats-card text-center">
                                        <h6>就绪状态</h6>
                                        <h3 id="ready-datasets">0</h3>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="stats-card text-center">
                                        <h6>处理中</h6>
                                        <h3 id="processing-datasets">0</h3>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="stats-card text-center">
                                        <h6>错误状态</h6>
                                        <h3 id="error-datasets">0</h3>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 数据集列表 -->
                <div class="dataset-list" id="dataset-list">
                    <div class="text-center text-muted">
                        <i class="bi bi-hourglass-split"></i> 正在加载数据集...
                    </div>
                </div>

                <!-- 批量操作 -->
                ${this.options.multiSelect ? `
                <div class="batch-actions mt-3" style="display: none;">
                    <div class="alert alert-info">
                        <strong>已选择 <span id="selected-count">0</span> 个数据集</strong>
                        <div class="mt-2">
                            <button type="button" class="btn btn-primary btn-sm" id="btn-compare-selected">
                                <i class="bi bi-layout-split"></i> 对比选中
                            </button>
                            <button type="button" class="btn btn-warning btn-sm" id="btn-export-selected">
                                <i class="bi bi-download"></i> 导出选中
                            </button>
                            <button type="button" class="btn btn-danger btn-sm" id="btn-delete-selected">
                                <i class="bi bi-trash"></i> 删除选中
                            </button>
                            <button type="button" class="btn btn-secondary btn-sm" id="btn-clear-selection">
                                <i class="bi bi-x-circle"></i> 清除选择
                            </button>
                        </div>
                    </div>
                </div>
                ` : ''}
            </div>
        `;
    }

    setupEventListeners() {
        // 刷新按钮
        this.container.querySelector('#btn-refresh-datasets').addEventListener('click', () => {
            this.loadDatasets();
        });

        // 上传新数据集按钮
        this.container.querySelector('#btn-upload-new').addEventListener('click', () => {
            this.emit('uploadNewDataset');
        });

        // 搜索筛选
        this.container.querySelector('#filter-datasets').addEventListener('input', (e) => {
            this.filterDatasets(e.target.value);
        });

        // 排序
        this.container.querySelector('#sort-datasets').addEventListener('change', (e) => {
            this.sortDatasets(e.target.value);
        });

        // 批量操作按钮
        if (this.options.multiSelect) {
            this.container.querySelector('#btn-compare-selected').addEventListener('click', () => {
                this.emit('compareSelected', this.selectedDatasets);
            });

            this.container.querySelector('#btn-export-selected').addEventListener('click', () => {
                this.emit('exportSelected', this.selectedDatasets);
            });

            this.container.querySelector('#btn-delete-selected').addEventListener('click', () => {
                this.deleteSelectedDatasets();
            });

            this.container.querySelector('#btn-clear-selection').addEventListener('click', () => {
                this.clearSelection();
            });
        }
    }

    async loadDatasets() {
        try {
            this.showLoading();
            const response = await this.apiService.getDatasets();

            // 后端API返回的数据结构是 {datasets: [...], total: N, skip: 0, limit: 100, has_more: false}
            if (response.datasets) {
                this.datasets = response.datasets;
                this.renderDatasets();
                this.updateStats();
                this.emit('datasetsLoaded', this.datasets);
            } else {
                this.showError('加载数据集失败: ' + (response.detail || '未知错误'));
            }
        } catch (error) {
            const errorMessage = this.apiService.handleApiError(error);
            this.showError('加载数据集失败: ' + errorMessage);
        }
    }

    renderDatasets() {
        const listContainer = this.container.querySelector('#dataset-list');

        if (this.datasets.length === 0) {
            listContainer.innerHTML = `
                <div class="text-center text-muted">
                    <i class="bi bi-inbox" style="font-size: 3rem;"></i>
                    <h5>暂无数据集</h5>
                    <p>点击"上传新数据集"开始使用</p>
                </div>
            `;
            return;
        }

        const datasetsHtml = this.datasets.map(dataset => this.createDatasetCard(dataset)).join('');
        listContainer.innerHTML = datasetsHtml;

        // 为每个数据集卡片添加事件监听器
        this.datasets.forEach(dataset => {
            this.setupDatasetCardEvents(dataset.dataset_id);
        });
    }

    createDatasetCard(dataset) {
        const statusClass = this.getStatusClass(dataset.processing_status || 'uploaded');
        const statusText = this.getStatusText(dataset.processing_status || 'uploaded');
        const isSelected = this.selectedDatasets.includes(dataset.dataset_id);

        return `
            <div class="dataset-card ${isSelected ? 'selected' : ''}" data-dataset-id="${dataset.dataset_id}">
                <div class="dataset-card-header">
                    <h6 class="dataset-card-title">
                        ${this.options.multiSelect ? `
                            <input type="checkbox" class="form-check-input dataset-checkbox me-2"
                                   ${isSelected ? 'checked' : ''}>
                        ` : ''}
                        ${dataset.filename || dataset.dataset_id}
                    </h6>
                    ${this.options.showStatus ? `
                        <span class="dataset-card-status ${statusClass}">${statusText}</span>
                    ` : ''}
                </div>

                <div class="dataset-card-meta">
                    <div>
                        <strong>样本数:</strong> ${dataset.total_rows?.toLocaleString() || 'N/A'}
                    </div>
                    <div>
                        <strong>特征数:</strong> ${dataset.total_columns || 'N/A'}
                    </div>
                    <div>
                        <strong>文件大小:</strong> ${this.formatFileSize(dataset.file_size_bytes)}
                    </div>
                    <div>
                        <strong>创建时间:</strong> ${this.formatDate(dataset.upload_timestamp)}
                    </div>
                    <div>
                        <strong>数据质量:</strong> ${this.formatDataQuality(dataset.data_quality_score)}
                    </div>
                </div>

                ${dataset.description ? `
                    <div class="mt-2">
                        <small class="text-muted">${dataset.description}</small>
                    </div>
                ` : ''}

                ${this.options.showActions ? `
                    <div class="dataset-card-actions">
                        <button type="button" class="btn btn-primary btn-sm btn-view-dataset"
                                data-dataset-id="${dataset.dataset_id}">
                            <i class="bi bi-eye"></i> 查看
                        </button>
                        <button type="button" class="btn btn-info btn-sm btn-process-dataset"
                                data-dataset-id="${dataset.dataset_id}">
                            <i class="bi bi-gear"></i> 处理
                        </button>
                        <button type="button" class="btn btn-success btn-sm btn-export-dataset"
                                data-dataset-id="${dataset.dataset_id}">
                            <i class="bi bi-download"></i> 导出
                        </button>
                        <button type="button" class="btn btn-danger btn-sm btn-delete-dataset"
                                data-dataset-id="${dataset.dataset_id}">
                            <i class="bi bi-trash"></i> 删除
                        </button>
                    </div>
                ` : ''}
            </div>
        `;
    }

    setupDatasetCardEvents(datasetId) {
        const card = this.container.querySelector(`[data-dataset-id="${datasetId}"]`);

        // 卡片点击事件
        card.addEventListener('click', (e) => {
            // 如果点击的是按钮或复选框，不触发卡片选择
            if (e.target.closest('button') || e.target.closest('input[type="checkbox"]')) {
                return;
            }

            this.selectDataset(datasetId);
        });

        // 复选框事件
        const checkbox = card.querySelector('.dataset-checkbox');
        if (checkbox) {
            checkbox.addEventListener('change', (e) => {
                e.stopPropagation();
                this.toggleDatasetSelection(datasetId);
            });
        }

        // 按钮事件
        const viewBtn = card.querySelector('.btn-view-dataset');
        if (viewBtn) {
            viewBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.emit('viewDataset', datasetId);
            });
        }

        const processBtn = card.querySelector('.btn-process-dataset');
        if (processBtn) {
            processBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.emit('processDataset', datasetId);
            });
        }

        const exportBtn = card.querySelector('.btn-export-dataset');
        if (exportBtn) {
            exportBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.emit('exportDataset', datasetId);
            });
        }

        const deleteBtn = card.querySelector('.btn-delete-dataset');
        if (deleteBtn) {
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.deleteDataset(datasetId);
            });
        }
    }

    selectDataset(datasetId) {
        if (this.options.multiSelect) {
            this.toggleDatasetSelection(datasetId);
        } else {
            this.selectedDatasets = [datasetId];
            this.updateSelectionUI();
            this.emit('datasetSelected', datasetId);
        }
    }

    toggleDatasetSelection(datasetId) {
        const index = this.selectedDatasets.indexOf(datasetId);
        if (index > -1) {
            this.selectedDatasets.splice(index, 1);
        } else {
            this.selectedDatasets.push(datasetId);
        }

        this.updateSelectionUI();
        this.emit('selectionChanged', this.selectedDatasets);
    }

    updateSelectionUI() {
        // 更新卡片选中状态
        this.datasets.forEach(dataset => {
            const card = this.container.querySelector(`[data-dataset-id="${dataset.dataset_id}"]`);
            const checkbox = card.querySelector('.dataset-checkbox');

            if (this.selectedDatasets.includes(dataset.dataset_id)) {
                card.classList.add('selected');
                if (checkbox) checkbox.checked = true;
            } else {
                card.classList.remove('selected');
                if (checkbox) checkbox.checked = false;
            }
        });

        // 更新批量操作面板
        if (this.options.multiSelect) {
            const batchActions = this.container.querySelector('.batch-actions');
            const selectedCount = this.container.querySelector('#selected-count');

            if (this.selectedDatasets.length > 0) {
                batchActions.style.display = 'block';
                selectedCount.textContent = this.selectedDatasets.length;
            } else {
                batchActions.style.display = 'none';
            }
        }
    }

    clearSelection() {
        this.selectedDatasets = [];
        this.updateSelectionUI();
        this.emit('selectionChanged', this.selectedDatasets);
    }

    filterDatasets(searchTerm) {
        const cards = this.container.querySelectorAll('.dataset-card');
        const term = searchTerm.toLowerCase();

        cards.forEach(card => {
            const text = card.textContent.toLowerCase();
            card.style.display = text.includes(term) ? 'block' : 'none';
        });
    }

    sortDatasets(sortBy) {
        let sortedDatasets = [...this.datasets];

        switch (sortBy) {
            case 'created_desc':
                sortedDatasets.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
                break;
            case 'created_asc':
                sortedDatasets.sort((a, b) => new Date(a.created_at) - new Date(b.created_at));
                break;
            case 'name_asc':
                sortedDatasets.sort((a, b) => (a.filename || '').localeCompare(b.filename || ''));
                break;
            case 'name_desc':
                sortedDatasets.sort((a, b) => (b.filename || '').localeCompare(a.filename || ''));
                break;
            case 'samples_desc':
                sortedDatasets.sort((a, b) => (b.total_rows || 0) - (a.total_rows || 0));
                break;
            case 'samples_asc':
                sortedDatasets.sort((a, b) => (a.total_rows || 0) - (b.total_rows || 0));
                break;
        }

        this.datasets = sortedDatasets;
        this.renderDatasets();
    }

    updateStats() {
        const total = this.datasets.length;
        const ready = this.datasets.filter(d => d.processing_status === 'uploaded' || d.processing_status === 'ready').length;
        const processing = this.datasets.filter(d => d.processing_status === 'processing').length;
        const error = this.datasets.filter(d => d.processing_status === 'error').length;

        this.container.querySelector('#total-datasets').textContent = total;
        this.container.querySelector('#ready-datasets').textContent = ready;
        this.container.querySelector('#processing-datasets').textContent = processing;
        this.container.querySelector('#error-datasets').textContent = error;
    }

    async deleteDataset(datasetId) {
        if (!confirm('确定要删除这个数据集吗？此操作不可恢复。')) {
            return;
        }

        try {
            const response = await this.apiService.deleteDataset(datasetId);

            // 后端API返回的数据结构是 {success: true, message: "...", deleted_at: "..."}
            if (response.success) {
                this.datasets = this.datasets.filter(d => d.dataset_id !== datasetId);
                this.selectedDatasets = this.selectedDatasets.filter(id => id !== datasetId);
                this.renderDatasets();
                this.updateStats();
                this.updateSelectionUI();
                this.emit('datasetDeleted', datasetId);
                this.showSuccess('数据集删除成功');
            } else {
                this.showError('删除失败: ' + (response.message || response.detail || '未知错误'));
            }
        } catch (error) {
            const errorMessage = this.apiService.handleApiError(error);
            this.showError('删除失败: ' + errorMessage);
        }
    }

    async deleteSelectedDatasets() {
        if (this.selectedDatasets.length === 0) {
            return;
        }

        if (!confirm(`确定要删除选中的 ${this.selectedDatasets.length} 个数据集吗？此操作不可恢复。`)) {
            return;
        }

        try {
            const promises = this.selectedDatasets.map(id => this.apiService.deleteDataset(id));
            const results = await Promise.all(promises);

            const successCount = results.filter(r => r.success).length;
            const failedCount = results.length - successCount;

            // 移除成功删除的数据集
            this.datasets = this.datasets.filter(d => !this.selectedDatasets.includes(d.dataset_id));
            this.selectedDatasets = [];
            this.renderDatasets();
            this.updateStats();
            this.updateSelectionUI();

            if (failedCount === 0) {
                this.showSuccess(`成功删除 ${successCount} 个数据集`);
            } else {
                this.showWarning(`成功删除 ${successCount} 个，失败 ${failedCount} 个`);
            }

            this.emit('datasetsDeleted', { successCount, failedCount });
        } catch (error) {
            const errorMessage = this.apiService.handleApiError(error);
            this.showError('批量删除失败: ' + errorMessage);
        }
    }

    startAutoRefresh() {
        this.refreshTimer = setInterval(() => {
            this.loadDatasets();
        }, this.options.refreshInterval);
    }

    stopAutoRefresh() {
        if (this.refreshTimer) {
            clearInterval(this.refreshTimer);
            this.refreshTimer = null;
        }
    }

    showLoading() {
        const listContainer = this.container.querySelector('#dataset-list');
        listContainer.innerHTML = `
            <div class="text-center text-muted">
                <div class="loading"></div>
                正在加载数据集...
            </div>
        `;
    }

    showError(message) {
        this.emit('error', message);
        // 可以在这里添加UI错误显示
    }

    showSuccess(message) {
        this.emit('success', message);
        // 可以在这里添加UI成功显示
    }

    showWarning(message) {
        this.emit('warning', message);
        // 可以在这里添加UI警告显示
    }

    getStatusClass(status) {
        const statusMap = {
            'ready': 'status-ready',
            'processing': 'status-processing',
            'error': 'status-error',
            'pending': 'status-warning'
        };
        return statusMap[status] || 'status-warning';
    }

    getStatusText(status) {
        const statusMap = {
            'ready': '就绪',
            'processing': '处理中',
            'error': '错误',
            'pending': '等待中'
        };
        return statusMap[status] || '未知';
    }

    formatDate(dateString) {
        if (!dateString) return 'N/A';
        return new Date(dateString).toLocaleString();
    }

    formatFileSize(bytes) {
        if (!bytes || bytes === 0) return 'N/A';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    formatDataQuality(score) {
        if (score === undefined || score === null) return 'N/A';
        const percentage = Math.round(score * 100);
        return `${percentage}%`;
    }

    // 事件处理
    on(event, handler) {
        if (!this.eventHandlers[event]) {
            this.eventHandlers[event] = [];
        }
        this.eventHandlers[event].push(handler);
    }

    emit(event, data) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event].forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`Event handler error for event '${event}':`, error);
                }
            });
        }
    }

    // 公共方法
    getSelectedDatasets() {
        return this.selectedDatasets;
    }

    getDatasets() {
        return this.datasets;
    }

    refresh() {
        this.loadDatasets();
    }

    destroy() {
        this.stopAutoRefresh();
        this.container.innerHTML = '';
        this.eventHandlers = {};
    }
}

export default DatasetSelectorComponent;