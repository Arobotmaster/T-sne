/**
 * 文件上传组件 - MOF数据t-SNE可视化应用
 * 处理CSV文件上传和验证
 */

import APIService from '../services/api.js';

class UploadComponent {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Container element with id '${containerId}' not found`);
        }

        this.options = {
            maxFileSize: 100 * 1024 * 1024, // 100MB
            allowedTypes: ['text/csv', 'application/csv'],
            acceptedExtensions: ['.csv'],
            autoProcess: true,
            ...options
        };

        this.currentFile = null;
        this.currentDataset = null;
        this.apiService = new APIService();
        this.eventHandlers = {};

        this.init();
    }

    init() {
        this.createUploadInterface();
        this.setupEventListeners();
        this.checkBackendConnection();
    }

    createUploadInterface() {
        this.container.innerHTML = `
            <div class="upload-container">
                <!-- 拖拽上传区域 -->
                <div class="upload-area" id="upload-area">
                    <div class="upload-content">
                        <i class="bi bi-cloud-upload upload-icon"></i>
                        <h4>拖拽CSV文件到此处</h4>
                        <p class="upload-text">或者点击选择文件</p>
                        <button type="button" class="btn btn-primary btn-select-file">
                            <i class="bi bi-folder2-open"></i> 选择文件
                        </button>
                    </div>
                </div>

                <!-- 隐藏的文件输入 -->
                <input type="file" id="file-input" accept=".csv" style="display: none;">

                <!-- 文件信息显示 -->
                <div id="file-info" class="file-info" style="display: none;">
                    <div class="card">
                        <div class="card-header">
                            <h5><i class="bi bi-file-earmark-text"></i> 文件信息</h5>
                        </div>
                        <div class="card-body">
                            <div class="file-details">
                                <div class="row">
                                    <div class="col-md-6">
                                        <strong>文件名:</strong> <span id="file-name"></span>
                                    </div>
                                    <div class="col-md-6">
                                        <strong>文件大小:</strong> <span id="file-size"></span>
                                    </div>
                                </div>
                                <div class="row mt-2">
                                    <div class="col-md-6">
                                        <strong>文件类型:</strong> <span id="file-type"></span>
                                    </div>
                                    <div class="col-md-6">
                                        <strong>最后修改:</strong> <span id="file-modified"></span>
                                    </div>
                                </div>
                            </div>
                            <div class="mt-3">
                                <button type="button" class="btn btn-success btn-upload" id="btn-upload">
                                    <i class="bi bi-cloud-upload"></i> 开始上传
                                </button>
                                <button type="button" class="btn btn-secondary btn-cancel" id="btn-cancel">
                                    <i class="bi bi-x-circle"></i> 取消
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 上传进度 -->
                <div id="upload-progress" class="upload-progress" style="display: none;">
                    <div class="card">
                        <div class="card-header">
                            <h5><i class="bi bi-arrow-repeat"></i> 上传进度</h5>
                        </div>
                        <div class="card-body">
                            <div class="progress">
                                <div class="progress-bar" role="progressbar" style="width: 0%"></div>
                            </div>
                            <div class="mt-2 text-center">
                                <span id="progress-text">准备上传...</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 上传结果 -->
                <div id="upload-result" class="upload-result" style="display: none;"></div>
            </div>
        `;
    }

    setupEventListeners() {
        const uploadArea = this.container.querySelector('#upload-area');
        const fileInput = this.container.querySelector('#file-input');
        const btnSelectFile = this.container.querySelector('.btn-select-file');
        const btnUpload = this.container.querySelector('#btn-upload');
        const btnCancel = this.container.querySelector('#btn-cancel');

        // 点击选择文件（阻止冒泡，避免触发上传区域 click 导致二次弹窗）
        btnSelectFile.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.click();
        });

        // 文件选择变化
        fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

        // 拖拽事件
        uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        uploadArea.addEventListener('drop', (e) => this.handleDrop(e));

        // 按钮事件
        btnUpload.addEventListener('click', () => this.startUpload());
        btnCancel.addEventListener('click', () => this.reset());

        // 点击上传区域也可以选择文件，但排除按钮点击以避免重复触发
        uploadArea.addEventListener('click', (e) => {
            if (e.target.closest('.btn-select-file')) return;
            fileInput.click();
        });
    }

    async checkBackendConnection() {
        try {
            await this.apiService.checkHealth();
            this.showStatus('info', '✅ 后端服务连接正常');
        } catch (error) {
            this.showStatus('warning', '⚠️ 后端服务连接异常，请检查服务是否启动');
        }
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            this.validateAndSetFile(file);
        }
    }

    handleDragOver(event) {
        event.preventDefault();
        event.stopPropagation();
        const uploadArea = this.container.querySelector('#upload-area');
        uploadArea.classList.add('dragover');
    }

    handleDragLeave(event) {
        event.preventDefault();
        event.stopPropagation();
        const uploadArea = this.container.querySelector('#upload-area');
        uploadArea.classList.remove('dragover');
    }

    handleDrop(event) {
        event.preventDefault();
        event.stopPropagation();
        const uploadArea = this.container.querySelector('#upload-area');
        uploadArea.classList.remove('dragover');

        const files = event.dataTransfer.files;
        if (files.length > 0) {
            this.validateAndSetFile(files[0]);
        }
    }

    validateAndSetFile(file) {
        // 验证文件扩展名
        const fileName = file.name.toLowerCase();
        const hasValidExtension = this.options.acceptedExtensions.some(ext => fileName.endsWith(ext));

        if (!hasValidExtension) {
            this.showStatus('error', `❌ 不支持的文件格式，仅支持: ${this.options.acceptedExtensions.join(', ')}`);
            return;
        }

        // 验证文件大小
        if (file.size > this.options.maxFileSize) {
            this.showStatus('error', `❌ 文件过大，最大支持 ${this.formatFileSize(this.options.maxFileSize)}`);
            return;
        }

        // 验证文件类型（如果浏览器提供了）
        if (file.type && !this.options.allowedTypes.includes(file.type)) {
            // 有些浏览器可能无法正确识别CSV文件类型，所以这里只作为警告
            console.warn('Unrecognized file type:', file.type);
        }

        this.currentFile = file;
        this.displayFileInfo();
        this.emit('fileSelected', file);
    }

    displayFileInfo() {
        const fileInfo = this.container.querySelector('#file-info');
        const fileName = this.container.querySelector('#file-name');
        const fileSize = this.container.querySelector('#file-size');
        const fileType = this.container.querySelector('#file-type');
        const fileModified = this.container.querySelector('#file-modified');

        fileName.textContent = this.currentFile.name;
        fileSize.textContent = this.formatFileSize(this.currentFile.size);
        fileType.textContent = this.currentFile.type || '未知';
        fileModified.textContent = new Date(this.currentFile.lastModified).toLocaleString();

        fileInfo.style.display = 'block';
        this.container.querySelector('#upload-area').style.display = 'none';
    }

    async startUpload() {
        if (!this.currentFile) {
            this.showStatus('error', '❌ 请先选择文件');
            return;
        }

        this.showProgress();
        this.emit('uploadStart', this.currentFile);

        try {
            const result = await this.apiService.uploadFile(this.currentFile);

            if (result.success) {
                this.currentDataset = result.data;
                this.showUploadSuccess(result.data);
                this.emit('uploadSuccess', result.data);

                if (this.options.autoProcess) {
                    this.emit('autoProcess', result.data);
                }
            } else {
                this.showStatus('error', `❌ 上传失败: ${result.detail || '未知错误'}`);
                this.emit('uploadError', new Error(result.detail || '未知错误'));
            }
        } catch (error) {
            const errorMessage = this.apiService.handleApiError(error);
            this.showStatus('error', `❌ ${errorMessage}`);
            this.emit('uploadError', error);
        }
    }

    showProgress() {
        const progressContainer = this.container.querySelector('#upload-progress');
        const progressBar = progressContainer.querySelector('.progress-bar');
        const progressText = this.container.querySelector('#progress-text');

        progressContainer.style.display = 'block';
        progressBar.style.width = '0%';
        progressText.textContent = '正在上传...';

        // 模拟上传进度
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 90) progress = 90;

            progressBar.style.width = `${progress}%`;
            progressText.textContent = `上传中... ${Math.round(progress)}%`;

            if (progress >= 90) {
                clearInterval(interval);
            }
        }, 200);
    }

    showUploadSuccess(datasetInfo) {
        const resultContainer = this.container.querySelector('#upload-result');

        resultContainer.innerHTML = `
            <div class="alert alert-success">
                <h5><i class="bi bi-check-circle"></i> 上传成功！</h5>
                <div class="upload-details">
                    <div class="row">
                        <div class="col-md-6">
                            <strong>数据集ID:</strong> ${datasetInfo.dataset_id}
                        </div>
                        <div class="col-md-6">
                            <strong>文件名:</strong> ${datasetInfo.filename}
                        </div>
                    </div>
                    <div class="row mt-2">
                        <div class="col-md-6">
                            <strong>样本数量:</strong> ${datasetInfo.total_rows.toLocaleString()}
                        </div>
                        <div class="col-md-6">
                            <strong>特征数量:</strong> ${datasetInfo.total_columns}
                        </div>
                    </div>
                    <div class="row mt-2">
                        <div class="col-md-6">
                            <strong>数据质量:</strong> ${(datasetInfo.data_quality_score * 100).toFixed(1)}%
                        </div>
                        <div class="col-md-6">
                            <strong>处理时间:</strong> ${datasetInfo.processing_time_ms}ms
                        </div>
                    </div>
                </div>
                <div class="mt-3">
                    <button type="button" class="btn btn-primary btn-process-data" id="btn-process-data">
                        <i class="bi bi-play-circle"></i> 开始数据处理
                    </button>
                    <button type="button" class="btn btn-secondary btn-reset" id="btn-reset">
                        <i class="bi bi-arrow-clockwise"></i> 上传新文件
                    </button>
                </div>
            </div>
        `;

        resultContainer.style.display = 'block';
        this.container.querySelector('#upload-progress').style.display = 'none';
        this.container.querySelector('#file-info').style.display = 'none';

        // 绑定处理数据按钮事件
        const btnProcessData = resultContainer.querySelector('#btn-process-data');
        const btnReset = resultContainer.querySelector('#btn-reset');

        btnProcessData.addEventListener('click', () => {
            this.emit('processData', datasetInfo);
        });

        btnReset.addEventListener('click', () => {
            this.reset();
        });
    }

    showStatus(type, message) {
        const existingAlert = this.container.querySelector('.upload-status-alert');
        if (existingAlert) {
            existingAlert.remove();
        }

        const alertClass = {
            'info': 'alert-info',
            'success': 'alert-success',
            'warning': 'alert-warning',
            'error': 'alert-danger'
        }[type] || 'alert-info';

        const alert = document.createElement('div');
        alert.className = `alert ${alertClass} upload-status-alert`;
        alert.innerHTML = message;

        this.container.insertBefore(alert, this.container.firstChild);

        // 自动隐藏成功和info消息
        if (type === 'success' || type === 'info') {
            setTimeout(() => {
                if (alert.parentNode) {
                    alert.remove();
                }
            }, 5000);
        }
    }

    reset() {
        this.currentFile = null;
        this.currentDataset = null;

        // 重置界面
        this.container.querySelector('#file-input').value = '';
        this.container.querySelector('#file-info').style.display = 'none';
        this.container.querySelector('#upload-progress').style.display = 'none';
        this.container.querySelector('#upload-result').style.display = 'none';
        this.container.querySelector('#upload-area').style.display = 'block';

        // 移除状态消息
        const statusAlert = this.container.querySelector('.upload-status-alert');
        if (statusAlert) {
            statusAlert.remove();
        }

        this.emit('reset');
    }

    // 工具方法
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
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
    getCurrentDataset() {
        return this.currentDataset;
    }

    getCurrentFile() {
        return this.currentFile;
    }

    destroy() {
        // 清理事件监听器
        this.container.innerHTML = '';
        this.eventHandlers = {};
    }
}

export default UploadComponent;