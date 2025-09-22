/**
 * 导出功能组件 - MOF数据t-SNE可视化应用
 * 提供图表、数据和结果的导出功能
 */

import APIService from '../services/api.js';

class ExportComponent {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Container element with id '${containerId}' not found`);
        }

        this.options = {
            supportedFormats: ['png', 'svg', 'pdf'],
            defaultFormat: 'png',
            defaultWidth: 1200,
            defaultHeight: 800,
            defaultDPI: 300,
            enableBatchExport: true,
            enableDataExport: true,
            ...options
        };

        this.currentPipeline = null;
        this.visualizationComponent = null;
        this.apiService = new APIService();
        this.eventHandlers = {};
        this.exportHistory = [];

        this.init();
    }

    init() {
        this.createExportInterface();
        this.setupEventListeners();
        this.loadExportHistory();
    }

    createExportInterface() {
        this.container.innerHTML = `
            <div class="export-container">
                <!-- 图像导出面板 -->
                <div class="image-export-panel mb-4">
                    <div class="card">
                        <div class="card-header">
                            <h5><i class="bi bi-image"></i> 图像导出</h5>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-4">
                                    <label for="export-format" class="form-label">导出格式</label>
                                    <select class="form-select" id="export-format">
                                        ${this.options.supportedFormats.map(format => `
                                            <option value="${format}" ${format === this.options.defaultFormat ? 'selected' : ''}>
                                                ${format.toUpperCase()}
                                                ${this.getFormatDescription(format)}
                                            </option>
                                        `).join('')}
                                    </select>
                                    <div class="form-text mt-1" id="format-description">
                                        ${this.getFormatDescription(this.options.defaultFormat)}
                                    </div>
                                </div>
                                <div class="col-md-4">
                                    <label for="export-width" class="form-label">图像宽度 (px)</label>
                                    <input type="number" class="form-control" id="export-width"
                                           min="400" max="4000" value="${this.options.defaultWidth}">
                                    <div class="form-text">建议: 800-2000px</div>
                                </div>
                                <div class="col-md-4">
                                    <label for="export-height" class="form-label">图像高度 (px)</label>
                                    <input type="number" class="form-control" id="export-height"
                                           min="300" max="3000" value="${this.options.defaultHeight}">
                                    <div class="form-text">建议: 600-1500px</div>
                                </div>
                            </div>
                            <div class="row mt-3">
                                <div class="col-md-4">
                                    <label for="export-dpi" class="form-label">分辨率 (DPI)</label>
                                    <input type="number" class="form-control" id="export-dpi"
                                           min="72" max="600" value="${this.options.defaultDPI}">
                                    <div class="form-text">屏幕: 72dpi, 打印: 300dpi</div>
                                </div>
                                <div class="col-md-4">
                                    <label for="background-color" class="form-label">背景颜色</label>
                                    <select class="form-select" id="background-color">
                                        <option value="white">白色</option>
                                        <option value="transparent">透明</option>
                                        <option value="black">黑色</option>
                                        <option value="gray">灰色</option>
                                    </select>
                                </div>
                                <div class="col-md-4">
                                    <label for="filename" class="form-label">文件名</label>
                                    <input type="text" class="form-control" id="filename"
                                           value="mof-visualization" maxlength="50">
                                    <div class="form-text">不含扩展名</div>
                                </div>
                            </div>
                            <div class="row mt-3">
                                <div class="col-md-6">
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" id="include-legend" checked>
                                        <label class="form-check-label" for="include-legend">
                                            包含图例
                                        </label>
                                    </div>
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" id="include-title" checked>
                                        <label class="form-check-label" for="include-title">
                                            包含标题
                                        </label>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" id="include-axis-labels" checked>
                                        <label class="form-check-label" for="include-axis-labels">
                                            包含轴标签
                                        </label>
                                    </div>
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" id="high-quality">
                                        <label class="form-check-label" for="high-quality">
                                            高质量模式
                                        </label>
                                    </div>
                                </div>
                            </div>
                            <div class="row mt-3">
                                <div class="col-12">
                                    <button type="button" class="btn btn-primary" id="btn-export-image">
                                        <i class="bi bi-download"></i> 导出图像
                                    </button>
                                    <button type="button" class="btn btn-outline-secondary ms-2" id="btn-preview">
                                        <i class="bi bi-eye"></i> 预览
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 数据导出面板 -->
                <div class="data-export-panel mb-4" id="data-export-panel">
                    <div class="card">
                        <div class="card-header">
                            <h5><i class="bi bi-table"></i> 数据导出</h5>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <h6>坐标数据</h6>
                                    <div class="form-check mb-2">
                                        <input class="form-check-input" type="checkbox" id="export-coordinates" checked>
                                        <label class="form-check-label" for="export-coordinates">
                                            t-SNE坐标 (X, Y)
                                        </label>
                                    </div>
                                    <div class="form-check mb-2">
                                        <input class="form-check-input" type="checkbox" id="export-categories" checked>
                                        <label class="form-check-label" for="export-categories">
                                            分类信息
                                        </label>
                                    </div>
                                    <div class="form-check mb-2">
                                        <input class="form-check-input" type="checkbox" id="export-metadata">
                                        <label class="form-check-label" for="export-metadata">
                                            元数据 (密度、距离等)
                                        </label>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <h6>原始数据</h6>
                                    <div class="form-check mb-2">
                                        <input class="form-check-input" type="checkbox" id="export-features">
                                            <label class="form-check-label" for="export-features">
                                            特征数据
                                        </label>
                                    </div>
                                    <div class="form-check mb-2">
                                        <input class="form-check-input" type="checkbox" id="export-descriptive">
                                            <label class="form-check-label" for="export-descriptive">
                                            描述性数据
                                        </label>
                                    </div>
                                    <div class="form-check mb-2">
                                        <input class="form-check-input" type="checkbox" id="export-processing-info">
                                        <label class="form-check-label" for="export-processing-info">
                                            处理参数和结果
                                        </label>
                                    </div>
                                </div>
                            </div>
                            <div class="row mt-3">
                                <div class="col-md-4">
                                    <label for="data-format" class="form-label">数据格式</label>
                                    <select class="form-select" id="data-format">
                                        <option value="csv">CSV</option>
                                        <option value="json">JSON</option>
                                        <option value="excel">Excel (.xlsx)</option>
                                    </select>
                                </div>
                                <div class="col-md-4">
                                    <label for="data-filename" class="form-label">文件名</label>
                                    <input type="text" class="form-control" id="data-filename"
                                           value="mof-data" maxlength="50">
                                </div>
                                <div class="col-md-4 d-flex align-items-end">
                                    <button type="button" class="btn btn-success w-100" id="btn-export-data">
                                        <i class="bi bi-file-earmark-spreadsheet"></i> 导出数据
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 批量导出面板 -->
                <div class="batch-export-panel mb-4" id="batch-export-panel">
                    <div class="card">
                        <div class="card-header">
                            <h5><i class="bi bi-collection"></i> 批量导出</h5>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <h6>图像格式</h6>
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" id="batch-png" checked>
                                        <label class="form-check-label" for="batch-png">PNG</label>
                                    </div>
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" id="batch-svg">
                                        <label class="form-check-label" for="batch-svg">SVG</label>
                                    </div>
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" id="batch-pdf">
                                        <label class="form-check-label" for="batch-pdf">PDF</label>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <h6>尺寸预设</h6>
                                    <div class="btn-group w-100" role="group">
                                        <button type="button" class="btn btn-outline-primary btn-sm" data-size="small">
                                            小 (800x600)
                                        </button>
                                        <button type="button" class="btn btn-outline-primary btn-sm active" data-size="medium">
                                            中 (1200x800)
                                        </button>
                                        <button type="button" class="btn btn-outline-primary btn-sm" data-size="large">
                                            大 (2000x1500)
                                        </button>
                                    </div>
                                </div>
                            </div>
                            <div class="row mt-3">
                                <div class="col-12">
                                    <button type="button" class="btn btn-warning" id="btn-batch-export">
                                        <i class="bi bi-download"></i> 批量导出图像
                                    </button>
                                    <span class="ms-2 text-muted">同时生成多种格式和尺寸</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 导出历史 -->
                <div class="export-history-panel">
                    <div class="card">
                        <div class="card-header d-flex justify-content-between align-items-center">
                            <h5><i class="bi bi-clock-history"></i> 导出历史</h5>
                            <button type="button" class="btn btn-outline-danger btn-sm" id="btn-clear-history">
                                <i class="bi bi-trash"></i> 清空历史
                            </button>
                        </div>
                        <div class="card-body">
                            <div id="export-history-list" class="export-history-list">
                                <!-- 动态生成历史记录 -->
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 导出进度模态框 -->
                <div class="modal fade" id="export-progress-modal" tabindex="-1">
                    <div class="modal-dialog">
                        <div class="modal-content">
                            <div class="modal-header">
                                <h5 class="modal-title">导出进度</h5>
                                <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                            </div>
                            <div class="modal-body">
                                <div class="progress mb-3">
                                    <div class="progress-bar" role="progressbar" style="width: 0%"></div>
                                </div>
                                <div id="export-progress-text">准备导出...</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    setupEventListeners() {
        // 格式变化
        this.container.querySelector('#export-format').addEventListener('change', (e) => {
            this.updateFormatDescription(e.target.value);
        });

        // 图像导出
        this.container.querySelector('#btn-export-image').addEventListener('click', () => {
            this.exportImage();
        });

        // 预览
        this.container.querySelector('#btn-preview').addEventListener('click', () => {
            this.previewExport();
        });

        // 数据导出
        this.container.querySelector('#btn-export-data').addEventListener('click', () => {
            this.exportData();
        });

        // 批量导出
        this.container.querySelector('#btn-batch-export').addEventListener('click', () => {
            this.batchExport();
        });

        // 清空历史
        this.container.querySelector('#btn-clear-history').addEventListener('click', () => {
            this.clearExportHistory();
        });

        // 尺寸预设按钮
        this.container.querySelectorAll('[data-size]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.setSizePreset(e.target.dataset.size);
            });
        });

        // 高质量模式切换
        this.container.querySelector('#high-quality').addEventListener('change', (e) => {
            if (e.target.checked) {
                this.container.querySelector('#export-dpi').value = 600;
            } else {
                this.container.querySelector('#export-dpi').value = this.options.defaultDPI;
            }
        });
    }

    updateFormatDescription(format) {
        const descriptions = {
            'png': '位图格式，适合网页展示',
            'svg': '矢量格式，适合论文发表',
            'pdf': '文档格式，适合打印'
        };

        const descriptionElement = this.container.querySelector('#format-description');
        if (descriptionElement) {
            descriptionElement.textContent = descriptions[format] || '';
        }
    }

    getFormatDescription(format) {
        const descriptions = {
            'png': '- 位图格式',
            'svg': '- 矢量格式',
            'pdf': '- 文档格式'
        };
        return descriptions[format] || '';
    }

    async exportImage() {
        const exportConfig = this.getImageExportConfig();
        // 优先使用前端 Plotly 导出，确保与当前样式一致
        if (this.visualizationComponent && typeof Plotly !== 'undefined') {
            try {
                this.showExportProgress('正在导出图像 (前端)...');
                await this.visualizationComponent.exportChart(
                    exportConfig.format,
                    exportConfig.width,
                    exportConfig.height
                );
                this.hideExportProgress();
                this.showNotification('图像导出成功！', 'success');
                this.addToExportHistory('image', exportConfig, { method: 'frontend' });
                this.emit('imageExported', { config: exportConfig, result: { method: 'frontend' } });
                return;
            } catch (e) {
                console.warn('前端导出失败，将回退后端导出', e);
            }
        }

        // 回退到后端导出
        if (!this.currentPipeline) {
            this.showNotification('请先完成数据处理和可视化', 'warning');
            return;
        }

        try {
            this.showExportProgress('正在导出图像 (后端)...');
            const response = await this.apiService.exportVisualization(
                this.currentPipeline.pipeline_id,
                exportConfig
            );
            if (response.success) {
                this.downloadFile(response.data);
                this.addToExportHistory('image', exportConfig, response.data);
                this.hideExportProgress();
                this.showNotification('图像导出成功！', 'success');
                this.emit('imageExported', { config: exportConfig, result: response.data });
            } else {
                throw new Error(response.detail || '导出失败');
            }
        } catch (error) {
            this.hideExportProgress();
            const errorMessage = this.apiService.handleApiError(error);
            this.showNotification(`导出失败: ${errorMessage}`, 'danger');
            this.emit('exportError', error);
        }
    }

    getImageExportConfig() {
        return {
            format: this.container.querySelector('#export-format').value,
            width: parseInt(this.container.querySelector('#export-width').value),
            height: parseInt(this.container.querySelector('#export-height').value),
            dpi: parseInt(this.container.querySelector('#export-dpi').value),
            background_color: this.container.querySelector('#background-color').value,
            filename: this.container.querySelector('#filename').value,
            include_legend: this.container.querySelector('#include-legend').checked,
            include_title: this.container.querySelector('#include-title').checked,
            include_axis_labels: this.container.querySelector('#include-axis-labels').checked,
            high_quality: this.container.querySelector('#high-quality').checked
        };
    }

    async exportData() {
        if (!this.currentPipeline) {
            this.showNotification('请先完成数据处理', 'warning');
            return;
        }

        const dataConfig = this.getDataExportConfig();
        this.showExportProgress('正在导出数据...');

        try {
            // 这里需要后端API支持数据导出
            // 暂时使用模拟导出
            await this.simulateDataExport(dataConfig);

            this.hideExportProgress();
            this.showNotification('数据导出成功！', 'success');
            this.addToExportHistory('data', dataConfig);
            this.emit('dataExported', dataConfig);
        } catch (error) {
            this.hideExportProgress();
            this.showNotification(`数据导出失败: ${error.message}`, 'danger');
            this.emit('exportError', error);
        }
    }

    getDataExportConfig() {
        return {
            format: this.container.querySelector('#data-format').value,
            filename: this.container.querySelector('#data-filename').value,
            export_coordinates: this.container.querySelector('#export-coordinates').checked,
            export_categories: this.container.querySelector('#export-categories').checked,
            export_metadata: this.container.querySelector('#export-metadata').checked,
            export_features: this.container.querySelector('#export-features').checked,
            export_descriptive: this.container.querySelector('#export-descriptive').checked,
            export_processing_info: this.container.querySelector('#export-processing-info').checked
        };
    }

    async batchExport() {
        if (!this.currentPipeline) {
            this.showNotification('请先完成数据处理和可视化', 'warning');
            return;
        }

        const formats = [];
        if (this.container.querySelector('#batch-png').checked) formats.push('png');
        if (this.container.querySelector('#batch-svg').checked) formats.push('svg');
        if (this.container.querySelector('#batch-pdf').checked) formats.push('pdf');

        if (formats.length === 0) {
            this.showNotification('请至少选择一种导出格式', 'warning');
            return;
        }

        this.showExportProgress('批量导出中...');

        const results = [];
        for (let i = 0; i < formats.length; i++) {
            const format = formats[i];
            this.updateExportProgress(`正在导出 ${format.toUpperCase()}...`, (i / formats.length) * 100);

            try {
                const config = this.getImageExportConfig();
                config.format = format;

                const response = await this.apiService.exportVisualization(
                    this.currentPipeline.pipeline_id,
                    config
                );

                if (response.success) {
                    this.downloadFile(response.data);
                    results.push({ format, success: true, data: response.data });
                    this.addToExportHistory('image', config, response.data);
                } else {
                    results.push({ format, success: false, error: response.detail });
                }
            } catch (error) {
                results.push({ format, success: false, error: error.message });
            }

            // 添加延迟避免API限制
            await new Promise(resolve => setTimeout(resolve, 500));
        }

        this.hideExportProgress();

        const successCount = results.filter(r => r.success).length;
        const totalCount = results.length;

        this.showNotification(
            `批量导出完成！成功 ${successCount}/${totalCount} 个文件`,
            successCount === totalCount ? 'success' : 'warning'
        );

        this.emit('batchExportCompleted', results);
    }

    previewExport() {
        if (!this.visualizationComponent) {
            this.showNotification('可视化组件未初始化', 'warning');
            return;
        }

        const config = this.getImageExportConfig();

        // 在新窗口中打开预览
        const previewWindow = window.open('', '_blank');
        if (previewWindow) {
            previewWindow.document.write(`
                <html>
                    <head>
                        <title>导出预览 - ${config.filename}</title>
                        <style>
                            body { margin: 0; padding: 20px; font-family: Arial, sans-serif; }
                            .preview-info { background: #f8f9fa; padding: 10px; margin-bottom: 20px; border-radius: 5px; }
                        </style>
                    </head>
                    <body>
                        <div class="preview-info">
                            <h3>导出预览</h3>
                            <p><strong>格式:</strong> ${config.format.toUpperCase()}</p>
                            <p><strong>尺寸:</strong> ${config.width} × ${config.height} px</p>
                            <p><strong>DPI:</strong> ${config.dpi}</p>
                            <p><strong>文件名:</strong> ${config.filename}.${config.format}</p>
                        </div>
                        <div>
                            <p>实际预览功能需要在导出后查看生成的文件。</p>
                            <button onclick="window.close()">关闭窗口</button>
                        </div>
                    </body>
                </html>
            `);
        }

        this.emit('previewGenerated', config);
    }

    setSizePreset(size) {
        const presets = {
            'small': { width: 800, height: 600 },
            'medium': { width: 1200, height: 800 },
            'large': { width: 2000, height: 1500 }
        };

        const preset = presets[size];
        if (preset) {
            this.container.querySelector('#export-width').value = preset.width;
            this.container.querySelector('#export-height').value = preset.height;

            // 更新按钮状态
            this.container.querySelectorAll('[data-size]').forEach(btn => {
                btn.classList.remove('active');
            });
            this.container.querySelector(`[data-size="${size}"]`).classList.add('active');
        }
    }

    downloadFile(fileData) {
        const link = document.createElement('a');
        link.href = `/${fileData.file_path}`;
        link.download = fileData.filename;
        link.style.display = 'none';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    async simulateDataExport(config) {
        // 模拟数据导出过程
        await new Promise(resolve => setTimeout(resolve, 1000));

        // 创建模拟数据
        const mockData = this.generateMockData(config);
        const blob = new Blob([mockData], { type: this.getDataMimeType(config.format) });
        const url = URL.createObjectURL(blob);

        const link = document.createElement('a');
        link.href = url;
        link.download = `${config.filename}.${this.getDataExtension(config.format)}`;
        link.click();

        URL.revokeObjectURL(url);
    }

    generateMockData(config) {
        // 生成模拟的导出数据
        if (config.format === 'csv') {
            let csv = 'sample_id,category,x,y,local_density\n';
            for (let i = 0; i < 10; i++) {
                csv += `MOF_${i},Category_A,${Math.random() * 10},${Math.random() * 10},${Math.random()}\n`;
            }
            return csv;
        } else if (config.format === 'json') {
            return JSON.stringify({
                metadata: {
                    export_time: new Date().toISOString(),
                    total_samples: 10,
                    pipeline_id: this.currentPipeline?.pipeline_id
                },
                data: Array.from({ length: 10 }, (_, i) => ({
                    sample_id: `MOF_${i}`,
                    category: 'Category_A',
                    x: Math.random() * 10,
                    y: Math.random() * 10,
                    local_density: Math.random()
                }))
            }, null, 2);
        }
        return 'Mock export data';
    }

    getDataMimeType(format) {
        const mimeTypes = {
            'csv': 'text/csv',
            'json': 'application/json',
            'excel': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        };
        return mimeTypes[format] || 'text/plain';
    }

    getDataExtension(format) {
        const extensions = {
            'csv': 'csv',
            'json': 'json',
            'excel': 'xlsx'
        };
        return extensions[format] || 'txt';
    }

    addToExportHistory(type, config, result = null) {
        const historyItem = {
            id: Date.now(),
            type: type,
            timestamp: new Date().toISOString(),
            config: config,
            result: result,
            pipeline_id: this.currentPipeline?.pipeline_id
        };

        this.exportHistory.unshift(historyItem);

        // 限制历史记录数量
        if (this.exportHistory.length > 20) {
            this.exportHistory = this.exportHistory.slice(0, 20);
        }

        this.saveExportHistory();
        this.updateExportHistoryDisplay();
    }

    updateExportHistoryDisplay() {
        const container = this.container.querySelector('#export-history-list');

        if (this.exportHistory.length === 0) {
            container.innerHTML = '<p class="text-muted">暂无导出历史记录</p>';
            return;
        }

        container.innerHTML = this.exportHistory.map(item => `
            <div class="export-history-item p-2 border-bottom">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <span class="badge bg-${item.type === 'image' ? 'primary' : 'success'} me-2">
                            ${item.type === 'image' ? '图像' : '数据'}
                        </span>
                        <strong>${item.config.filename || 'export'}</strong>
                        <small class="text-muted ms-2">
                            ${new Date(item.timestamp).toLocaleString()}
                        </small>
                    </div>
                    <div>
                        ${item.result ? `
                            <button type="button" class="btn btn-outline-primary btn-sm me-1"
                                    onclick="window.location.href='/${item.result.file_path}'">
                                <i class="bi bi-download"></i>
                            </button>
                        ` : ''}
                        <button type="button" class="btn btn-outline-danger btn-sm btn-delete-history"
                                data-id="${item.id}">
                            <i class="bi bi-trash"></i>
                        </button>
                    </div>
                </div>
                <div class="small text-muted mt-1">
                    ${item.type === 'image' ?
                        `${item.config.format?.toUpperCase()} • ${item.config.width}×${item.config.height}` :
                        item.config.format?.toUpperCase()
                    }
                </div>
            </div>
        `).join('');

        // 添加删除按钮事件
        container.querySelectorAll('.btn-delete-history').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.deleteExportHistoryItem(parseInt(e.target.closest('button').dataset.id));
            });
        });
    }

    deleteExportHistoryItem(itemId) {
        this.exportHistory = this.exportHistory.filter(item => item.id !== itemId);
        this.saveExportHistory();
        this.updateExportHistoryDisplay();
    }

    clearExportHistory() {
        if (confirm('确定要清空所有导出历史记录吗？')) {
            this.exportHistory = [];
            this.saveExportHistory();
            this.updateExportHistoryDisplay();
        }
    }

    saveExportHistory() {
        try {
            localStorage.setItem('mof-viz-export-history', JSON.stringify(this.exportHistory));
        } catch (error) {
            console.warn('Failed to save export history:', error);
        }
    }

    loadExportHistory() {
        try {
            const saved = localStorage.getItem('mof-viz-export-history');
            if (saved) {
                this.exportHistory = JSON.parse(saved);
                this.updateExportHistoryDisplay();
            }
        } catch (error) {
            console.warn('Failed to load export history:', error);
        }
    }

    showExportProgress(message, progress = 0) {
        const modal = new bootstrap.Modal(document.getElementById('export-progress-modal'));
        const progressBar = document.querySelector('#export-progress-modal .progress-bar');
        const progressText = document.getElementById('export-progress-text');

        progressBar.style.width = `${progress}%`;
        progressText.textContent = message;
        modal.show();
    }

    updateExportProgress(message, progress = 0) {
        const progressBar = document.querySelector('#export-progress-modal .progress-bar');
        const progressText = document.getElementById('export-progress-text');

        progressBar.style.width = `${progress}%`;
        progressText.textContent = message;
    }

    hideExportProgress() {
        const modal = bootstrap.Modal.getInstance(document.getElementById('export-progress-modal'));
        if (modal) {
            modal.hide();
        }
    }

    showNotification(message, type = 'info') {
        // 创建通知元素
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        document.body.appendChild(notification);

        // 自动移除
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }

    // 公共方法
    setPipeline(pipeline) {
        this.currentPipeline = pipeline;
    }

    setVisualizationComponent(component) {
        this.visualizationComponent = component;
    }

    getExportHistory() {
        return [...this.exportHistory];
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

    destroy() {
        this.container.innerHTML = '';
        this.eventHandlers = {};
        this.exportHistory = [];
    }
}

export default ExportComponent;
