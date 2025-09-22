/**
 * 对比可视化组件 - MOF数据t-SNE可视化应用
 * 提供原始数据与筛选后数据的对比可视化功能
 */

import APIService from '../services/api.js';

class ComparisonComponent {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Container element with id '${containerId}' not found`);
        }

        this.options = {
            width: 1200,
            height: 600,
            enableSideBySide: true,
            enableOverlay: true,
            defaultOpacity: {
                original: 0.3,
                filtered: 1.0
            },
            ...options
        };

        this.currentPipeline = null;
        this.originalData = null;
        this.filteredData = null;
        this.comparisonConfig = null;
        this.apiService = new APIService();
        this.eventHandlers = {};
        this.selectedCategories = new Set();

        this.init();
    }

    init() {
        this.createComparisonInterface();
        this.setupEventListeners();
    }

    createComparisonInterface() {
        this.container.innerHTML = `
            <div class="comparison-container">
                <!-- 控制面板 -->
                <div class="control-panel mb-4">
                    <div class="card">
                        <div class="card-header">
                            <h5><i class="bi bi-layers"></i> 对比可视化控制</h5>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-4">
                                    <label class="form-label">对比模式</label>
                                    <div class="btn-group w-100" role="group">
                                        <button type="button" class="btn btn-outline-primary active" data-mode="side-by-side">
                                            并列显示
                                        </button>
                                        <button type="button" class="btn btn-outline-primary" data-mode="overlay">
                                            叠加显示
                                        </button>
                                        <button type="button" class="btn btn-outline-primary" data-mode="split">
                                            分屏显示
                                        </button>
                                    </div>
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label">显示内容</label>
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" id="show-original" checked>
                                        <label class="form-check-label" for="show-original">
                                            显示原始数据
                                        </label>
                                    </div>
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" id="show-filtered" checked>
                                        <label class="form-check-label" for="show-filtered">
                                            显示筛选数据
                                        </label>
                                    </div>
                                </div>
                                <div class="col-md-4">
                                    <label class="form-label">透明度</label>
                                    <div class="row">
                                        <div class="col-6">
                                            <label for="original-opacity" class="form-label small">原始</label>
                                            <input type="range" class="form-range" id="original-opacity"
                                                   min="0.1" max="1" step="0.1" value="0.3">
                                            <div class="text-center small" id="original-opacity-value">0.3</div>
                                        </div>
                                        <div class="col-6">
                                            <label for="filtered-opacity" class="form-label small">筛选</label>
                                            <input type="range" class="form-range" id="filtered-opacity"
                                                   min="0.1" max="1" step="0.1" value="1.0">
                                            <div class="text-center small" id="filtered-opacity-value">1.0</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 筛选条件面板 -->
                <div class="filter-panel mb-4">
                    <div class="card">
                        <div class="card-header">
                            <h5><i class="bi bi-funnel"></i> 数据筛选条件</h5>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <h6>分类筛选</h6>
                                    <div id="category-filters" class="category-filters">
                                        <!-- 动态生成分类筛选器 -->
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <h6>数值范围筛选</h6>
                                    <div class="mb-3">
                                        <label for="density-range" class="form-label">密度范围</label>
                                        <div class="row">
                                            <div class="col-6">
                                                <input type="number" class="form-control" id="density-min"
                                                       placeholder="最小值" step="0.01">
                                            </div>
                                            <div class="col-6">
                                                <input type="number" class="form-control" id="density-max"
                                                       placeholder="最大值" step="0.01">
                                            </div>
                                        </div>
                                    </div>
                                    <div class="mb-3">
                                        <label for="distance-range" class="form-label">到中心距离范围</label>
                                        <div class="row">
                                            <div class="col-6">
                                                <input type="number" class="form-control" id="distance-min"
                                                       placeholder="最小值" step="0.01">
                                            </div>
                                            <div class="col-6">
                                                <input type="number" class="form-control" id="distance-max"
                                                       placeholder="最大值" step="0.01">
                                            </div>
                                        </div>
                                    </div>
                                    <div class="mb-3">
                                        <label for="sample-count" class="form-label">样本数量限制</label>
                                        <input type="number" class="form-control" id="sample-count"
                                               placeholder="最大样本数" min="1">
                                    </div>
                                </div>
                            </div>
                            <div class="row mt-3">
                                <div class="col-12">
                                    <button type="button" class="btn btn-primary" id="btn-apply-filter">
                                        <i class="bi bi-check-circle"></i> 应用筛选
                                    </button>
                                    <button type="button" class="btn btn-outline-secondary ms-2" id="btn-reset-filter">
                                        <i class="bi bi-arrow-counterclockwise"></i> 重置筛选
                                    </button>
                                    <button type="button" class="btn btn-outline-success ms-2" id="btn-random-sample">
                                        <i class="bi bi-shuffle"></i> 随机采样
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 对比图表容器 -->
                <div class="comparison-charts">
                    <!-- 并列显示模式 -->
                    <div id="side-by-side-mode" class="comparison-mode">
                        <div class="row">
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-header">
                                        <h6><i class="bi bi-circle"></i> 原始数据
                                            <small class="text-muted">(${this.getSampleCountText('original')})</small>
                                        </h6>
                                    </div>
                                    <div class="card-body">
                                        <div id="original-chart" style="height: 500px;"></div>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-header">
                                        <h6><i class="bi bi-circle-fill"></i> 筛选后数据
                                            <small class="text-muted">(${this.getSampleCountText('filtered')})</small>
                                        </h6>
                                    </div>
                                    <div class="card-body">
                                        <div id="filtered-chart" style="height: 500px;"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- 叠加显示模式 -->
                    <div id="overlay-mode" class="comparison-mode" style="display: none;">
                        <div class="card">
                            <div class="card-header">
                                <h6><i class="bi bi-layers"></i> 叠加对比
                                    <small class="text-muted">原始数据 + 筛选数据</small>
                                </h6>
                            </div>
                            <div class="card-body">
                                <div id="overlay-chart" style="height: 600px;"></div>
                            </div>
                        </div>
                    </div>

                    <!-- 分屏显示模式 -->
                    <div id="split-mode" class="comparison-mode" style="display: none;">
                        <div class="card">
                            <div class="card-header">
                                <h6><i class="bi bi-layout-split"></i> 分屏对比
                                    <small class="text-muted">左半屏：原始数据 | 右半屏：筛选数据</small>
                                </h6>
                            </div>
                            <div class="card-body">
                                <div id="split-chart" style="height: 600px;"></div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 统计对比面板 -->
                <div class="statistics-panel mt-4">
                    <div class="card">
                        <div class="card-header">
                            <h5><i class="bi bi-bar-chart"></i> 统计对比</h5>
                        </div>
                        <div class="card-body">
                            <div id="comparison-statistics">
                                <!-- 动态生成统计对比 -->
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 导出选项 -->
                <div class="export-options mt-4">
                    <div class="card">
                        <div class="card-header">
                            <h5><i class="bi bi-download"></i> 导出对比结果</h5>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-3">
                                    <label for="comparison-format" class="form-label">导出格式</label>
                                    <select class="form-select" id="comparison-format">
                                        <option value="png">PNG图片</option>
                                        <option value="svg">SVG矢量图</option>
                                        <option value="pdf">PDF文档</option>
                                    </select>
                                </div>
                                <div class="col-md-3">
                                    <label for="comparison-width" class="form-label">宽度</label>
                                    <input type="number" class="form-control" id="comparison-width"
                                           value="1600" min="800" max="4000">
                                </div>
                                <div class="col-md-3">
                                    <label for="comparison-height" class="form-label">高度</label>
                                    <input type="number" class="form-control" id="comparison-height"
                                           value="800" min="400" max="2000">
                                </div>
                                <div class="col-md-3 d-flex align-items-end">
                                    <button type="button" class="btn btn-primary w-100" id="btn-export-comparison">
                                        <i class="bi bi-download"></i> 导出对比图
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    setupEventListeners() {
        // 对比模式切换
        this.container.querySelectorAll('[data-mode]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchComparisonMode(e.target.dataset.mode);
            });
        });

        // 透明度滑块
        this.container.querySelector('#original-opacity').addEventListener('input', (e) => {
            document.getElementById('original-opacity-value').textContent = e.target.value;
            this.updateChartOpacity();
        });

        this.container.querySelector('#filtered-opacity').addEventListener('input', (e) => {
            document.getElementById('filtered-opacity-value').textContent = e.target.value;
            this.updateChartOpacity();
        });

        // 显示控制
        this.container.querySelector('#show-original').addEventListener('change', () => {
            this.updateChartVisibility();
        });

        this.container.querySelector('#show-filtered').addEventListener('change', () => {
            this.updateChartVisibility();
        });

        // 筛选控制
        this.container.querySelector('#btn-apply-filter').addEventListener('click', () => {
            this.applyFilters();
        });

        this.container.querySelector('#btn-reset-filter').addEventListener('click', () => {
            this.resetFilters();
        });

        this.container.querySelector('#btn-random-sample').addEventListener('click', () => {
            this.randomSample();
        });

        // 导出控制
        this.container.querySelector('#btn-export-comparison').addEventListener('click', () => {
            this.exportComparison();
        });
    }

    async loadData(pipelineId) {
        try {
            this.showLoading('加载对比可视化数据...');

            // 获取原始可视化数据
            const response = await this.apiService.getVisualizationData(pipelineId);
            if (response.success) {
                this.originalData = response.data;
                this.currentPipeline = { pipeline_id: pipelineId };
                this.filteredData = this.cloneData(this.originalData);

                this.createCategoryFilters();
                this.updateComparisonCharts();
                this.updateStatistics();
                this.hideLoading();

                this.emit('dataLoaded', { original: this.originalData, filtered: this.filteredData });
            } else {
                throw new Error(response.detail || 'Failed to load visualization data');
            }
        } catch (error) {
            this.showError(`加载数据失败: ${error.message}`);
            this.emit('loadError', error);
        }
    }

    cloneData(data) {
        return JSON.parse(JSON.stringify(data));
    }

    createCategoryFilters() {
        if (!this.originalData || !this.originalData.categories) return;

        const container = this.container.querySelector('#category-filters');
        container.innerHTML = '';

        this.originalData.categories.forEach(category => {
            const filterDiv = document.createElement('div');
            filterDiv.className = 'form-check form-check-inline me-3';
            filterDiv.innerHTML = `
                <input class="form-check-input category-filter" type="checkbox"
                       id="cat-comp-${category.category_id}" value="${category.category_id}" checked>
                <label class="form-check-label" for="cat-comp-${category.category_id}">
                    <span class="category-indicator" style="background-color: ${category.color_code}; width: 12px; height: 12px; display: inline-block; border-radius: 50%; margin-right: 4px;"></span>
                    ${category.category_name}
                </label>
            `;
            container.appendChild(filterDiv);
        });
    }

    switchComparisonMode(mode) {
        // 更新按钮状态
        this.container.querySelectorAll('[data-mode]').forEach(btn => {
            btn.classList.remove('active');
        });
        this.container.querySelector(`[data-mode="${mode}"]`).classList.add('active');

        // 隐藏所有模式
        this.container.querySelectorAll('.comparison-mode').forEach(el => {
            el.style.display = 'none';
        });

        // 显示选中的模式
        const modeElement = this.container.querySelector(`#${mode}-mode`);
        if (modeElement) {
            modeElement.style.display = 'block';
            this.currentMode = mode;
            this.updateComparisonCharts();
        }

        this.emit('modeChanged', mode);
    }

    applyFilters() {
        if (!this.originalData) return;

        this.showLoading('应用筛选条件...');

        // 获取筛选条件
        const selectedCategories = Array.from(this.container.querySelectorAll('.category-filter:checked'))
            .map(cb => parseInt(cb.value));

        const densityMin = parseFloat(this.container.querySelector('#density-min').value) || -Infinity;
        const densityMax = parseFloat(this.container.querySelector('#density-max').value) || Infinity;
        const distanceMin = parseFloat(this.container.querySelector('#distance-min').value) || -Infinity;
        const distanceMax = parseFloat(this.container.querySelector('#distance-max').value) || Infinity;
        const sampleCount = parseInt(this.container.querySelector('#sample-count').value) || Infinity;

        // 应用筛选
        this.filteredData = this.cloneData(this.originalData);

        this.filteredData.coordinates = this.originalData.coordinates.filter(coord => {
            // 分类筛选
            if (selectedCategories.length > 0 && !selectedCategories.includes(coord.category_id)) {
                return false;
            }

            // 密度筛选
            if (coord.local_density < densityMin || coord.local_density > densityMax) {
                return false;
            }

            // 距离筛选
            if (coord.distance_to_center < distanceMin || coord.distance_to_center > distanceMax) {
                return false;
            }

            return true;
        });

        // 样本数量限制
        if (this.filteredData.coordinates.length > sampleCount) {
            // 随机采样
            const shuffled = [...this.filteredData.coordinates].sort(() => Math.random() - 0.5);
            this.filteredData.coordinates = shuffled.slice(0, sampleCount);
        }

        // 更新分类统计
        this.updateCategoryStatistics();

        this.updateComparisonCharts();
        this.updateStatistics();
        this.hideLoading();

        this.emit('filtersApplied', {
            selectedCategories,
            densityRange: [densityMin, densityMax],
            distanceRange: [distanceMin, distanceMax],
            sampleCount: this.filteredData.coordinates.length
        });
    }

    updateCategoryStatistics() {
        if (!this.filteredData) return;

        const categoryCounts = {};
        this.filteredData.coordinates.forEach(coord => {
            categoryCounts[coord.category_id] = (categoryCounts[coord.category_id] || 0) + 1;
        });

        this.filteredData.categories = this.filteredData.categories.map(cat => ({
            ...cat,
            sample_count: categoryCounts[cat.category_id] || 0
        }));
    }

    resetFilters() {
        // 重置所有筛选条件
        this.container.querySelectorAll('.category-filter').forEach(cb => {
            cb.checked = true;
        });

        this.container.querySelector('#density-min').value = '';
        this.container.querySelector('#density-max').value = '';
        this.container.querySelector('#distance-min').value = '';
        this.container.querySelector('#distance-max').value = '';
        this.container.querySelector('#sample-count').value = '';

        // 重置数据
        this.filteredData = this.cloneData(this.originalData);
        this.updateComparisonCharts();
        this.updateStatistics();

        this.emit('filtersReset');
    }

    randomSample() {
        if (!this.originalData) return;

        const sampleCount = parseInt(this.container.querySelector('#sample-count').value) ||
                           Math.floor(this.originalData.coordinates.length * 0.3);

        if (sampleCount > this.originalData.coordinates.length) {
            this.showNotification('采样数量不能超过总样本数', 'warning');
            return;
        }

        this.container.querySelector('#sample-count').value = sampleCount;

        // 随机采样
        const shuffled = [...this.originalData.coordinates].sort(() => Math.random() - 0.5);
        this.filteredData = this.cloneData(this.originalData);
        this.filteredData.coordinates = shuffled.slice(0, sampleCount);

        this.updateCategoryStatistics();
        this.updateComparisonCharts();
        this.updateStatistics();

        this.emit('randomSampled', { sampleCount });
    }

    updateComparisonCharts() {
        if (!this.originalData || !this.filteredData) return;

        const mode = this.currentMode || 'side-by-side';

        switch (mode) {
            case 'side-by-side':
                this.renderSideBySideCharts();
                break;
            case 'overlay':
                this.renderOverlayChart();
                break;
            case 'split':
                this.renderSplitChart();
                break;
        }
    }

    renderSideBySideCharts() {
        const showOriginal = this.container.querySelector('#show-original').checked;
        const showFiltered = this.container.querySelector('#show-filtered').checked;

        if (showOriginal && this.originalData) {
            this.renderChart('original-chart', this.originalData.coordinates, '原始数据');
        }

        if (showFiltered && this.filteredData) {
            this.renderChart('filtered-chart', this.filteredData.coordinates, '筛选后数据');
        }
    }

    renderOverlayChart() {
        const showOriginal = this.container.querySelector('#show-original').checked;
        const showFiltered = this.container.querySelector('#show-filtered').checked;

        const traces = [];

        if (showOriginal && this.originalData) {
            const originalOpacity = parseFloat(this.container.querySelector('#original-opacity').value);
            traces.push(this.createTrace(this.originalData.coordinates, '原始数据', originalOpacity, 'circle'));
        }

        if (showFiltered && this.filteredData) {
            const filteredOpacity = parseFloat(this.container.querySelector('#filtered-opacity').value);
            traces.push(this.createTrace(this.filteredData.coordinates, '筛选后数据', filteredOpacity, 'diamond'));
        }

        const layout = this.createComparisonLayout('叠加对比');
        const config = this.createPlotlyConfig();

        if (typeof Plotly !== 'undefined') {
            Plotly.newPlot('overlay-chart', traces, layout, config);
        }
    }

    renderSplitChart() {
        if (!this.originalData || !this.filteredData) return;

        // 创建分屏布局
        const originalData = this.originalData.coordinates.map(coord => ({
            ...coord,
            x: coord.x > 0 ? coord.x - 5 : coord.x + 5 // 左移原始数据
        }));

        const filteredData = this.filteredData.coordinates.map(coord => ({
            ...coord,
            x: coord.x < 0 ? coord.x + 5 : coord.x - 5 // 右移筛选数据
        }));

        const traces = [
            this.createTrace(originalData, '原始数据', 0.5, 'circle'),
            this.createTrace(filteredData, '筛选后数据', 0.8, 'diamond')
        ];

        const layout = this.createComparisonLayout('分屏对比');
        const config = this.createPlotlyConfig();

        if (typeof Plotly !== 'undefined') {
            Plotly.newPlot('split-chart', traces, layout, config);
        }
    }

    renderChart(containerId, coordinates, name) {
        const trace = this.createTrace(coordinates, name);
        const layout = this.createComparisonLayout(name);
        const config = this.createPlotlyConfig();

        if (typeof Plotly !== 'undefined') {
            Plotly.newPlot(containerId, [trace], layout, config);
        }
    }

    createTrace(coordinates, name, opacity = 0.8, markerType = 'circle') {
        return {
            x: coordinates.map(p => p.x),
            y: coordinates.map(p => p.y),
            mode: 'markers',
            type: 'scatter',
            name: name,
            text: coordinates.map(p => `${p.sample_id}<br>分类: ${p.category_name}<br>密度: ${p.local_density.toFixed(3)}`),
            marker: {
                size: 8,
                color: coordinates.map(p => p.category_id),
                colorscale: 'Viridis',
                showscale: true,
                colorbar: {
                    title: '分类'
                },
                opacity: opacity,
                symbol: markerType,
                line: {
                    color: 'rgba(255, 255, 255, 0.5)',
                    width: 0.5
                }
            },
            hovertemplate: '<b>%{text}</b><br><extra></extra>'
        };
    }

    createComparisonLayout(title) {
        return {
            title: {
                text: title,
                font: {
                    size: 16,
                    weight: 'bold'
                },
                x: 0.5,
                xanchor: 'center'
            },
            xaxis: {
                title: 't-SNE维度1',
                gridcolor: '#e0e0e0',
                zerolinecolor: '#e0e0e0'
            },
            yaxis: {
                title: 't-SNE维度2',
                gridcolor: '#e0e0e0',
                zerolinecolor: '#e0e0e0'
            },
            width: this.options.width,
            height: this.options.height,
            showlegend: true,
            hovermode: 'closest',
            plot_bgcolor: 'white',
            paper_bgcolor: 'white'
        };
    }

    createPlotlyConfig() {
        return {
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
        };
    }

    updateChartOpacity() {
        if (this.currentMode === 'overlay') {
            this.renderOverlayChart();
        }
    }

    updateChartVisibility() {
        this.updateComparisonCharts();
    }

    updateStatistics() {
        if (!this.originalData || !this.filteredData) return;

        const container = this.container.querySelector('#comparison-statistics');

        const originalCount = this.originalData.coordinates.length;
        const filteredCount = this.filteredData.coordinates.length;
        const reductionRate = ((originalCount - filteredCount) / originalCount * 100).toFixed(1);

        const originalCategories = this.originalData.categories.length;
        const filteredCategories = this.filteredData.categories.filter(cat => cat.sample_count > 0).length;

        const originalAvgDensity = this.calculateAverageDensity(this.originalData.coordinates);
        const filteredAvgDensity = this.calculateAverageDensity(this.filteredData.coordinates);

        container.innerHTML = `
            <div class="row">
                <div class="col-md-3">
                    <div class="stats-card">
                        <h6>样本数量</h6>
                        <div class="d-flex justify-content-between">
                            <span>原始:</span>
                            <strong>${String(Number.isFinite(originalCount) ? originalCount : (parseInt(originalCount)||0))}</strong>
                        </div>
                        <div class="d-flex justify-content-between">
                            <span>筛选:</span>
                            <strong>${String(Number.isFinite(filteredCount) ? filteredCount : (parseInt(filteredCount)||0))}</strong>
                        </div>
                        <div class="d-flex justify-content-between text-danger">
                            <span>减少:</span>
                            <strong>${reductionRate}%</strong>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stats-card">
                        <h6>分类数量</h6>
                        <div class="d-flex justify-content-between">
                            <span>原始:</span>
                            <strong>${originalCategories}</strong>
                        </div>
                        <div class="d-flex justify-content-between">
                            <span>筛选:</span>
                            <strong>${filteredCategories}</strong>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stats-card">
                        <h6>平均密度</h6>
                        <div class="d-flex justify-content-between">
                            <span>原始:</span>
                            <strong>${originalAvgDensity.toFixed(3)}</strong>
                        </div>
                        <div class="d-flex justify-content-between">
                            <span>筛选:</span>
                            <strong>${filteredAvgDensity.toFixed(3)}</strong>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stats-card">
                        <h6>分布变化</h6>
                        <div class="text-center">
                            <div class="progress">
                                <div class="progress-bar" role="progressbar"
                                     style="width: ${reductionRate}%">
                                    ${reductionRate}%
                                </div>
                            </div>
                            <small class="text-muted">数据压缩率</small>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    calculateAverageDensity(coordinates) {
        if (!coordinates || coordinates.length === 0) return 0;
        const sum = coordinates.reduce((acc, coord) => acc + coord.local_density, 0);
        return sum / coordinates.length;
    }

    getSampleCountText(type) {
        if (type === 'original' && this.originalData) {
            return `${this.originalData.coordinates.length} 样本`;
        } else if (type === 'filtered' && this.filteredData) {
            return `${this.filteredData.coordinates.length} 样本`;
        }
        return '加载中...';
    }

    async exportComparison() {
        if (!this.currentPipeline) {
            this.showNotification('请先加载数据', 'warning');
            return;
        }

        const format = this.container.querySelector('#comparison-format').value;
        const width = parseInt(this.container.querySelector('#comparison-width').value);
        const height = parseInt(this.container.querySelector('#comparison-height').value);

        this.showNotification('对比图导出功能需要后端API支持', 'info');

        // 这里可以调用后端API进行对比图导出
        this.emit('exportRequested', { format, width, height });
    }

    showLoading(message = '加载中...') {
        // 在各个图表容器中显示加载状态
        const charts = ['original-chart', 'filtered-chart', 'overlay-chart', 'split-chart'];
        charts.forEach(chartId => {
            const element = document.getElementById(chartId);
            if (element) {
                element.innerHTML = `
                    <div class="d-flex justify-content-center align-items-center h-100">
                        <div class="text-center">
                            <div class="spinner-border text-primary mb-3" role="status">
                                <span class="visually-hidden">Loading...</span>
                            </div>
                            <div>${message}</div>
                        </div>
                    </div>
                `;
            }
        });
    }

    showError(message) {
        // 在第一个图表容器中显示错误
        const firstChart = this.container.querySelector('[id$="-chart"]');
        if (firstChart) {
            firstChart.innerHTML = `
                <div class="alert alert-danger m-3">
                    <i class="bi bi-exclamation-triangle"></i> ${message}
                </div>
            `;
        }
    }

    hideLoading() {
        // Loading will be replaced by chart rendering
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        document.body.appendChild(notification);

        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
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
    getCurrentData() {
        return {
            original: this.originalData,
            filtered: this.filteredData
        };
    }

    getComparisonConfig() {
        return {
            mode: this.currentMode,
            showOriginal: this.container.querySelector('#show-original').checked,
            showFiltered: this.container.querySelector('#show-filtered').checked,
            originalOpacity: parseFloat(this.container.querySelector('#original-opacity').value),
            filteredOpacity: parseFloat(this.container.querySelector('#filtered-opacity').value)
        };
    }

    destroy() {
        this.container.innerHTML = '';
        this.eventHandlers = {};
        this.selectedCategories.clear();
    }
}

export default ComparisonComponent;
