/**
 * 可视化组件 - MOF数据t-SNE可视化应用
 * 使用Plotly.js渲染t-SNE降维结果的交互式图表
 */

import APIService from '../services/api.js';

class VisualizationComponent {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Container element with id '${containerId}' not found`);
        }

        this.options = {
            width: 1200,
            height: 600,
            responsive: true,
            showLegend: true,
            enableInteractions: true,
            colorScheme: 'Viridis',
            markerSize: 8,
            ...options
        };

        this.currentData = null;
        this.currentConfig = null;
        this.plotlyInstance = null;
        this.apiService = new APIService();
        this.eventHandlers = {};
        this.selectedPoints = new Set();
        this.filteredCategories = new Set();
        this.embeddings = { E1: null, E2: null }; // { pipelineId }
        this.currentEmbedding = null;
        this.highlightMode = 'all';
        this.e2SampleIdSet = new Set();
        this.colorMap = {}; // category_name -> hex
        this.bgColor = '#787878';

        this.init();
    }

    init() {
        this.createVisualizationInterface();
        this.setupEventListeners();
        this.loadSavedSettings();
        this.checkPlotlyAvailability();
    }

    createVisualizationInterface() {
        this.container.innerHTML = `
            <div class="visualization-container">
                <!-- 控制面板 -->
                <div class="controls-panel mb-3">
                    <div class="card">
                        <div class="card-header">
                            <h5><i class="bi bi-sliders"></i> 可视化控制</h5>
                        </div>
                        <div class="card-body">
                            <div class="row mb-3">
                                <div class="col-md-3">
                                    <label for="viz-embedding-select" class="form-label">嵌入选择</label>
                                    <select class="form-select form-select-sm" id="viz-embedding-select">
                                        <option value="E2">E2（4519）</option>
                                        <option value="E1">E1（12088）</option>
                                    </select>
                                </div>
                                <div class="col-md-3">
                                    <label for="viz-highlight-mode" class="form-label">高亮模式</label>
                                    <select class="form-select form-select-sm" id="viz-highlight-mode">
                                        <option value="all">全部样本</option>
                                        <option value="adsorbent">吸水（strong）</option>
                                        <option value="non_adsorbent">不吸水（weak）</option>
                                        <option value="stable">水稳定（stable）</option>
                                        <option value="unstable">水不稳定（unstable）</option>
                                        <option value="strong_stable">strong&stable</option>
                                        <option value="strong_unstable">strong&unstable</option>
                                        <option value="weak_stable">weak&stable</option>
                                        <option value="weak_unstable">weak&unstable</option>
                                        <option value="pld_highlight">PLD>4.9（E1高亮4519）</option>
                                    </select>
                                </div>
                            </div>
                            <div class="row">
                                <div class="col-md-3">
                                    <label for="chart-title" class="form-label">图表标题</label>
                                    <input type="text" class="form-control form-control-sm" id="chart-title"
                                           value="MOF数据t-SNE可视化" maxlength="50">
                                </div>
                                <div class="col-md-3">
                                    <label for="x-axis-label" class="form-label">X轴标签</label>
                                    <input type="text" class="form-control form-control-sm" id="x-axis-label"
                                           value="t-SNE维度1" maxlength="30">
                                </div>
                                <div class="col-md-3">
                                    <label for="y-axis-label" class="form-label">Y轴标签</label>
                                    <input type="text" class="form-control form-control-sm" id="y-axis-label"
                                           value="t-SNE维度2" maxlength="30">
                                </div>
                                <div class="col-md-3">
                                    <label for="color-scheme" class="form-label">配色方案</label>
                                    <select class="form-select form-select-sm" id="color-scheme">
                                        <option value="Viridis">Viridis</option>
                                        <option value="Plasma">Plasma</option>
                                        <option value="Inferno">Inferno</option>
                                        <option value="Magma">Magma</option>
                                        <option value="Cividis">Cividis</option>
                                        <option value="Rainbow">Rainbow</option>
                                        <option value="Category10">Category10</option>
                                        <option value="Dark2">Dark2</option>
                                    </select>
                                </div>
                            </div>
                            <div class="row mt-3">
                                <div class="col-md-3">
                                    <label for="marker-size" class="form-label">标记大小</label>
                                    <input type="range" class="form-range" style="height: 0.5rem;" id="marker-size"
                                           min="3" max="20" value="8">
                                    <div class="form-text">当前: <span id="marker-size-value">8</span></div>
                                </div>
                                <div class="col-md-3">
                                    <label for="marker-opacity" class="form-label">透明度</label>
                                    <input type="range" class="form-range" style="height: 0.5rem;" id="marker-opacity"
                                           min="0.1" max="1" step="0.1" value="0.8">
                                    <div class="form-text">当前: <span id="marker-opacity-value">0.8</span></div>
                                </div>
                                <div class="col-md-3">
                                    <label for="bg-opacity" class="form-label">背景透明度</label>
                                    <input type="range" class="form-range" style="height: 0.5rem;" id="bg-opacity"
                                           min="0.05" max="1" step="0.05" value="0.25">
                                    <div class="form-text">当前: <span id="bg-opacity-value">0.25</span></div>
                                </div>
                        
                    </div>

                    <!-- 颜色设置（折叠 + 集中控制） -->
                    <div class="row mt-3">
                        <div class="col-12">
                            <h6><i class="bi bi-palette"></i> 颜色设置</h6>
                            <div class="d-flex align-items-center gap-2 mb-2">
                                <button type="button" class="btn btn-sm btn-outline-secondary" id="toggle-color-settings">
                                    打开/收起
                                </button>
                            </div>
                            <div id="color-settings-panel" class="mt-1" style="display: none;">
                                <div class="d-flex align-items-center flex-wrap gap-2">
                                    <label class="form-label mb-0" for="category-color-target">目标</label>
                                    <select class="form-select form-select-sm" id="category-color-target"></select>
                                    <input type="color" class="form-control form-control-color" style="width: 46px; padding: 0;" id="category-color-picker" value="#1f77b4"/>
                                    <div class="form-check ms-2">
                                        <input class="form-check-input" type="checkbox" id="show-legend" checked>
                                        <label class="form-check-label" for="show-legend">显示图例</label>
                                    </div>
                                </div>
                            </div>
                            <div id="category-colors" class="d-flex flex-wrap gap-3" style="display: none;"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

                <!-- 图表容器 -->
                <div class="chart-container">
                    <div class="card">
                        <div class="card-header">
                            <div class="d-flex justify-content-between align-items-center">
                                <h5><i class="bi bi-graph-up"></i> t-SNE散点图</h5>
                                <div class="chart-actions">
                                    <button type="button" class="btn btn-outline-success btn-sm me-2" id="btn-show-report">
                                        <i class="bi bi-clipboard-data"></i> 报告
                                    </button>
                                    <button type="button" class="btn btn-outline-secondary btn-sm me-2" id="btn-toggle-info">
                                        <i class="bi bi-info-circle"></i> 信息
                                    </button>
                                    <button type="button" class="btn btn-outline-primary btn-sm me-2" id="btn-toggle-selection">
                                        <i class="bi bi-cursor"></i> 选择
                                    </button>
                                    <button type="button" class="btn btn-outline-warning btn-sm" id="btn-clear-selection">
                                        <i class="bi bi-x-circle"></i> 清除选择
                                    </button>
                                </div>
                            </div>
                        </div>
                        <div class="card-body">
                            <div id="plotly-chart" style="width: 100%; height: 600px;"></div>
                            <div id="training-report" class="mt-3" style="display:none;">
                                <div class="card">
                                    <div class="card-header d-flex justify-content-between align-items-center">
                                        <strong>训练报告</strong>
                                        <div>
                                            <button class="btn btn-sm btn-outline-secondary" id="btn-hide-report">隐藏</button>
                                        </div>
                                    </div>
                                    <div class="card-body">
                                        <div class="row">
                                            <div class="col-md-5">
                                                <h6>最终参数</h6>
                                                <pre id="report-params" class="p-2 bg-light" style="max-height: 260px; overflow:auto;"></pre>
                                            </div>
                                            <div class="col-md-7">
                                                <div class="mb-3">
                                                    <h6 class="mb-1">PCA 累计解释方差</h6>
                                                    <div id="report-pca-curve" style="height: 180px;"></div>
                                                </div>
                                                <div>
                                                    <h6 class="mb-1">自动调参 KL 对比（越低越好）</h6>
                                                    <div id="report-tuning-bars" style="height: 220px;"></div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 统计信息面板 -->
                <div class="statistics-panel mt-3">
                    <div class="row">
                        <div class="col-md-3">
                            <div class="stats-card">
                                <h6>总样本数</h6>
                                <h3 id="total-samples">0</h3>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="stats-card">
                                <h6>分类数量</h6>
                                <h3 id="category-count">0</h3>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="stats-card">
                                <h6>已选择</h6>
                                <h3 id="selected-count">0</h3>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="stats-card">
                                <h6>渲染时间</h6>
                                <h3 id="render-time">0ms</h3>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 分类过滤器 -->
                <div class="category-filter mt-3">
                    <div class="card">
                        <div class="card-header">
                            <h5><i class="bi bi-funnel"></i> 分类过滤器</h5>
                        </div>
                        <div class="card-body">
                            <div id="category-filters" class="category-filters">
                                <!-- 动态生成分类过滤器 -->
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 选中点详情 -->
                <div class="selection-details mt-3" id="selection-details" style="display: none;">
                    <div class="card">
                        <div class="card-header">
                            <h5><i class="bi bi-list-ul"></i> 选中点详情</h5>
                        </div>
                        <div class="card-body">
                            <div id="selected-points-table" class="table-responsive">
                                <!-- 动态生成选中点表格 -->
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    setupEventListeners() {
        // 控制面板事件
        const embSel = this.container.querySelector('#viz-embedding-select');
        if (embSel) {
            embSel.addEventListener('change', async (e) => {
                this.currentEmbedding = e.target.value;
                if (this.embeddings[this.currentEmbedding]?.pipelineId) {
                    await this.loadData(this.embeddings[this.currentEmbedding].pipelineId);
                }
            });
        }

        const modeSel = this.container.querySelector('#viz-highlight-mode');
        if (modeSel) {
            modeSel.addEventListener('change', () => {
                this.highlightMode = modeSel.value;
                this.updateChartStyle();
            });
        }

        this.container.querySelector('#marker-size').addEventListener('input', (e) => {
            const v = document.getElementById('marker-size-value');
            if (v) v.textContent = e.target.value;
            this.updateChartStyle();
        });

        this.container.querySelector('#marker-opacity').addEventListener('input', (e) => {
            const v = document.getElementById('marker-opacity-value');
            if (v) v.textContent = e.target.value;
            this.updateChartStyle();
        });

        this.container.querySelector('#bg-opacity').addEventListener('input', (e) => {
            document.getElementById('bg-opacity-value').textContent = e.target.value;
            this.updateChartStyle();
        });

        // 操作按钮已移除，保留快捷键或菜单时可复用 resetView/applySettings

        this.container.querySelector('#btn-clear-selection').addEventListener('click', () => {
            this.clearSelection();
        });

        // 报告按钮
        const btnShowReport = this.container.querySelector('#btn-show-report');
        const btnHideReport = this.container.querySelector('#btn-hide-report');
        if (btnShowReport) btnShowReport.addEventListener('click', () => this.toggleReport(true));
        if (btnHideReport) btnHideReport.addEventListener('click', () => this.toggleReport(false));

        // 实时更新事件
        const inputs = ['chart-title', 'x-axis-label', 'y-axis-label', 'color-scheme'];
        inputs.forEach(id => {
            this.container.querySelector(`#${id}`).addEventListener('input', () => {
                this.updateChartStyle();
            });
        });

        this.container.querySelector('#show-legend').addEventListener('change', () => {
            this.updateChartStyle();
        });

        // 背景颜色通过颜色设置目标选择“背景”来实时更新

        // 颜色设置折叠与应用
        const toggleBtn = this.container.querySelector('#toggle-color-settings');
        const colorPanel = this.container.querySelector('#color-settings-panel');
        if (toggleBtn && colorPanel) {
            toggleBtn.addEventListener('click', () => {
                const show = colorPanel.style.display === 'none';
                colorPanel.style.display = show ? 'block' : 'none';
                // 同时显示/隐藏旧的背景控件（已通过CSS隐藏，双保险）
                const bgLbl = this.container.querySelector('label[for="bg-color"]');
                const bgCtl = this.container.querySelector('#bg-color');
                if (bgLbl) bgLbl.style.display = show ? 'inline-block' : 'none';
                if (bgCtl) bgCtl.style.display = show ? 'inline-block' : 'none';
            });
        }

        const targetSel = this.container.querySelector('#category-color-target');
        const picker = this.container.querySelector('#category-color-picker');
        if (targetSel && picker) {
            picker.addEventListener('input', () => {
                const target = targetSel.value;
                const value = picker.value;
                const cats = Array.isArray(this.currentData?.categories) ? this.currentData.categories : [];
                if (target === '__background__') {
                    this.bgColor = value;
                } else if (['adsorbent','non_adsorbent','stable','unstable'].includes(target)) {
                    if (!this.highlightColorMap) this.highlightColorMap = {};
                    this.highlightColorMap[target] = value;
                } else if (target === '__all__') {
                    cats.forEach(c => { this.colorMap[c.category_name] = value; });
                } else {
                    this.colorMap[target] = value;
                }
                this.saveColorSettings();
                this.updateChartStyle();
            });
        }
    }

    checkPlotlyAvailability() {
        if (typeof Plotly === 'undefined') {
            console.error('Plotly.js is not loaded');
            this.container.querySelector('#plotly-chart').innerHTML = `
                <div class="alert alert-danger">
                    <i class="bi bi-exclamation-triangle"></i>
                    Plotly.js 未加载，请检查网络连接或刷新页面
                </div>
            `;
            return false;
        }
        return true;
    }

    async loadData(pipelineId) {
        try {
            this.showLoading('加载可视化数据...');

            const response = await this.apiService.getVisualizationData(pipelineId);

            if (!response || response.success !== true || !response.data) {
                throw new Error(response?.detail || 'Failed to load visualization data');
            }

            this.currentData = response.data;
            this.currentConfig = response.data.config || {};

            // 渲染主图
            this.renderVisualization();

            // 非关键路径：统计与分类过滤构建中的异常不应中断渲染
            try { this.updateStatistics(); } catch (e) { console.warn('updateStatistics error:', e); }
            try { this.createCategoryFilters(); } catch (e) { console.warn('createCategoryFilters error:', e); }

            this.hideLoading();
            this.emit('dataLoaded', this.currentData);
        } catch (error) {
            // 降级：尽可能显示基本图表而不是覆盖错误层
            console.error('loadData error:', error);
            try {
                if (this.currentData && Array.isArray(this.currentData.coordinates)) {
                    this.renderVisualization();
                    try { this.updateStatistics(); } catch (_) {}
                } else {
                    this.showError(`加载可视化数据失败: ${error?.message || error}`);
                }
            } finally {
                this.hideLoading();
                this.emit('loadError', error);
            }
        }
    }

    renderVisualization() {
        if (!this.currentData || !this.checkPlotlyAvailability()) return;

        const startTime = performance.now();
        // 根据高亮模式拆分前景/背景；背景始终存在（灰色），前景为当前高亮集合
        const { fgPoints, bgPoints } = this.splitByHighlight(this.currentData.coordinates || []);
        const traces = [];
        const layout = this.createLayout();
        const config = this.createPlotlyConfig();

        const bgTrace = this.createBackgroundTrace(bgPoints);
        if (bgTrace) traces.push(bgTrace);
        const fgTrace = this.createForegroundTrace(fgPoints);
        if (fgTrace) traces.push(fgTrace);

        try {
            Plotly.newPlot('plotly-chart', traces, layout, config);
            this.setupPlotlyInteractions();

            const renderTime = Math.round(performance.now() - startTime);
            document.getElementById('render-time').textContent = `${renderTime}ms`;

            this.emit('rendered', { renderTime, pointCount: (fgPoints.length + bgPoints.length) });
        } catch (error) {
            console.error('Plotly rendering error:', error);
            this.showError('图表渲染失败');
        }
    }

    splitByHighlight(coordinates) {
        const highlight = [];
        const background = [];
        const predicate = (p) => this.isHighlighted(p);
        // 先应用分类过滤（复选框），再按高亮模式分前景/背景
        const filtered = (this.filteredCategories && this.filteredCategories.size > 0)
            ? coordinates.filter(p => this.filteredCategories.has(p.category_id))
            : coordinates;
        for (const p of filtered) {
            (predicate(p) ? highlight : background).push(p);
        }
        return { fgPoints: highlight, bgPoints: background };
    }

    isHighlighted(point) {
        const tokens = this.parseCategoryTokens(point.category_name);
        switch (this.highlightMode) {
            case 'adsorbent':
                return tokens.has('strong');
            case 'non_adsorbent':
                return tokens.has('weak');
            case 'stable':
                return tokens.has('stable') && !tokens.has('unstable');
            case 'unstable':
                return tokens.has('unstable');
            case 'strong_stable':
                return tokens.has('strong') && tokens.has('stable');
            case 'strong_unstable':
                return tokens.has('strong') && tokens.has('unstable');
            case 'weak_stable':
                return tokens.has('weak') && tokens.has('stable');
            case 'weak_unstable':
                return tokens.has('weak') && tokens.has('unstable');
            case 'pld_highlight':
                return this.currentEmbedding === 'E1' && this.e2SampleIdSet.has(String(point.sample_id));
            case 'all':
            default:
                return true;
        }
    }

    // 解析类别名为 token 集（lowercase），避免 'stable' 命中 'unstable' 的子串问题
    parseCategoryTokens(name) {
        const s = String(name || '').toLowerCase().trim();
        if (!s) return new Set();
        const parts = s.split(/[\s&/,\\_-]+/).filter(Boolean);
        return new Set(parts);
    }

    createForegroundTrace(coordinates) {
        const markerSize = parseInt(this.container.querySelector('#marker-size').value);
        const markerOpacity = parseFloat(this.container.querySelector('#marker-opacity').value);
        const colorScheme = this.container.querySelector('#color-scheme').value;

        // 前景颜色：若处于四种组高亮之一且设置了组颜色，则统一色；否则按类别映射
        let colors = [];
        if (['adsorbent','non_adsorbent','stable','unstable'].includes(this.highlightMode)
            && this.highlightColorMap && this.highlightColorMap[this.highlightMode]) {
            const c = this.highlightColorMap[this.highlightMode];
            colors = coordinates.map(() => c);
        } else {
            const palette = this.getPalette(colorScheme);
            colors = coordinates.map(p => this.colorForCategory(p.category_name, palette));
        }

        return {
            x: coordinates.map(p => p.x),
            y: coordinates.map(p => p.y),
            mode: 'markers',
            type: 'scatter',
            text: coordinates.map(p => this.createHoverText(p)),
            marker: {
                size: markerSize,
                color: colors,
                opacity: markerOpacity,
                line: {
                    color: 'rgba(255, 255, 255, 0.5)',
                    width: 0.5
                }
            },
            name: '高亮',
            hovertemplate: this.getHoverTemplate(),
            selectedpoints: Array.from(this.selectedPoints),
            selected: {
                marker: {
                    color: 'red',
                    size: markerSize + 2,
                    line: {
                        color: 'darkred',
                        width: 2
                    }
                }
            },
            unselected: {
                marker: {
                    opacity: markerOpacity * 0.3,
                    size: markerSize * 0.8
                }
            }
        };
    }

    createBackgroundTrace(coordinates) {
        if (!coordinates || coordinates.length === 0) return null;
        const bgOpacity = parseFloat(this.container.querySelector('#bg-opacity').value || '0.25');
        const markerSize = parseInt(this.container.querySelector('#marker-size').value);
        return {
            x: coordinates.map(p => p.x),
            y: coordinates.map(p => p.y),
            mode: 'markers',
            type: 'scatter',
            text: coordinates.map(p => this.createHoverText(p)),
            marker: {
                size: markerSize,
                color: this.bgColor || '#787878',
                opacity: bgOpacity,
                line: {
                    color: 'rgba(255,255,255,0.3)',
                    width: 0.5
                }
            },
            name: '背景',
            hovertemplate: this.getHoverTemplate(),
        };
    }

    getPalette(scheme) {
        const palettes = {
            'Category10': ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'],
            'Dark2': ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d','#666666'],
            'Viridis': ['#440154','#3b528b','#21918c','#5ec962','#fde725']
        };
        return palettes[scheme] || palettes['Category10'];
    }

    colorForCategory(name, palette) {
        const str = String(name || 'unknown');
        if (this.colorMap && this.colorMap[str]) {
            return this.colorMap[str];
        }
        let hash = 0;
        for (let i = 0; i < str.length; i++) hash = ((hash<<5)-hash) + str.charCodeAt(i);
        const idx = Math.abs(hash) % palette.length;
        return palette[idx];
    }

    createLayout() {
        const title = this.container.querySelector('#chart-title').value;
        const xAxisLabel = this.container.querySelector('#x-axis-label').value;
        const yAxisLabel = this.container.querySelector('#y-axis-label').value;

        return {
            title: {
                text: title,
                font: {
                    size: 18,
                    weight: 'bold'
                },
                x: 0.5,
                xanchor: 'center'
            },
            xaxis: {
                title: {
                    text: xAxisLabel,
                    font: {
                        size: 14
                    }
                },
                gridcolor: '#e0e0e0',
                zerolinecolor: '#e0e0e0',
                showline: true,
                linewidth: 1,
                mirror: true
            },
            yaxis: {
                title: {
                    text: yAxisLabel,
                    font: {
                        size: 14
                    }
                },
                gridcolor: '#e0e0e0',
                zerolinecolor: '#e0e0e0',
                showline: true,
                linewidth: 1,
                mirror: true,
                scaleanchor: 'x',
                scaleratio: 1
            },
            width: this.options.width,
            height: this.options.height,
            showlegend: this.container.querySelector('#show-legend').checked,
            hovermode: 'closest',
            plot_bgcolor: 'white',
            paper_bgcolor: 'white',
            margin: {
                l: 80,
                r: 50,
                t: 80,
                b: 80
            },
            clickmode: 'event+select'
        };
    }

    createPlotlyConfig() {
        return {
            responsive: this.options.responsive,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
            toImageButtonOptions: {
                format: 'png',
                filename: 'mof-tsne-visualization',
                height: this.options.height,
                width: this.options.width,
                scale: 2
            },
            editable: false,
            scrollZoom: true,
            doubleClick: 'reset+autosize'
        };
    }

    setupPlotlyInteractions() {
        const chartElement = this.container.querySelector('#plotly-chart');

        chartElement.on('plotly_click', (data) => {
            this.handlePointClick(data);
        });

        chartElement.on('plotly_selected', (eventData) => {
            this.handleSelection(eventData);
        });

        chartElement.on('plotly_hover', (data) => {
            this.handleHover(data);
        });
    }

    handlePointClick(data) {
        if (!data.points || data.points.length === 0) return;

        const point = data.points[0];
        const pointIndex = point.pointIndex;

        if (this.selectedPoints.has(pointIndex)) {
            this.selectedPoints.delete(pointIndex);
        } else {
            this.selectedPoints.add(pointIndex);
        }

        this.updateSelectionDisplay();
        this.emit('pointSelected', {
            pointIndex,
            point: this.currentData.coordinates[pointIndex],
            selected: this.selectedPoints.has(pointIndex)
        });
    }

    handleSelection(eventData) {
        if (eventData && eventData.points) {
            this.selectedPoints.clear();
            eventData.points.forEach(point => {
                this.selectedPoints.add(point.pointIndex);
            });
            this.updateSelectionDisplay();
            this.emit('batchSelected', Array.from(this.selectedPoints));
        }
    }

    handleHover(data) {
        if (data && data.points && data.points.length > 0) {
            const point = data.points[0];
            this.emit('pointHovered', {
                pointIndex: point.pointIndex,
                point: this.currentData.coordinates[point.pointIndex]
            });
        }
    }

    createCategoryFilters() {
        if (!this.currentData || !this.currentData.categories) return;

        const container = this.container.querySelector('#category-filters');
        container.innerHTML = '';

        this.currentData.categories.forEach(category => {
            const filterDiv = document.createElement('div');
            filterDiv.className = 'form-check form-check-inline';
            filterDiv.innerHTML = `
                <input class="form-check-input category-filter" type="checkbox"
                       id="cat-${category.category_id}" value="${category.category_id}" checked>
                <label class="form-check-label" for="cat-${category.category_id}">
                    <span class="category-indicator" style="background-color: ${category.color_code}"></span>
                    ${category.category_name} (${category.sample_count})
                </label>
            `;
            container.appendChild(filterDiv);
        });

        // 添加过滤器事件监听
        container.querySelectorAll('.category-filter').forEach(checkbox => {
            checkbox.addEventListener('change', () => this.updateCategoryFilters());
        });

        // 渲染颜色设置（下拉+选择器）
        try { this.renderColorSettingsUI(); } catch (e) { console.warn('renderColorSettingsUI failed', e); }
    }

    updateCategoryFilters() {
        this.filteredCategories.clear();

        this.container.querySelectorAll('.category-filter:checked').forEach(checkbox => {
            this.filteredCategories.add(parseInt(checkbox.value));
        });

        this.renderVisualization();
        this.emit('categoriesFiltered', Array.from(this.filteredCategories));
    }

    updateSelectionDisplay() {
        const selectedCount = this.selectedPoints.size;
        document.getElementById('selected-count').textContent = selectedCount;

        const detailsPanel = this.container.querySelector('#selection-details');
        if (selectedCount > 0) {
            this.showSelectedPointsTable();
            detailsPanel.style.display = 'block';
        } else {
            detailsPanel.style.display = 'none';
        }
    }

    showSelectedPointsTable() {
        const tableContainer = this.container.querySelector('#selected-points-table');
        const selectedCoords = Array.from(this.selectedPoints).map(index =>
            this.currentData.coordinates[index]
        );

        const tableHTML = `
            <table class="table table-striped table-hover">
                <thead>
                    <tr>
                        <th>样本ID</th>
                        <th>分类</th>
                        <th>坐标 (X, Y)</th>
                        <th>密度</th>
                        <th>操作</th>
                    </tr>
                </thead>
                <tbody>
                    ${selectedCoords.map(coord => `
                        <tr>
                            <td>${coord.sample_id}</td>
                            <td><span class="badge bg-primary">${coord.category_name}</span></td>
                            <td>(${coord.x.toFixed(3)}, ${coord.y.toFixed(3)})</td>
                            <td>${coord.local_density.toFixed(3)}</td>
                            <td>
                                <button type="button" class="btn btn-sm btn-outline-danger"
                                        onclick="this.closest('tr').remove()">
                                    <i class="bi bi-trash"></i>
                                </button>
                            </td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;

        tableContainer.innerHTML = tableHTML;
    }

    updateStatistics() {
        if (!this.currentData) return;

        let total = 0;
        if (this.currentData && this.currentData.statistics && typeof this.currentData.statistics.total_samples !== 'undefined') {
            total = this.currentData.statistics.total_samples;
        } else if (this.currentData && typeof this.currentData.total_samples !== 'undefined') {
            total = this.currentData.total_samples;
        } else if (this.currentData && Array.isArray(this.currentData.coordinates)) {
            total = this.currentData.coordinates.length;
        }

        const totalElem = document.getElementById('total-samples');
        if (totalElem) totalElem.textContent = String(Number.isFinite(total) ? total : (parseInt(total) || 0));

        const catElem = document.getElementById('category-count');
        if (catElem) catElem.textContent = Array.isArray(this.currentData.categories) ? this.currentData.categories.length : 0;

        const filteredCount = Array.isArray(this.currentData.coordinates)
            ? this.filterVisiblePoints(this.currentData.coordinates).length
            : 0;
        if (totalElem) totalElem.textContent = String(filteredCount);
    }

    updateChartStyle() {
        if (this.currentData) {
            this.renderVisualization();
        }
    }

    filterVisiblePoints(coordinates) {
        return coordinates.filter(coord => {
            const catOk = (this.filteredCategories.size === 0) || this.filteredCategories.has(coord.category_id);
            const hlOk = this.isHighlighted(coord);
            return catOk && hlOk;
        });
    }

    applySettings() {
        const settings = this.getCurrentSettings();
        this.saveSettings(settings);
        this.emit('settingsApplied', settings);
    }

    resetView() {
        if (typeof Plotly !== 'undefined') {
            Plotly.relayout('plotly-chart', {
                'xaxis.autorange': true,
                'yaxis.autorange': true
            });
        }
    }

    clearSelection() {
        this.selectedPoints.clear();
        this.updateSelectionDisplay();
        if (this.currentData) {
            this.renderVisualization();
        }
        this.emit('selectionCleared');
    }

    getCurrentSettings() {
        return {
            title: this.container.querySelector('#chart-title').value,
            xAxisLabel: this.container.querySelector('#x-axis-label').value,
            yAxisLabel: this.container.querySelector('#y-axis-label').value,
            colorScheme: this.container.querySelector('#color-scheme').value,
            markerSize: parseInt(this.container.querySelector('#marker-size').value),
            markerOpacity: parseFloat(this.container.querySelector('#marker-opacity').value),
            bgOpacity: parseFloat(this.container.querySelector('#bg-opacity').value),
            showLegend: this.container.querySelector('#show-legend').checked
        };
    }

    loadSavedSettings() {
        try {
            const raw = localStorage.getItem('mof-viz-settings');
            if (!raw) return;
            const s = JSON.parse(raw);
            const setVal = (sel, val) => { const el = this.container.querySelector(sel); if (el && typeof val !== 'undefined') el.value = val; };
            const setChk = (sel, val) => { const el = this.container.querySelector(sel); if (el && typeof val !== 'undefined') el.checked = !!val; };
            setVal('#chart-title', s.title);
            setVal('#x-axis-label', s.xAxisLabel);
            setVal('#y-axis-label', s.yAxisLabel);
            setVal('#color-scheme', s.colorScheme);
            setVal('#marker-size', s.markerSize);
            const ms = this.container.querySelector('#marker-size'); if (ms) this.container.querySelector('#marker-size-value').textContent = ms.value;
            setVal('#marker-opacity', s.markerOpacity);
            const mo = this.container.querySelector('#marker-opacity'); if (mo) this.container.querySelector('#marker-opacity-value').textContent = mo.value;
            setVal('#bg-opacity', s.bgOpacity);
            const bo = this.container.querySelector('#bg-opacity'); if (bo) this.container.querySelector('#bg-opacity-value').textContent = bo.value;
            setChk('#show-legend', s.showLegend);
            // 颜色设置
            const cmraw = localStorage.getItem('mof-viz-colors');
            if (cmraw) this.colorMap = JSON.parse(cmraw) || {};
            const bgc = localStorage.getItem('mof-viz-bgcolor');
            if (bgc) {
                this.bgColor = bgc;
                const bgcEl = this.container.querySelector('#bg-color');
                if (bgcEl) bgcEl.value = bgc;
            }
            const hlraw = localStorage.getItem('mof-viz-hl-colors');
            if (hlraw) this.highlightColorMap = JSON.parse(hlraw) || {};
        } catch (e) {
            console.warn('loadSavedSettings failed', e);
        }
    }

    saveSettings(settings) {
        try {
            localStorage.setItem('mof-viz-settings', JSON.stringify(settings));
        } catch (e) {
            console.warn('saveSettings failed', e);
        }
    }

    // 工具方法
    createHoverText(point) {
        return `${point.sample_id}<br>分类: ${point.category_name}<br>密度: ${point.local_density.toFixed(3)}`;
    }

    getHoverTemplate() {
        return '<b>%{text}</b><br><extra></extra>';
    }

    getColorScale(scheme) {
        const scales = {
            'Viridis': [[0, 'rgb(68, 1, 84)'], [0.25, 'rgb(59, 82, 139)'], [0.5, 'rgb(33, 145, 140)'], [0.75, 'rgb(94, 201, 98)'], [1, 'rgb(253, 231, 37)']],
            'Plasma': [[0, 'rgb(13, 8, 135)'], [0.25, 'rgb(153, 16, 180)'], [0.5, 'rgb(219, 92, 20)'], [0.75, 'rgb(251, 176, 45)'], [1, 'rgb(247, 254, 66)']],
            'Inferno': [[0, 'rgb(0, 0, 4)'], [0.25, 'rgb(120, 28, 109)'], [0.5, 'rgb(227, 105, 21)'], [0.75, 'rgb(251, 183, 39)'], [1, 'rgb(252, 254, 164)']],
            'Magma': [[0, 'rgb(0, 0, 4)'], [0.25, 'rgb(64, 19, 117)'], [0.5, 'rgb(169, 57, 88)'], [0.75, 'rgb(230, 171, 2)'], [1, 'rgb(251, 252, 191)']],
            'Cividis': [[0, 'rgb(0, 32, 76)'], [0.25, 'rgb(40, 52, 139)'], [0.5, 'rgb(96, 123, 139)'], [0.75, 'rgb(188, 189, 51)'], [1, 'rgb(253, 231, 37)']],
            'Rainbow': [[0, 'red'], [0.2, 'orange'], [0.4, 'yellow'], [0.6, 'green'], [0.8, 'blue'], [1, 'purple']],
            'Category10': [[0, '#1f77b4'], [0.11, '#ff7f0e'], [0.22, '#2ca02c'], [0.33, '#d62728'], [0.44, '#9467bd'], [0.56, '#8c564b'], [0.67, '#e377c2'], [0.78, '#7f7f7f'], [0.89, '#bcbd22'], [1, '#17becf']],
            'Dark2': [[0, '#1b9e77'], [0.2, '#d95f02'], [0.4, '#7570b3'], [0.6, '#e7298a'], [0.8, '#66a61e'], [1, '#e6ab02']]
        };
        return scales[scheme] || scales['Viridis'];
    }

    showLoading(message = '加载中...') {
        const chartElement = this.container.querySelector('#plotly-chart');
        chartElement.innerHTML = `
            <div class="d-flex justify-content-center align-items-center" style="height: 400px;">
                <div class="text-center">
                    <div class="spinner-border text-primary mb-3" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <div>${message}</div>
                </div>
            </div>
        `;
    }

    showError(message) {
        const chartElement = this.container.querySelector('#plotly-chart');
        chartElement.innerHTML = `
            <div class="alert alert-danger m-3">
                <i class="bi bi-exclamation-triangle"></i> ${message}
            </div>
        `;
    }

    hideLoading() {
        // Loading will be replaced by chart rendering
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
        return this.currentData;
    }

    getSelectedPoints() {
        return Array.from(this.selectedPoints);
    }

    exportChart(format = 'png', width = 1200, height = 600) {
        if (typeof Plotly !== 'undefined') {
            return Plotly.downloadImage('plotly-chart', {
                format: format,
                width: width,
                height: height,
                filename: 'mof-tsne-visualization'
            });
        }
        return Promise.reject('Plotly not available');
    }

    destroy() {
        if (this.plotlyInstance && typeof Plotly !== 'undefined') {
            Plotly.purge('plotly-chart');
        }
        this.container.innerHTML = '';
        this.eventHandlers = {};
        this.selectedPoints.clear();
        this.filteredCategories.clear();
    }

    async toggleReport(show) {
        const panel = this.container.querySelector('#training-report');
        if (!panel) return;
        if (show) {
            panel.style.display = 'block';
            try {
                const pid = this.embeddings[this.currentEmbedding]?.pipelineId || this.currentData?.pipeline_id;
                if (!pid) return;
                const report = await this.apiService.getTrainingReport(pid);
                const data = report?.data || {};
                // 参数
                const params = data.parameters || {};
                const pre = this.container.querySelector('#report-params');
                pre.textContent = JSON.stringify(params, null, 2);
                // PCA曲线
                const pcaMeta = (data.metrics && data.metrics.pca) || {};
                const cum = Array.isArray(pcaMeta.cumulative_variance_ratio) ? pcaMeta.cumulative_variance_ratio : [];
                if (typeof Plotly !== 'undefined') {
                    const x = cum.map((_, i) => i + 1);
                    const y = cum.map(v => Number(v));
                    const trace = { x, y, mode: 'lines+markers', type: 'scatter', name: '累积方差' };
                    const layout = { margin: {l:40,r:10,t:10,b:40}, yaxis: {range:[0,1]} };
                    Plotly.newPlot(this.container.querySelector('#report-pca-curve'), [trace], layout, {displayModeBar:false});
                }
                // 调参条形图
                const tuning = data.tuning || {};
                const results = Array.isArray(tuning.results) ? tuning.results.filter(r=>r.success) : [];
                if (results.length && typeof Plotly !== 'undefined') {
                    const sorted = results.sort((a,b)=>(a.kl_divergence||1e9)-(b.kl_divergence||1e9)).slice(0,12);
                    const x = sorted.map(r => `${r.metric}\nperp=${r.perplexity}`);
                    const y = sorted.map(r => Number(r.kl_divergence));
                    const bar = { x, y, type: 'bar', marker: {color: '#4e79a7'} };
                    const layout2 = { margin: {l:60,r:10,t:10,b:60} };
                    Plotly.newPlot(this.container.querySelector('#report-tuning-bars'), [bar], layout2, {displayModeBar:false});
                }
            } catch (e) {
                console.warn('加载训练报告失败', e);
            }
        } else {
            panel.style.display = 'none';
        }
    }
}

export default VisualizationComponent;

// 提供外部接口：设置可用嵌入
VisualizationComponent.prototype.setEmbeddings = async function(embeddings) {
    this.embeddings = { ...this.embeddings, ...embeddings };
    // 默认优先 E2
    const preferred = this.embeddings.E2?.pipelineId ? 'E2' : (this.embeddings.E1?.pipelineId ? 'E1' : null);
    if (preferred) {
        this.currentEmbedding = preferred;
        const sel = this.container.querySelector('#viz-embedding-select');
        if (sel) sel.value = preferred;
        await this.loadData(this.embeddings[preferred].pipelineId);
    }
    // 若 E2 存在，预取其 sample_id 集合用于 E1 高亮
    try {
        if (this.embeddings.E2?.pipelineId) {
            const resp = await this.apiService.getVisualizationData(this.embeddings.E2.pipelineId);
            if (resp && resp.success && resp.data && Array.isArray(resp.data.coordinates)) {
                this.e2SampleIdSet = new Set(resp.data.coordinates.map(p => String(p.sample_id)));
            }
        }
    } catch (e) {
        console.warn('预取 E2 样本集失败', e);
    }
}

// 渲染分类颜色选择器
VisualizationComponent.prototype.renderCategoryColorPickers = function() {
    const wrap = this.container.querySelector('#category-colors');
    if (!wrap) return;
    wrap.innerHTML = '';
    const categories = Array.isArray(this.currentData?.categories) ? this.currentData.categories : [];
    const palette = this.getPalette(this.container.querySelector('#color-scheme').value);
    categories.forEach((cat, idx) => {
        const name = cat.category_name || `C${idx}`;
        if (!this.colorMap[name]) {
            this.colorMap[name] = cat.color_code || this.colorForCategory(name, palette);
        }
        const color = this.colorMap[name];
        const div = document.createElement('div');
        div.className = 'd-flex align-items-center gap-2 me-3 mb-2';
        const id = `color-${idx}`;
        div.innerHTML = `
            <label for="${id}" class="form-label mb-0">${name}</label>
            <input type="color" id="${id}" class="form-control form-control-color" value="${color}" title="为 ${name} 选择颜色"/>
        `;
        wrap.appendChild(div);
        div.querySelector('input').addEventListener('input', (e) => {
            this.colorMap[name] = e.target.value;
            this.saveColorSettings();
            this.updateChartStyle();
        });
    });
}

VisualizationComponent.prototype.saveColorSettings = function() {
    try {
        localStorage.setItem('mof-viz-colors', JSON.stringify(this.colorMap));
        localStorage.setItem('mof-viz-bgcolor', String(this.bgColor || '#787878'));
        if (this.highlightColorMap) {
            localStorage.setItem('mof-viz-hl-colors', JSON.stringify(this.highlightColorMap));
        }
    } catch (e) { console.warn('saveColorSettings failed', e); }
}

// 渲染折叠式颜色设置（下拉 + 选择器）
VisualizationComponent.prototype.renderColorSettingsUI = function() {
    const targetSel = this.container.querySelector('#category-color-target');
    const picker = this.container.querySelector('#category-color-picker');
    if (!targetSel || !picker) return;

    const categories = Array.isArray(this.currentData?.categories) ? this.currentData.categories : [];
    const prev = targetSel.value;
    targetSel.innerHTML = '';
    const makeOpt = (val, text) => { const o = document.createElement('option'); o.value = val; o.textContent = text; return o; };
    targetSel.appendChild(makeOpt('__background__','背景'));
    targetSel.appendChild(makeOpt('__all__','全部类别（批量）'));
    // 高亮模式的四个分组
    targetSel.appendChild(makeOpt('adsorbent','吸水（strong）'));
    targetSel.appendChild(makeOpt('non_adsorbent','不吸水（weak）'));
    targetSel.appendChild(makeOpt('stable','水稳定（stable）'));
    targetSel.appendChild(makeOpt('unstable','水不稳定（unstable）'));
    // 类别列表
    categories.forEach((c) => {
        targetSel.appendChild(makeOpt(c.category_name, c.category_name));
    });
    // 恢复选择
    if (prev && Array.from(targetSel.options).some(o => o.value === prev)) {
        targetSel.value = prev;
    } else {
        targetSel.value = '__background__';
    }
    // 设置当前颜色
    const setPicker = () => {
        const sel = targetSel.value;
        if (sel === '__background__') {
            picker.value = this.bgColor || '#787878';
        } else if (['adsorbent','non_adsorbent','stable','unstable'].includes(sel)) {
            if (!this.highlightColorMap) this.highlightColorMap = {};
            picker.value = this.highlightColorMap[sel] || '#1f77b4';
        } else if (sel === '__all__') {
            const name = categories[0]?.category_name;
            picker.value = (name && this.colorMap[name]) || '#1f77b4';
        } else {
            picker.value = this.colorMap[sel] || '#1f77b4';
        }
    };
    setPicker();
    // 更新选择时同步颜色选择器
    targetSel.onchange = setPicker;
}

// 保留空实现以兼容旧代码（不再使用高亮独立颜色）
VisualizationComponent.prototype.isHighlightKey = function(key) { return false; }
