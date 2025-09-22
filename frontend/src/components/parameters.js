/**
 * 参数配置组件 - MOF数据t-SNE可视化应用
 * 提供PCA和t-SNE算法参数的配置界面
 */

import APIService from '../services/api.js';

class ParametersComponent {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Container element with id '${containerId}' not found`);
        }

        this.options = {
            defaultConfig: {
                // 嵌入选择（E1=12088，E2=4519）
                embedding: 'E2',
                preprocessing_config: {
                    missing_value_strategy: 'mean',
                    scaling_method: 'standard',
                    outlier_detection: true,
                    outlier_threshold: 3.0
                },
                pca_config: {
                    n_components: 50,
                    variance_retention: null,
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
            },
            showAdvanced: false,
            enablePresets: true,
            ...options
        };

        this.currentConfig = { ...this.options.defaultConfig };
        this.eventHandlers = {};
        this.api = new APIService();
        this.presets = this.initializePresets();

        this.init();
    }

    init() {
        this.createParameterInterface();
        this.setupEventListeners();
        this.loadSavedConfig();
    }

    createParameterInterface() {
        this.container.innerHTML = `
            <div class="parameters-container">
                <!-- 预设配置 -->
                <div class="presets-section" id="presets-section">
                    <div class="card">
                        <div class="card-header">
                            <h5><i class="bi bi-speedometer2"></i> 快速预设</h5>
                        </div>
                        <div class="card-body">
                            <!-- 嵌入选择 -->
                            <div class="mb-4">
                                <h6><i class="bi bi-layers"></i> 嵌入选择</h6>
                                <div class="row">
                                    <div class="col-md-6 mb-3">
                                        <label for="embedding-select" class="form-label">嵌入类型</label>
                                        <select class="form-select" id="embedding-select">
                                            <option value="E2">E2（4519，PLD>4.9）</option>
                                            <option value="E1">E1（12088，完整集）</option>
                                        </select>
                                        <div class="form-text">E2 更流畅，E1 更全面（单图展示）</div>
                                    </div>
                                </div>
                            </div>
                            <div class="row">
                                <div class="col-md-3 mb-2">
                                    <button type="button" class="btn btn-outline-primary btn-preset w-100" data-preset="balanced">
                                        <i class="bi bi-balance-scale"></i> 平衡模式
                                    </button>
                                </div>
                                <div class="col-md-3 mb-2">
                                    <button type="button" class="btn btn-outline-success btn-preset w-100" data-preset="speed">
                                        <i class="bi bi-lightning"></i> 快速模式
                                    </button>
                                </div>
                                <div class="col-md-3 mb-2">
                                    <button type="button" class="btn btn-outline-warning btn-preset w-100" data-preset="quality">
                                        <i class="bi bi-award"></i> 高质量模式
                                    </button>
                                </div>
                                <div class="col-md-3 mb-2">
                                    <button type="button" class="btn btn-outline-info btn-preset w-100" data-preset="exploration">
                                        <i class="bi bi-compass"></i> 探索模式
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 基础参数 -->
                <div class="basic-parameters">
                    <div class="card">
                        <div class="card-header">
                            <h5><i class="bi bi-sliders"></i> 基础参数</h5>
                        </div>
                        <div class="card-body">
                            <!-- PCA参数 -->
                            <div class="mb-4">
                                <h6><i class="bi bi-diagram-3"></i> PCA降维参数</h6>
                                <div class="row">
                                    <div class="col-md-6 mb-3">
                                        <label for="pca-components" class="form-label">主成分数量</label>
                                        <input type="number" class="form-control" id="pca-components"
                                               min="2" max="100" value="50">
                                        <div class="form-text">建议: 10-100，值越大保留信息越多</div>
                                    </div>
                                    <div class="col-md-6 mb-3">
                                        <label for="pca-variance" class="form-label">方差保留率 (%)</label>
                                        <input type="number" class="form-control" id="pca-variance"
                                               min="80" max="99" step="1" value="95">
                                        <div class="form-text">留空则使用固定主成分数量</div>
                                    </div>
                                </div>
                                <div class="form-check">
                                    <input class="form-check-input" type="checkbox" id="pca-whiten">
                                    <label class="form-check-label" for="pca-whiten">
                                        启用白化处理
                                    </label>
                                    <div class="form-text">白化可以改善t-SNE效果但会增加计算时间</div>
                                </div>
                            </div>

                            <!-- t-SNE参数 -->
                            <div class="mb-4">
                                <h6><i class="bi bi-graph-up"></i> t-SNE参数</h6>
                                <div class="row">
                                    <div class="col-md-6 mb-3">
                                        <label for="tsne-perplexity" class="form-label">Perplexity</label>
                                        <input type="number" class="form-control" id="tsne-perplexity"
                                               min="5" max="50" value="30">
                                        <div class="form-text">建议: 5-50，数据量大时值应该更大</div>
                                    </div>
                                    <div class="col-md-6 mb-3">
                                        <label for="tsne-learning-rate" class="form-label">学习率</label>
                                        <input type="number" class="form-control" id="tsne-learning-rate"
                                               min="10" max="1000" value="200">
                                        <div class="form-text">建议: 10-1000，通常使用200</div>
                                    </div>
                                </div>
                                <div class="row">
                                    <div class="col-md-6 mb-3">
                                        <label for="tsne-iterations" class="form-label">迭代次数</label>
                                        <input type="number" class="form-control" id="tsne-iterations"
                                               min="250" max="5000" value="1000">
                                        <div class="form-text">建议: 1000-2000，更多迭代更稳定</div>
                                    </div>
                                    <div class="col-md-6 mb-3">
                                        <label for="tsne-metric" class="form-label">距离度量</label>
                                        <select class="form-select" id="tsne-metric">
                                            <option value="euclidean">欧氏距离</option>
                                            <option value="manhattan">曼哈顿距离</option>
                                            <option value="chebyshev">切比雪夫距离</option>
                                            <option value="minkowski">闵可夫斯基距离</option>
                                        </select>
                                    </div>
                                </div>
                            </div>

                            <!-- 预处理参数 -->
                            <div class="mb-4">
                                <h6><i class="bi bi-funnel"></i> 数据预处理</h6>
                                <div class="row">
                                    <div class="col-md-6 mb-3">
                                        <label for="missing-value-strategy" class="form-label">缺失值处理</label>
                                        <select class="form-select" id="missing-value-strategy">
                                            <option value="mean">均值填充</option>
                                            <option value="median">中位数填充</option>
                                            <option value="mode">众数填充</option>
                                            <option value="interpolate">插值填充</option>
                                            <option value="drop">删除缺失行</option>
                                        </select>
                                    </div>
                                    <div class="col-md-6 mb-3">
                                        <label for="scaling-method" class="form-label">标准化方法</label>
                                        <select class="form-select" id="scaling-method">
                                            <option value="standard">Z-score标准化</option>
                                            <option value="minmax">Min-Max标准化</option>
                                            <option value="robust">Robust标准化</option>
                                            <option value="none">不标准化</option>
                                        </select>
                                    </div>
                                </div>
                                <div class="form-check">
                                    <input class="form-check-input" type="checkbox" id="outlier-detection" checked>
                                    <label class="form-check-label" for="outlier-detection">
                                        启用异常值检测
                                    </label>
                                </div>
                                <div class="row mt-2">
                                    <div class="col-md-6">
                                        <label for="outlier-threshold" class="form-label">异常值阈值 (Z-score)</label>
                                        <input type="number" class="form-control" id="outlier-threshold"
                                               min="1.5" max="5.0" step="0.1" value="3.0">
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 高级参数 -->
                <div class="advanced-parameters" id="advanced-parameters" style="display: none;">
                    <div class="card">
                        <div class="card-header">
                            <h5><i class="bi bi-gear"></i> 高级参数</h5>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-6 mb-3">
                                    <label for="pca-svd-solver" class="form-label">PCA求解器</label>
                                    <select class="form-select" id="pca-svd-solver">
                                        <option value="auto">自动选择</option>
                                        <option value="full">完整SVD</option>
                                        <option value="arpack">ARPACK</option>
                                        <option value="randomized">随机化SVD</option>
                                    </select>
                                </div>
                                <div class="col-md-6 mb-3">
                                    <label for="tsne-angle" class="form-label">t-SNE角度</label>
                                    <input type="number" class="form-control" id="tsne-angle"
                                           min="0.1" max="0.8" step="0.1" value="0.5">
                                    <div class="form-text">控制速度-精度权衡</div>
                                </div>
                            </div>
                            <div class="row">
                                <div class="col-md-6 mb-3">
                                    <label for="tsne-early-exaggeration" class="form-label">早期放大因子</label>
                                    <input type="number" class="form-control" id="tsne-early-exaggeration"
                                           min="1" max="100" value="12">
                                    <div class="form-text">控制簇间间距</div>
                                </div>
                                <div class="col-md-6 mb-3">
                                    <label for="random-seed" class="form-label">随机种子</label>
                                    <input type="number" class="form-control" id="random-seed" value="42">
                                    <div class="form-text">设为-1使用随机种子</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 操作按钮 -->
                <div class="action-buttons mt-3">
                    <div class="row">
                        <div class="col-md-3">
                            <button type="button" class="btn btn-outline-secondary w-100" id="btn-toggle-advanced">
                                <i class="bi bi-gear"></i> 高级参数
                            </button>
                        </div>
                        <div class="col-md-3">
                            <button type="button" class="btn btn-outline-success w-100" id="btn-auto-tune">
                                <i class="bi bi-magic"></i> 自动调参
                            </button>
                        </div>
                        <div class="col-md-3">
                            <button type="button" class="btn btn-outline-warning w-100" id="btn-reset-defaults">
                                <i class="bi bi-arrow-counterclockwise"></i> 重置默认
                            </button>
                        </div>
                        <div class="col-md-3">
                            <button type="button" class="btn btn-primary w-100" id="btn-apply-parameters">
                                <i class="bi bi-check-circle"></i> 应用参数
                            </button>
                        </div>
                    </div>
                </div>

                <!-- 参数预览 -->
                <div class="parameter-preview mt-3">
                    <div class="card">
                        <div class="card-header">
                            <h6><i class="bi bi-eye"></i> 当前配置预览</h6>
                        </div>
                        <div class="card-body">
                            <pre id="config-preview" class="config-preview"></pre>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    setupEventListeners() {
        // 预设按钮
        this.container.querySelectorAll('.btn-preset').forEach(btn => {
            btn.addEventListener('click', (e) => this.applyPreset(e.target.dataset.preset));
        });

        // 高级参数切换
        this.container.querySelector('#btn-toggle-advanced').addEventListener('click', () => {
            this.toggleAdvanced();
        });

        // 自动调参
        this.container.querySelector('#btn-auto-tune').addEventListener('click', async () => {
            await this.autoTune();
        });

        // 重置默认值
        this.container.querySelector('#btn-reset-defaults').addEventListener('click', () => {
            this.resetToDefaults();
        });

        // 应用参数
        this.container.querySelector('#btn-apply-parameters').addEventListener('click', () => {
            this.applyParameters();
        });

        // 参数变化监听
        this.setupParameterListeners();
    }

    async autoTune() {
        try {
            const app = window.mofApp;
            const ds = app && typeof app.getCurrentDataset === 'function' ? app.getCurrentDataset() : null;
            if (!ds || !ds.dataset_id) {
                alert('请先选择或上传数据集');
                return;
            }

            // 调用后端网格搜索
            const resp = await this.api.autoTuneDataset(ds.dataset_id, { max_search_rows: 5000 });
            if (!resp || resp.success !== true) {
                alert('自动调参失败，请稍后再试');
                return;
            }
            const rc = resp.data?.recommended_config;
            if (!rc) {
                alert('自动调参未返回结果');
                return;
            }

            this.currentConfig = {
                embedding: this.currentConfig.embedding || 'E2',
                preprocessing_config: rc.preprocessing_config || this.currentConfig.preprocessing_config,
                pca_config: rc.pca_config || this.currentConfig.pca_config,
                tsne_config: rc.tsne_config || this.currentConfig.tsne_config
            };
            this.updateUIFromConfig();
            this.updateConfigPreview();

            // 结果提示（Top3 展示 KL）
            const results = Array.isArray(resp.data?.results) ? resp.data.results : [];
            // 保存调参摘要到当前配置，后端会落盘 tuning_summary.json
            this.currentConfig.tuning_summary = {
                results,
                search_space: resp.data?.search_space,
                recommended: {
                    pca_config: rc.pca_config || null,
                    tsne_config: rc.tsne_config || null
                }
            };
            const sorted = results.filter(r => r.success).sort((a,b) => (a.kl_divergence||1e9) - (b.kl_divergence||1e9));
            let msg = '自动调参完成。\n推荐：' +
                `PCA=${rc.pca_config?.n_components}, ` +
                `${rc.tsne_config?.metric||''}, perp=${rc.tsne_config?.perplexity}, ` +
                `lr=${rc.tsne_config?.learning_rate}, ex=${rc.tsne_config?.early_exaggeration}`;
            if (sorted.length) {
                const top3 = sorted.slice(0,3).map(r => `PCA=${r.pca_n}, ${r.metric}, perp=${r.perplexity}, lr=${r.learning_rate}, ex=${r.early_exaggeration}, KL=${Number(r.kl_divergence).toFixed(4)}`).join(' | ');
                msg += `\nTop3: ${top3}`;
            }
            if (window.mofApp && typeof window.mofApp.showNotification === 'function') {
                window.mofApp.showNotification(msg.replace(/\n/g,'<br/>'), 'info');
            } else {
                console.log(msg);
            }
            const confirmed = confirm('是否立即应用推荐参数并开始处理？');
            if (confirmed) this.applyParameters();
        } catch (e) {
            console.error('自动调参失败', e);
            alert('自动调参失败，请稍后再试');
        }
    }

    setupParameterListeners() {
        const inputs = this.container.querySelectorAll('input, select');
        inputs.forEach(input => {
            input.addEventListener('change', () => this.updateConfigFromUI());
            input.addEventListener('input', () => this.updateConfigFromUI());
        });
    }

    updateConfigFromUI() {
        // 嵌入类型
        const embeddingSelect = this.container.querySelector('#embedding-select');
        if (embeddingSelect) {
            this.currentConfig.embedding = embeddingSelect.value;
        }
        // PCA配置
        this.currentConfig.pca_config.n_components = parseInt(this.container.querySelector('#pca-components').value);
        this.currentConfig.pca_config.variance_retention = this.container.querySelector('#pca-variance').value ?
            parseFloat(this.container.querySelector('#pca-variance').value) / 100 : null;
        this.currentConfig.pca_config.whiten = this.container.querySelector('#pca-whiten').checked;

        // t-SNE配置
        this.currentConfig.tsne_config.perplexity = parseFloat(this.container.querySelector('#tsne-perplexity').value);
        this.currentConfig.tsne_config.learning_rate = parseFloat(this.container.querySelector('#tsne-learning-rate').value);
        this.currentConfig.tsne_config.n_iter = parseInt(this.container.querySelector('#tsne-iterations').value);
        this.currentConfig.tsne_config.metric = this.container.querySelector('#tsne-metric').value;

        // 预处理配置
        this.currentConfig.preprocessing_config.missing_value_strategy = this.container.querySelector('#missing-value-strategy').value;
        this.currentConfig.preprocessing_config.scaling_method = this.container.querySelector('#scaling-method').value;
        this.currentConfig.preprocessing_config.outlier_detection = this.container.querySelector('#outlier-detection').checked;
        this.currentConfig.preprocessing_config.outlier_threshold = parseFloat(this.container.querySelector('#outlier-threshold').value);

        // 随机种子
        const randomSeed = parseInt(this.container.querySelector('#random-seed').value);
        if (randomSeed !== -1) {
            this.currentConfig.pca_config.random_state = randomSeed;
            this.currentConfig.tsne_config.random_state = randomSeed;
        }

        this.updateConfigPreview();
        this.emit('parameterChange', this.currentConfig);
    }

    updateConfigPreview() {
        const preview = this.container.querySelector('#config-preview');
        const configCopy = JSON.parse(JSON.stringify(this.currentConfig, null, 2));
        preview.textContent = JSON.stringify(configCopy, null, 2);
    }

    toggleAdvanced() {
        const advancedSection = this.container.querySelector('#advanced-parameters');
        const btn = this.container.querySelector('#btn-toggle-advanced');

        if (advancedSection.style.display === 'none') {
            advancedSection.style.display = 'block';
            btn.innerHTML = '<i class="bi bi-gear-fill"></i> 隐藏高级';
        } else {
            advancedSection.style.display = 'none';
            btn.innerHTML = '<i class="bi bi-gear"></i> 高级参数';
        }
    }

    applyPreset(presetName) {
        const preset = this.presets[presetName];
        if (!preset) return;

        this.currentConfig = JSON.parse(JSON.stringify(preset));
        this.updateUIFromConfig();
        this.updateConfigPreview();

        // 高亮选中的预设按钮
        this.container.querySelectorAll('.btn-preset').forEach(btn => {
            btn.classList.remove('active');
        });
        this.container.querySelector(`[data-preset="${presetName}"]`).classList.add('active');

        this.emit('presetApplied', { preset: presetName, config: this.currentConfig });
    }

    updateUIFromConfig() {
        // 嵌入类型
        const embeddingSelect = this.container.querySelector('#embedding-select');
        if (embeddingSelect) {
            embeddingSelect.value = this.currentConfig.embedding || 'E2';
        }
        // 更新PCA参数
        this.container.querySelector('#pca-components').value = this.currentConfig.pca_config.n_components;
        this.container.querySelector('#pca-variance').value = this.currentConfig.pca_config.variance_retention ?
            this.currentConfig.pca_config.variance_retention * 100 : '';
        this.container.querySelector('#pca-whiten').checked = this.currentConfig.pca_config.whiten;

        // 更新t-SNE参数
        this.container.querySelector('#tsne-perplexity').value = this.currentConfig.tsne_config.perplexity;
        this.container.querySelector('#tsne-learning-rate').value = this.currentConfig.tsne_config.learning_rate;
        this.container.querySelector('#tsne-iterations').value = this.currentConfig.tsne_config.n_iter;
        this.container.querySelector('#tsne-metric').value = this.currentConfig.tsne_config.metric;

        // 更新预处理参数
        this.container.querySelector('#missing-value-strategy').value = this.currentConfig.preprocessing_config.missing_value_strategy;
        this.container.querySelector('#scaling-method').value = this.currentConfig.preprocessing_config.scaling_method;
        this.container.querySelector('#outlier-detection').checked = this.currentConfig.preprocessing_config.outlier_detection;
        this.container.querySelector('#outlier-threshold').value = this.currentConfig.preprocessing_config.outlier_threshold;

        // 更新随机种子
        const seed = this.currentConfig.pca_config.random_state;
        this.container.querySelector('#random-seed').value = seed;
    }

    resetToDefaults() {
        this.currentConfig = JSON.parse(JSON.stringify(this.options.defaultConfig));
        this.updateUIFromConfig();
        this.updateConfigPreview();

        // 清除预设按钮高亮
        this.container.querySelectorAll('.btn-preset').forEach(btn => {
            btn.classList.remove('active');
        });

        this.emit('parametersReset', this.currentConfig);
    }

    applyParameters() {
        this.updateConfigFromUI();
        this.saveConfig();
        this.emit('parametersApplied', this.currentConfig);
    }

    initializePresets() {
        return {
            balanced: {
                preprocessing_config: {
                    missing_value_strategy: 'mean',
                    scaling_method: 'standard',
                    outlier_detection: true,
                    outlier_threshold: 3.0
                },
                pca_config: {
                    n_components: 50,
                    variance_retention: null,
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
            },
            speed: {
                preprocessing_config: {
                    missing_value_strategy: 'mean',
                    scaling_method: 'standard',
                    outlier_detection: false,
                    outlier_threshold: 3.0
                },
                pca_config: {
                    n_components: 30,
                    variance_retention: null,
                    whiten: false,
                    random_state: 42
                },
                tsne_config: {
                    perplexity: 20,
                    n_components: 2,
                    learning_rate: 200,
                    n_iter: 500,
                    random_state: 42,
                    metric: 'euclidean'
                }
            },
            quality: {
                preprocessing_config: {
                    missing_value_strategy: 'median',
                    scaling_method: 'robust',
                    outlier_detection: true,
                    outlier_threshold: 2.5
                },
                pca_config: {
                    n_components: 80,
                    variance_retention: null,
                    whiten: true,
                    random_state: 42
                },
                tsne_config: {
                    perplexity: 40,
                    n_components: 2,
                    learning_rate: 200,
                    n_iter: 2000,
                    random_state: 42,
                    metric: 'euclidean'
                }
            },
            exploration: {
                preprocessing_config: {
                    missing_value_strategy: 'mean',
                    scaling_method: 'standard',
                    outlier_detection: true,
                    outlier_threshold: 3.5
                },
                pca_config: {
                    n_components: 60,
                    variance_retention: null,
                    whiten: false,
                    random_state: 42
                },
                tsne_config: {
                    perplexity: 35,
                    n_components: 2,
                    learning_rate: 150,
                    n_iter: 1500,
                    random_state: 42,
                    metric: 'manhattan'
                }
            }
        };
    }

    saveConfig() {
        try {
            localStorage.setItem('mof-viz-params', JSON.stringify(this.currentConfig));
        } catch (error) {
            console.warn('Failed to save parameters to localStorage:', error);
        }
    }

    loadSavedConfig() {
        try {
            const saved = localStorage.getItem('mof-viz-params');
            if (saved) {
                this.currentConfig = JSON.parse(saved);
                this.updateUIFromConfig();
                this.updateConfigPreview();
            }
        } catch (error) {
            console.warn('Failed to load parameters from localStorage:', error);
        }
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
    getConfig() {
        return { ...this.currentConfig };
    }

    setConfig(config) {
        this.currentConfig = { ...config };
        this.updateUIFromConfig();
        this.updateConfigPreview();
    }

    reset() {
        this.resetToDefaults();
    }

    destroy() {
        this.container.innerHTML = '';
        this.eventHandlers = {};
    }
}

export default ParametersComponent;
