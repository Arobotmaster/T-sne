/**
 * 前端测试环境设置
 *
 * 为Jest测试提供DOM环境和全局配置
 */

// 模拟浏览器API
global.fetch = jest.fn();

// 模拟DOM API
global.document = {
    createElement: jest.fn(() => ({
        appendChild: jest.fn(),
        setAttribute: jest.fn(),
        addEventListener: jest.fn(),
        classList: { add: jest.fn(), remove: jest.fn(), toggle: jest.fn() },
        style: {},
        innerHTML: '',
        textContent: '',
        value: '',
        click: jest.fn(),
    })),
    getElementById: jest.fn(() => ({
        appendChild: jest.fn(),
        setAttribute: jest.fn(),
        addEventListener: jest.fn(),
        classList: { add: jest.fn(), remove: jest.fn(), toggle: jest.fn() },
        style: {},
        innerHTML: '',
        textContent: '',
        value: '',
        click: jest.fn(),
    })),
    querySelector: jest.fn(() => ({
        appendChild: jest.fn(),
        setAttribute: jest.fn(),
        addEventListener: jest.fn(),
        classList: { add: jest.fn(), remove: jest.fn(), toggle: jest.fn() },
        style: {},
        innerHTML: '',
        textContent: '',
        value: '',
        click: jest.fn(),
    })),
    querySelectorAll: jest.fn(() => []),
    body: {
        appendChild: jest.fn(),
        innerHTML: '',
    },
};

global.window = {
    location: { href: 'http://localhost:3000' },
    alert: jest.fn(),
    setTimeout: jest.fn(),
    clearTimeout: jest.fn(),
    setInterval: jest.fn(),
    clearInterval: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    localStorage: {
        getItem: jest.fn(),
        setItem: jest.fn(),
        removeItem: jest.fn(),
        clear: jest.fn(),
    },
    sessionStorage: {
        getItem: jest.fn(),
        setItem: jest.fn(),
        removeItem: jest.fn(),
        clear: jest.fn(),
    },
};

// 模拟console方法
global.console = {
    log: jest.fn(),
    error: jest.fn(),
    warn: jest.fn(),
    info: jest.fn(),
    debug: jest.fn(),
};

// 模拟File API
global.File = jest.fn((parts, name, options) => ({
    parts,
    name,
    options,
    size: parts.reduce((acc, part) => acc + part.length, 0),
    type: options?.type || 'text/plain',
}));

global.FileReader = jest.fn(() => ({
    readAsText: jest.fn(),
    readAsDataURL: jest.fn(),
    onload: null,
    onerror: null,
}));

// 模拟FormData
global.FormData = jest.fn(() => ({
    append: jest.fn(),
    get: jest.fn(),
    getAll: jest.fn(),
    set: jest.fn(),
    delete: jest.fn(),
    has: jest.fn(),
    keys: jest.fn(() => []),
    values: jest.fn(() => []),
    entries: jest.fn(() => []),
}));

// 模拟Blob
global.Blob = jest.fn((parts, options) => ({
    parts,
    options,
    size: parts.reduce((acc, part) => acc + part.length, 0),
    type: options?.type || 'text/plain',
    slice: jest.fn(),
}));

// 模拟Plotly.js
global.Plotly = {
    newPlot: jest.fn(),
    update: jest.fn(),
    react: jest.fn(),
    downloadImage: jest.fn(),
    toImage: jest.fn(),
    purge: jest.fn(),
    redraw: jest.fn(),
    extendTraces: jest.fn(),
    prependTraces: jest.fn(),
    deleteTraces: jest.fn(),
    moveTraces: jest.fn(),
    restyle: jest.fn(),
    relayout: jest.fn(),
    updateTraces: jest.fn(),
    addFrames: jest.fn(),
    addTraces: jest.fn(),
    animate: jest.fn(),
    onPlotlyEvent: jest.fn(),
    offPlotlyEvent: jest.fn(),
};

// 模拟Bootstrap
global.bootstrap = {
    Tooltip: jest.fn(),
    Popover: jest.fn(),
    Modal: jest.fn(),
    Dropdown: jest.fn(),
    Collapse: jest.fn(),
    Tab: jest.fn(),
    Alert: jest.fn(),
    Button: jest.fn(),
    Carousel: jest.fn(),
    Offcanvas: jest.fn(),
    Toast: jest.fn(),
};

// 测试工具函数
global.createMockFile = (name, content, type = 'text/csv') => {
    const blob = new Blob([content], { type });
    return new File([blob], name, { type });
};

global.createMockResponse = (data, status = 200, statusText = 'OK') => ({
    ok: status >= 200 && status < 300,
    status,
    statusText,
    json: () => Promise.resolve(data),
    text: () => Promise.resolve(JSON.stringify(data)),
    blob: () => Promise.resolve(new Blob([JSON.stringify(data)])),
    arrayBuffer: () => Promise.resolve(new TextEncoder().encode(JSON.stringify(data))),
});

global.createMockErrorResponse = (message, status = 400) => ({
    ok: false,
    status,
    statusText: 'Error',
    json: () => Promise.resolve({ error: message }),
    text: () => Promise.resolve(JSON.stringify({ error: message })),
});

// 模拟CSSOM
global.CSS = {
    supports: jest.fn(),
    escape: jest.fn(),
};

// 模拟性能API
global.performance = {
    now: jest.fn(() => Date.now()),
    mark: jest.fn(),
    measure: jest.fn(),
    getEntriesByType: jest.fn(() => []),
    getEntriesByName: jest.fn(() => []),
    clearMarks: jest.fn(),
    clearMeasures: jest.fn(),
};

// 设置测试超时
jest.setTimeout(10000);

// 全局测试清理
beforeEach(() => {
    // 清理所有mock调用
    jest.clearAllMocks();

    // 重置fetch mock
    global.fetch.mockClear();
});

afterEach(() => {
    // 清理测试后的状态
    if (global.document.body.innerHTML) {
        global.document.body.innerHTML = '';
    }
});

// 添加自定义匹配器
expect.extend({
    toBeValidApiResponse(received) {
        const isValid = received &&
                       typeof received === 'object' &&
                       'success' in received &&
                       typeof received.success === 'boolean';

        if (isValid) {
            return {
                message: () => `expected ${received} not to be a valid API response`,
                pass: true,
            };
        } else {
            return {
                message: () => `expected ${received} to be a valid API response with success property`,
                pass: false,
            };
        }
    },

    toBeValidFile(received) {
        const isValid = received &&
                       typeof received === 'object' &&
                       'name' in received &&
                       'size' in received &&
                       'type' in received;

        if (isValid) {
            return {
                message: () => `expected ${received} not to be a valid file object`,
                pass: true,
            };
        } else {
            return {
                message: () => `expected ${received} to be a valid file object with name, size, and type properties`,
                pass: false,
            };
        }
    },
});

// 设置全局测试配置
global.TEST_CONFIG = {
    API_BASE_URL: 'http://localhost:8000/api',
    TIMEOUT: 5000,
    MOCK_DATA: {
        SAMPLE_CSV: `sample_id,feature_1,feature_2,feature_3,category
sample_001,1.2,3.4,5.6,MOF_A
sample_002,2.1,4.3,6.5,MOF_B
sample_003,1.5,3.2,5.1,MOF_A`,
        SAMPLE_COORDINATES: [
            { x: 1.2, y: 3.4, sample_id: 'sample_001', category_id: 1, category_name: 'MOF_A' },
            { x: 2.1, y: 4.3, sample_id: 'sample_002', category_id: 2, category_name: 'MOF_B' },
            { x: 1.5, y: 3.2, sample_id: 'sample_003', category_id: 1, category_name: 'MOF_A' },
        ],
        SAMPLE_DATASET: {
            dataset_id: 'test_dataset_001',
            filename: 'test_data.csv',
            file_size: 1024,
            row_count: 100,
            column_count: 10,
            data_quality_score: 0.95,
            columns: [
                { name: 'sample_id', type: 'string', is_id: true },
                { name: 'feature_1', type: 'numeric', is_feature: true },
                { name: 'feature_2', type: 'numeric', is_feature: true },
                { name: 'category', type: 'string', is_category: true },
            ],
        },
    },
};

console.log('前端测试环境设置完成');