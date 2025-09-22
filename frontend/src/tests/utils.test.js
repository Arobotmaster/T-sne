/**
 * 工具函数测试
 */

describe('工具函数测试', () => {
    test('应该能够正确创建模拟文件', () => {
        const file = createMockFile('test.csv', 'test,content', 'text/csv');

        expect(file).toBeValidFile();
        expect(file.name).toBe('test.csv');
        expect(file.type).toBe('text/csv');
    });

    test('应该能够正确创建模拟API响应', () => {
        const data = { success: true, data: { test: 'value' } };
        const response = createMockResponse(data);

        expect(response.ok).toBe(true);
        expect(response.status).toBe(200);
    });

    test('应该能够正确创建模拟错误响应', () => {
        const response = createMockErrorResponse('测试错误', 400);

        expect(response.ok).toBe(false);
        expect(response.status).toBe(400);
    });

    test('TEST_CONFIG应该包含正确的测试数据', () => {
        expect(TEST_CONFIG.API_BASE_URL).toBe('http://localhost:8000/api');
        expect(TEST_CONFIG.MOCK_DATA.SAMPLE_CSV).toContain('sample_id');
        expect(TEST_CONFIG.MOCK_DATA.SAMPLE_COORDINATES).toHaveLength(3);
    });
});

describe('DOM API模拟测试', () => {
    beforeEach(() => {
        // 清理document mock
        jest.clearAllMocks();
    });

    test('应该能够模拟document.getElementById', () => {
        const mockElement = { innerHTML: '', appendChild: jest.fn() };
        document.getElementById.mockReturnValue(mockElement);

        const element = document.getElementById('test-id');
        expect(element).toBe(mockElement);
        expect(document.getElementById).toHaveBeenCalledWith('test-id');
    });

    test('应该能够模拟document.querySelector', () => {
        const mockElement = { addEventListener: jest.fn() };
        document.querySelector.mockReturnValue(mockElement);

        const element = document.querySelector('.test-class');
        expect(element).toBe(mockElement);
        expect(document.querySelector).toHaveBeenCalledWith('.test-class');
    });

    test('应该能够模拟window.alert', () => {
        window.alert('测试消息');
        expect(window.alert).toHaveBeenCalledWith('测试消息');
    });
});

describe('Plotly.js模拟测试', () => {
    test('应该能够模拟Plotly.newPlot', () => {
        const container = { id: 'test-container' };
        const data = [{ x: [1, 2, 3], y: [4, 5, 6] }];
        const layout = { title: '测试图表' };

        Plotly.newPlot(container, data, layout);

        expect(Plotly.newPlot).toHaveBeenCalledWith(container, data, layout);
    });

    test('应该能够模拟Plotly.downloadImage', () => {
        const container = { id: 'test-container' };
        const format = 'png';
        const width = 800;
        const height = 600;

        Plotly.downloadImage(container, { format, width, height });

        expect(Plotly.downloadImage).toHaveBeenCalledWith(container, { format, width, height });
    });
});

describe('File API模拟测试', () => {
    test('应该能够创建File对象', () => {
        const content = 'test,csv,content\nrow1,value1';
        const file = new File([content], 'test.csv', { type: 'text/csv' });

        expect(file).toBeValidFile();
        expect(file.name).toBe('test.csv');
        expect(file.type).toBe('text/csv');
        expect(file.size).toBe(content.length);
    });

    test('应该能够模拟FileReader', () => {
        const reader = new FileReader();

        expect(typeof reader.readAsText).toBe('function');
        expect(typeof reader.onload).toBe('object');
    });
});

describe('Storage API模拟测试', () => {
    test('应该能够模拟localStorage', () => {
        localStorage.setItem('test-key', 'test-value');
        localStorage.getItem('test-key');

        expect(localStorage.setItem).toHaveBeenCalledWith('test-key', 'test-value');
        expect(localStorage.getItem).toHaveBeenCalledWith('test-key');
    });

    test('应该能够模拟sessionStorage', () => {
        sessionStorage.setItem('session-key', 'session-value');
        sessionStorage.getItem('session-key');

        expect(sessionStorage.setItem).toHaveBeenCalledWith('session-key', 'session-value');
        expect(sessionStorage.getItem).toHaveBeenCalledWith('session-key');
    });
});

describe('自定义匹配器测试', () => {
    test('toBeValidApiResponse匹配器应该正常工作', () => {
        const validResponse = { success: true, data: { test: 'value' } };
        const invalidResponse = { data: { test: 'value' } };

        expect(validResponse).toBeValidApiResponse();
        expect(invalidResponse).not.toBeValidApiResponse();
    });

    test('toBeValidFile匹配器应该正常工作', () => {
        const validFile = { name: 'test.csv', size: 1024, type: 'text/csv' };
        const invalidFile = { name: 'test.csv', size: 1024 };

        expect(validFile).toBeValidFile();
        expect(invalidFile).not.toBeValidFile();
    });
});