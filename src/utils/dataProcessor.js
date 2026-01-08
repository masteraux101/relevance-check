/**
 * Data processing module - Candlestick data fetching, preprocessing, normalization
 */

// Mock data generator (can be replaced with real API in production)
export class DataProcessor {
    constructor() {
        console.log('[DataProcessor] 📊 Initializing data processor');
        this.cache = new Map();
        this.normalizers = new Map();
        this.csvData = new Map();  // Store user-uploaded CSV data
    }

    /**
     * Parse yfinance format CSV file
     * yfinance CSV format: Date,Open,High,Low,Close,Adj Close,Volume
     * @param {string} csvContent - CSV file content
     * @param {string} symbol - Data identifier (e.g. filename)
     * @returns {Array} Parsed data array
     */
    parseYFinanceCSV(csvContent, symbol = 'CUSTOM') {
        console.log('[DataProcessor] parseYFinanceCSV() Parsing CSV data...');
        console.log('[DataProcessor] Data identifier:', symbol);
        
        const lines = csvContent.trim().split('\n');
        if (lines.length < 2) {
            throw new Error('Insufficient CSV data, at least header and one row needed');
        }
        
        // Parse header
        const header = lines[0].split(',').map(h => h.trim().toLowerCase());
        console.log('[DataProcessor] CSV header:', header);
        
        // Find column indices - support multiple formats
        const findColumnIndex = (names) => {
            for (const name of names) {
                const idx = header.indexOf(name.toLowerCase());
                if (idx !== -1) return idx;
            }
            return -1;
        };
        
        const dateIdx = findColumnIndex(['date', 'datetime', 'time', 'timestamp']);
        const openIdx = findColumnIndex(['open', 'open price']);
        const highIdx = findColumnIndex(['high', 'high price']);
        const lowIdx = findColumnIndex(['low', 'low price']);
        const closeIdx = findColumnIndex(['close', 'close price', 'adj close', 'adjusted close']);
        const volumeIdx = findColumnIndex(['volume', 'vol']);
        
        console.log('[DataProcessor] Column indices:', { dateIdx, openIdx, highIdx, lowIdx, closeIdx, volumeIdx });
        
        // 验证必需列
        if (openIdx === -1 || highIdx === -1 || lowIdx === -1 || closeIdx === -1) {
            throw new Error(`CSV缺少必需列。需要: Open, High, Low, Close。找到: ${header.join(', ')}`);
        }
        
        const data = [];
        const errors = [];
        
        for (let i = 1; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line) continue;
            
            // 处理可能包含逗号的字段（如带引号的字段）
            const values = this.parseCSVLine(line);
            
            try {
                const open = parseFloat(values[openIdx]);
                const high = parseFloat(values[highIdx]);
                const low = parseFloat(values[lowIdx]);
                const close = parseFloat(values[closeIdx]);
                const volume = volumeIdx !== -1 ? parseInt(values[volumeIdx]) || 0 : 0;
                
                // 验证数值有效性
                if (isNaN(open) || isNaN(high) || isNaN(low) || isNaN(close)) {
                    errors.push(`第${i + 1}行数据无效`);
                    continue;
                }
                
                // 处理日期
                let dateStr;
                if (dateIdx !== -1 && values[dateIdx]) {
                    dateStr = values[dateIdx].trim();
                    // 处理各种日期格式
                    if (dateStr.includes('/')) {
                        const parts = dateStr.split('/');
                        if (parts.length === 3) {
                            dateStr = `${parts[2]}-${parts[0].padStart(2, '0')}-${parts[1].padStart(2, '0')}`;
                        }
                    }
                } else {
                    // 如果没有日期列，使用索引生成日期
                    const date = new Date();
                    date.setDate(date.getDate() - (lines.length - 1 - i));
                    dateStr = date.toISOString().split('T')[0];
                }
                
                data.push({
                    date: dateStr,
                    timestamp: new Date(dateStr).getTime(),
                    open: Math.max(0.01, open),
                    high: Math.max(0.01, high),
                    low: Math.max(0.01, low),
                    close: Math.max(0.01, close),
                    volume: Math.max(0, volume)
                });
            } catch (e) {
                errors.push(`第${i + 1}行解析错误: ${e.message}`);
            }
        }
        
        if (errors.length > 0) {
            console.warn('[DataProcessor] 解析警告:', errors.slice(0, 5));
        }
        
        if (data.length === 0) {
            throw new Error('没有成功解析任何数据行');
        }
        
        // 按日期排序（升序）
        data.sort((a, b) => a.timestamp - b.timestamp);
        
        console.log(`[DataProcessor] ✅ 成功解析 ${data.length} 行数据`);
        console.log('[DataProcessor] 日期范围:', data[0].date, '至', data[data.length - 1].date);
        console.log('[DataProcessor] 价格范围:', 
            Math.min(...data.map(d => d.low)).toFixed(2), '至',
            Math.max(...data.map(d => d.high)).toFixed(2)
        );
        
        return data;
    }

    /**
     * 解析单行 CSV（处理引号内的逗号）
     */
    parseCSVLine(line) {
        const result = [];
        let current = '';
        let inQuotes = false;
        
        for (let i = 0; i < line.length; i++) {
            const char = line[i];
            if (char === '"') {
                inQuotes = !inQuotes;
            } else if (char === ',' && !inQuotes) {
                result.push(current.trim());
                current = '';
            } else {
                current += char;
            }
        }
        result.push(current.trim());
        
        return result;
    }

    /**
     * 从 File 对象读取并解析 CSV
     * @param {File} file - 文件对象
     * @returns {Promise<{symbol: string, data: Array}>}
     */
    async loadCSVFile(file) {
        console.log('[DataProcessor] loadCSVFile():', file.name);
        
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = (e) => {
                try {
                    const content = e.target.result;
                    // 从文件名提取 symbol（去掉扩展名）
                    const symbol = file.name.replace(/\.csv$/i, '').toUpperCase();
                    const data = this.parseYFinanceCSV(content, symbol);
                    
                    // 存储到 csvData
                    this.csvData.set(symbol, data);
                    // 同时存入 cache 以便后续使用
                    this.cache.set(`${symbol}_csv`, data);
                    
                    resolve({ symbol, data, rowCount: data.length });
                } catch (error) {
                    reject(error);
                }
            };
            
            reader.onerror = () => {
                reject(new Error(`读取文件失败: ${file.name}`));
            };
            
            reader.readAsText(file);
        });
    }

    /**
     * 批量加载多个 CSV 文件
     * @param {FileList|Array<File>} files - 文件列表
     * @returns {Promise<Array<{symbol: string, data: Array, rowCount: number}>>}
     */
    async loadMultipleCSVFiles(files) {
        console.log('[DataProcessor] loadMultipleCSVFiles():', files.length, '个文件');
        
        const results = [];
        for (const file of files) {
            try {
                const result = await this.loadCSVFile(file);
                results.push(result);
            } catch (error) {
                console.error(`[DataProcessor] 加载 ${file.name} 失败:`, error);
                results.push({ 
                    symbol: file.name.replace(/\.csv$/i, '').toUpperCase(), 
                    error: error.message 
                });
            }
        }
        
        return results;
    }

    /**
     * 获取已加载的 CSV 数据
     * @param {string} symbol - 数据标识符
     * @returns {Array|null}
     */
    getCSVData(symbol) {
        return this.csvData.get(symbol.toUpperCase()) || null;
    }

    /**
     * 获取所有已加载的 CSV 符号
     * @returns {Array<string>}
     */
    getLoadedCSVSymbols() {
        return Array.from(this.csvData.keys());
    }

    /**
     * 清除所有 CSV 数据
     */
    clearCSVData() {
        console.log('[DataProcessor] clearCSVData() 清除所有CSV数据');
        this.csvData.clear();
        // 清除 cache 中的 csv 数据
        for (const key of this.cache.keys()) {
            if (key.endsWith('_csv')) {
                this.cache.delete(key);
            }
        }
    }

    /**
     * 使用 CSV 数据准备训练数据
     * @param {string} targetSymbol - 预测目标符号
     * @param {Array<string>} relatedSymbols - 关联数据符号列表
     * @param {number} lookback - 回看窗口
     * @param {number} forecast - 预测长度
     */
    prepareTrainingDataFromCSV(targetSymbol, relatedSymbols, lookback, forecast) {
        console.log('[DataProcessor] prepareTrainingDataFromCSV()');
        console.log('[DataProcessor] 目标:', targetSymbol, '关联:', relatedSymbols);
        
        // 收集所有数据源
        const multiSourceData = {};
        
        // 添加目标数据
        const targetData = this.csvData.get(targetSymbol.toUpperCase());
        if (!targetData) {
            throw new Error(`找不到目标数据: ${targetSymbol}`);
        }
        multiSourceData[targetSymbol] = targetData;
        
        // 添加关联数据
        for (const symbol of relatedSymbols) {
            const data = this.csvData.get(symbol.toUpperCase());
            if (data) {
                multiSourceData[symbol] = data;
            } else {
                console.warn(`[DataProcessor] 找不到关联数据: ${symbol}，跳过`);
            }
        }
        
        // 使用现有的 prepareTrainingData 方法
        return this.prepareTrainingData(multiSourceData, targetSymbol, lookback, forecast);
    }

    /**
     * 生成模拟K线数据（使用几何布朗运动模型）
     * @param {string} symbol - 股票代码
     * @param {number} days - 天数
     * @param {object} params - 参数配置
     */
    generateMockData(symbol, days, params = {}) {
        console.log('[DataProcessor] generateMockData():', { symbol, days, params });
        const basePrice = params.basePrice || 100;
        const volatility = params.volatility || 0.02;
        const drift = params.drift || 0.0001;
        
        const data = [];
        let price = basePrice;
        const startDate = new Date();
        startDate.setDate(startDate.getDate() - days);

        for (let i = 0; i < days; i++) {
            const date = new Date(startDate);
            date.setDate(date.getDate() + i);
            
            // 几何布朗运动
            const randomShock = (Math.random() - 0.5) * 2 * volatility;
            const dailyReturn = drift + randomShock;
            
            const open = price;
            const change = price * dailyReturn;
            const high = open + Math.abs(change) * (1 + Math.random() * 0.5);
            const low = open - Math.abs(change) * (1 + Math.random() * 0.5);
            const close = open + change;
            const volume = Math.floor(1000000 * (0.5 + Math.random()));

            price = close;

            data.push({
                date: date.toISOString().split('T')[0],
                timestamp: date.getTime(),
                open: Math.max(0.01, open),
                high: Math.max(0.01, high),
                low: Math.max(0.01, low),
                close: Math.max(0.01, close),
                volume: volume
            });
        }

        return data;
    }

    /**
     * 获取股票数据（使用模拟数据或API）
     * @param {string} symbol - 股票代码
     * @param {number} days - 天数
     */
    async fetchData(symbol, days) {
        console.log('[DataProcessor] fetchData():', { symbol, days });
        
        // 检查缓存
        const cacheKey = `${symbol}_${days}`;
        if (this.cache.has(cacheKey)) {
            console.log('[DataProcessor] 使用缓存数据:', cacheKey);
            return this.cache.get(cacheKey);
        }
        console.log('[DataProcessor] 生成新数据:', cacheKey);

        // 配置不同标的的参数
        const symbolConfigs = {
            'SPY': { basePrice: 450, volatility: 0.012, drift: 0.0003 },
            'QQQ': { basePrice: 380, volatility: 0.015, drift: 0.0004 },
            'VIX': { basePrice: 18, volatility: 0.08, drift: -0.001 },
            'TLT': { basePrice: 95, volatility: 0.01, drift: 0.0001 },
            'GLD': { basePrice: 180, volatility: 0.008, drift: 0.0002 },
            'DXY': { basePrice: 104, volatility: 0.005, drift: 0.0001 }
        };

        const config = symbolConfigs[symbol] || { basePrice: 100, volatility: 0.02, drift: 0.0001 };
        
        // 添加跨市场相关性
        let data = this.generateMockData(symbol, days, config);
        
        // 如果有SPY数据，添加相关性
        if (symbol !== 'SPY' && this.cache.has(`SPY_${days}`)) {
            const spyData = this.cache.get(`SPY_${days}`);
            data = this.addCorrelation(data, spyData, symbol);
        }

        this.cache.set(cacheKey, data);
        return data;
    }

    /**
     * 添加跨市场相关性
     */
    addCorrelation(targetData, spyData, symbol) {
        // 不同标的与SPY的相关性
        const correlations = {
            'VIX': -0.7,   // VIX与SPY负相关
            'TLT': -0.3,   // 国债与SPY轻微负相关
            'GLD': 0.2,    // 黄金与SPY轻微正相关
            'DXY': -0.2,   // 美元与SPY轻微负相关
            'QQQ': 0.9     // QQQ与SPY高度正相关
        };

        const corr = correlations[symbol] || 0;
        
        return targetData.map((item, i) => {
            if (i < spyData.length) {
                const spyReturn = i > 0 
                    ? (spyData[i].close - spyData[i-1].close) / spyData[i-1].close 
                    : 0;
                
                // 添加相关性影响
                const correlatedReturn = corr * spyReturn * 0.5;
                const factor = 1 + correlatedReturn;
                
                return {
                    ...item,
                    open: item.open * factor,
                    high: item.high * factor,
                    low: item.low * factor,
                    close: item.close * factor
                };
            }
            return item;
        });
    }

    /**
     * 提取OHLCV特征
     * @param {Array} data - K线数据
     */
    extractFeatures(data) {
        return data.map(item => [
            item.open,
            item.high,
            item.low,
            item.close,
            item.volume / 1000000  // 标准化成交量
        ]);
    }

    /**
     * 计算技术指标
     * @param {Array} data - K线数据
     */
    calculateIndicators(data) {
        const closes = data.map(d => d.close);
        
        // 计算移动平均
        const ma5 = this.movingAverage(closes, 5);
        const ma10 = this.movingAverage(closes, 10);
        const ma20 = this.movingAverage(closes, 20);
        
        // 计算RSI
        const rsi = this.calculateRSI(closes, 14);
        
        // 计算MACD
        const macd = this.calculateMACD(closes);
        
        // 计算布林带
        const bollinger = this.calculateBollinger(closes, 20);

        return data.map((item, i) => ({
            ...item,
            ma5: ma5[i],
            ma10: ma10[i],
            ma20: ma20[i],
            rsi: rsi[i],
            macd: macd.macd[i],
            macdSignal: macd.signal[i],
            bollingerUpper: bollinger.upper[i],
            bollingerLower: bollinger.lower[i],
            bollingerMid: bollinger.mid[i]
        }));
    }

    /**
     * 移动平均
     */
    movingAverage(data, period) {
        const result = [];
        for (let i = 0; i < data.length; i++) {
            if (i < period - 1) {
                result.push(data[i]);
            } else {
                const sum = data.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
                result.push(sum / period);
            }
        }
        return result;
    }

    /**
     * 计算RSI
     */
    calculateRSI(data, period = 14) {
        const rsi = [];
        const gains = [];
        const losses = [];

        for (let i = 0; i < data.length; i++) {
            if (i === 0) {
                gains.push(0);
                losses.push(0);
                rsi.push(50);
            } else {
                const change = data[i] - data[i - 1];
                gains.push(Math.max(0, change));
                losses.push(Math.max(0, -change));

                if (i >= period) {
                    const avgGain = gains.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / period;
                    const avgLoss = losses.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / period;
                    
                    if (avgLoss === 0) {
                        rsi.push(100);
                    } else {
                        const rs = avgGain / avgLoss;
                        rsi.push(100 - (100 / (1 + rs)));
                    }
                } else {
                    rsi.push(50);
                }
            }
        }
        return rsi;
    }

    /**
     * 计算MACD
     */
    calculateMACD(data, fast = 12, slow = 26, signal = 9) {
        const emaFast = this.ema(data, fast);
        const emaSlow = this.ema(data, slow);
        const macdLine = emaFast.map((v, i) => v - emaSlow[i]);
        const signalLine = this.ema(macdLine, signal);
        
        return {
            macd: macdLine,
            signal: signalLine,
            histogram: macdLine.map((v, i) => v - signalLine[i])
        };
    }

    /**
     * 指数移动平均
     */
    ema(data, period) {
        const k = 2 / (period + 1);
        const result = [data[0]];
        
        for (let i = 1; i < data.length; i++) {
            result.push(data[i] * k + result[i - 1] * (1 - k));
        }
        return result;
    }

    /**
     * 计算布林带
     */
    calculateBollinger(data, period = 20, stdDev = 2) {
        const ma = this.movingAverage(data, period);
        const upper = [];
        const lower = [];

        for (let i = 0; i < data.length; i++) {
            if (i < period - 1) {
                upper.push(ma[i] * 1.02);
                lower.push(ma[i] * 0.98);
            } else {
                const slice = data.slice(i - period + 1, i + 1);
                const mean = slice.reduce((a, b) => a + b, 0) / period;
                const variance = slice.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / period;
                const std = Math.sqrt(variance);
                
                upper.push(ma[i] + stdDev * std);
                lower.push(ma[i] - stdDev * std);
            }
        }

        return { upper, lower, mid: ma };
    }

    /**
     * Min-Max归一化
     */
    normalize(data, key = null) {
        const values = key ? data.map(d => d[key]) : data;
        const min = Math.min(...values);
        const max = Math.max(...values);
        const range = max - min || 1;

        const normalized = values.map(v => (v - min) / range);
        
        // 存储归一化参数用于反归一化
        const normalizeKey = key || 'default';
        this.normalizers.set(normalizeKey, { min, max, range });

        return normalized;
    }

    /**
     * 反归一化
     */
    denormalize(normalizedValues, key = null) {
        const normalizeKey = key || 'default';
        const params = this.normalizers.get(normalizeKey);
        
        if (!params) {
            console.warn(`No normalizer found for key: ${normalizeKey}`);
            return normalizedValues;
        }

        return normalizedValues.map(v => v * params.range + params.min);
    }

    /**
     * 准备训练数据
     * @param {Object} multiSourceData - 多源数据 { symbol: data[] }
     * @param {string} targetSymbol - 预测目标
     * @param {number} lookback - 回看窗口
     * @param {number} forecast - 预测长度
     */
    prepareTrainingData(multiSourceData, targetSymbol, lookback, forecast) {
        console.log('[DataProcessor] prepareTrainingData() 开始准备训练数据...');
        console.log('[DataProcessor] 参数:', { targetSymbol, lookback, forecast });
        console.log('[DataProcessor] 数据源:', Object.keys(multiSourceData));
        
        const sources = Object.keys(multiSourceData);
        const targetData = multiSourceData[targetSymbol];
        
        // 确保所有数据长度一致
        const minLength = Math.min(...Object.values(multiSourceData).map(d => d.length));
        console.log('[DataProcessor] 最小数据长度:', minLength);
        
        // 特征提取和归一化
        const normalizedSources = {};
        sources.forEach(symbol => {
            const data = multiSourceData[symbol].slice(0, minLength);
            const features = this.extractFeatures(data);
            
            // 对每个特征维度进行归一化
            const normalizedFeatures = [];
            for (let i = 0; i < features.length; i++) {
                const normalizedRow = features[i].map((val, dim) => {
                    const key = `${symbol}_dim${dim}`;
                    if (!this.normalizers.has(key)) {
                        // 首次遇到，计算归一化参数
                        const allValues = features.map(f => f[dim]);
                        const min = Math.min(...allValues);
                        const max = Math.max(...allValues);
                        this.normalizers.set(key, { min, max, range: max - min || 1 });
                    }
                    const params = this.normalizers.get(key);
                    return (val - params.min) / params.range;
                });
                normalizedFeatures.push(normalizedRow);
            }
            normalizedSources[symbol] = normalizedFeatures;
        });

        // 创建训练样本
        const X = [];  // 输入序列
        const Y = [];  // 目标序列
        
        const totalSamples = minLength - lookback - forecast;
        
        for (let i = 0; i < totalSamples; i++) {
            // 多源输入 [lookback, numSources, features]
            const inputSequence = [];
            for (let t = 0; t < lookback; t++) {
                const timeStep = [];
                sources.forEach(symbol => {
                    timeStep.push(...normalizedSources[symbol][i + t]);
                });
                inputSequence.push(timeStep);
            }
            X.push(inputSequence);

            // 目标输出 [forecast, 4] (OHLC)
            const targetSequence = [];
            for (let t = 0; t < forecast; t++) {
                const targetFeatures = normalizedSources[targetSymbol][i + lookback + t];
                targetSequence.push(targetFeatures.slice(0, 4)); // 只取OHLC
            }
            Y.push(targetSequence);
        }

        console.log('[DataProcessor] ✅ 训练数据准备完成');
        console.log('[DataProcessor] X 形状:', [X.length, X[0]?.length, X[0]?.[0]?.length]);
        console.log('[DataProcessor] Y 形状:', [Y.length, Y[0]?.length, Y[0]?.[0]?.length]);
        console.log('[DataProcessor] 特征数:', sources.length * 5);
        
        return {
            X: X,
            Y: Y,
            sources: sources,
            featureCount: sources.length * 5,
            normalizers: this.normalizers
        };
    }

    /**
     * 反归一化预测结果
     */
    denormalizePrediction(prediction, targetSymbol) {
        console.log('[DataProcessor] denormalizePrediction() 开始反归一化...');
        console.log('[DataProcessor] 输入预测形状:', [prediction?.length, prediction?.[0]?.length]);
        console.log('[DataProcessor] 目标符号:', targetSymbol);
        console.log('[DataProcessor] 归一化器数量:', this.normalizers.size);
        
        // 检查归一化器是否存在
        for (let dim = 0; dim < 4; dim++) {
            const key = `${targetSymbol}_dim${dim}`;
            const params = this.normalizers.get(key);
            console.log(`[DataProcessor] 归一化器 ${key}:`, params);
        }
        
        const result = [];
        for (let t = 0; t < prediction.length; t++) {
            const row = [];
            for (let dim = 0; dim < 4; dim++) {
                const key = `${targetSymbol}_dim${dim}`;
                const params = this.normalizers.get(key);
                if (params) {
                    row.push(prediction[t][dim] * params.range + params.min);
                } else {
                    console.warn(`[DataProcessor] ⚠️ 找不到归一化器: ${key}`);
                    row.push(prediction[t][dim]);
                }
            }
            result.push({
                open: row[0],
                high: row[1],
                low: row[2],
                close: row[3]
            });
        }
        
        console.log('[DataProcessor] ✅ 反归一化完成');
        console.log('[DataProcessor] 输出结果:', result);
        return result;
    }

    /**
     * 清除缓存
     */
    clearCache() {
        this.cache.clear();
    }
}

export default DataProcessor;
