/**
 * CSV 数据读取测试
 * 
 * 运行方式: node tests/csv-data.test.js
 * 
 * 测试前准备:
 * 1. 使用 Python yfinance 下载数据：
 *    import yfinance as yf
 *    spy = yf.download('SPY', start='2023-01-01', end='2024-01-01')
 *    spy.to_csv('tests/data/SPY.csv')
 * 
 * 2. 或者使用本测试生成的模拟 CSV 文件
 */

import * as tf from '@tensorflow/tfjs';
import { DataProcessor } from '../src/utils/dataProcessor.js';
import { SSMAttentionModel } from '../src/models/model.js';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 等待 TensorFlow 初始化
await tf.ready();

console.log('');
console.log('═'.repeat(60));
console.log('           CSV 数据读取与预测测试');
console.log('═'.repeat(60));
console.log('');
console.log(`TensorFlow.js 版本: ${tf.version.tfjs}`);
console.log(`后端: ${tf.getBackend()}`);
console.log('');

const dataProcessor = new DataProcessor();

// 测试结果
const testResults = [];

/**
 * 生成 yfinance 格式的模拟 CSV 数据
 */
function generateYFinanceCSV(symbol, days = 100) {
    const lines = ['Date,Open,High,Low,Close,Adj Close,Volume'];
    
    let price = symbol === 'SPY' ? 450 : symbol === 'VIX' ? 18 : 100;
    const volatility = symbol === 'VIX' ? 0.08 : 0.015;
    
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);
    
    for (let i = 0; i < days; i++) {
        const date = new Date(startDate);
        date.setDate(date.getDate() + i);
        
        // 跳过周末
        if (date.getDay() === 0 || date.getDay() === 6) continue;
        
        const change = (Math.random() - 0.5) * 2 * volatility * price;
        const open = price;
        const close = price + change;
        const high = Math.max(open, close) * (1 + Math.random() * 0.01);
        const low = Math.min(open, close) * (1 - Math.random() * 0.01);
        const volume = Math.floor(50000000 + Math.random() * 100000000);
        
        price = close;
        
        const dateStr = date.toISOString().split('T')[0];
        lines.push(`${dateStr},${open.toFixed(2)},${high.toFixed(2)},${low.toFixed(2)},${close.toFixed(2)},${close.toFixed(2)},${volume}`);
    }
    
    return lines.join('\n');
}

/**
 * 测试 1: CSV 解析功能
 */
async function testCSVParsing() {
    console.log('🔍 测试 1: CSV 解析功能\n');
    
    // 生成测试 CSV
    const csvContent = generateYFinanceCSV('SPY', 100);
    console.log('  生成的 CSV 示例（前5行）:');
    csvContent.split('\n').slice(0, 6).forEach(line => console.log('    ' + line));
    console.log('');
    
    try {
        const startTime = performance.now();
        const data = dataProcessor.parseYFinanceCSV(csvContent, 'SPY');
        const parseTime = performance.now() - startTime;
        
        console.log(`  ✅ 解析成功`);
        console.log(`  ⏱️  解析时间: ${parseTime.toFixed(2)} ms`);
        console.log(`  📊 解析行数: ${data.length}`);
        console.log(`  📅 日期范围: ${data[0].date} 至 ${data[data.length - 1].date}`);
        console.log(`  💰 价格范围: ${Math.min(...data.map(d => d.low)).toFixed(2)} - ${Math.max(...data.map(d => d.high)).toFixed(2)}`);
        
        // 验证数据结构
        const sample = data[0];
        const requiredFields = ['date', 'timestamp', 'open', 'high', 'low', 'close', 'volume'];
        const hasAllFields = requiredFields.every(f => sample.hasOwnProperty(f));
        
        console.log(`  🔍 数据结构验证: ${hasAllFields ? '✅ 通过' : '❌ 失败'}`);
        console.log('  📋 示例数据:', JSON.stringify(sample, null, 2).split('\n').map(l => '     ' + l).join('\n'));
        
        testResults.push({ name: 'CSV解析', status: 'PASS', time: parseTime });
        return data;
    } catch (error) {
        console.log(`  ❌ 解析失败: ${error.message}`);
        testResults.push({ name: 'CSV解析', status: 'FAIL', error: error.message });
        return null;
    }
}

/**
 * 测试 2: 多种 CSV 格式兼容性
 */
async function testCSVFormats() {
    console.log('\n🔍 测试 2: CSV 格式兼容性\n');
    
    const formats = [
        {
            name: 'yfinance 标准格式',
            csv: 'Date,Open,High,Low,Close,Adj Close,Volume\n2024-01-02,100.0,101.5,99.5,101.0,101.0,1000000'
        },
        {
            name: '无 Adj Close',
            csv: 'Date,Open,High,Low,Close,Volume\n2024-01-02,100.0,101.5,99.5,101.0,1000000'
        },
        {
            name: '大小写混合',
            csv: 'DATE,open,HIGH,Low,CLOSE,volume\n2024-01-02,100.0,101.5,99.5,101.0,1000000'
        },
        {
            name: '无日期列（使用索引）',
            csv: 'Open,High,Low,Close,Volume\n100.0,101.5,99.5,101.0,1000000\n101.0,102.0,100.0,101.5,1100000'
        },
        {
            name: 'MM/DD/YYYY 日期格式',
            csv: 'Date,Open,High,Low,Close,Volume\n01/02/2024,100.0,101.5,99.5,101.0,1000000'
        }
    ];
    
    let passed = 0;
    for (const format of formats) {
        try {
            const data = dataProcessor.parseYFinanceCSV(format.csv, 'TEST');
            console.log(`  ✅ ${format.name}: 解析成功 (${data.length} 行)`);
            passed++;
        } catch (error) {
            console.log(`  ❌ ${format.name}: ${error.message}`);
        }
    }
    
    console.log(`\n  总计: ${passed}/${formats.length} 格式支持`);
    testResults.push({ name: 'CSV格式兼容', status: passed === formats.length ? 'PASS' : 'PARTIAL' });
}

/**
 * 测试 3: 使用 CSV 数据训练模型
 */
async function testTrainingWithCSV() {
    console.log('\n🔍 测试 3: 使用 CSV 数据训练模型\n');
    
    try {
        // 生成多个数据源的 CSV
        const spyCsv = generateYFinanceCSV('SPY', 120);
        const vixCsv = generateYFinanceCSV('VIX', 120);
        
        // 解析 CSV
        const spyData = dataProcessor.parseYFinanceCSV(spyCsv, 'SPY');
        const vixData = dataProcessor.parseYFinanceCSV(vixCsv, 'VIX');
        
        // 存储到 csvData
        dataProcessor.csvData.set('SPY', spyData);
        dataProcessor.csvData.set('VIX', vixData);
        
        console.log(`  📊 SPY 数据: ${spyData.length} 条`);
        console.log(`  📊 VIX 数据: ${vixData.length} 条`);
        
        // 构建多源数据
        const multiSourceData = {
            'SPY': spyData,
            'VIX': vixData
        };
        
        // 准备训练数据
        const lookback = 30;
        const forecast = 5;
        
        console.log(`  ⚙️  配置: lookback=${lookback}, forecast=${forecast}`);
        
        const startPrepare = performance.now();
        const trainingData = dataProcessor.prepareTrainingData(
            multiSourceData,
            'SPY',
            lookback,
            forecast
        );
        const prepareTime = performance.now() - startPrepare;
        
        console.log(`  ⏱️  数据准备时间: ${prepareTime.toFixed(2)} ms`);
        console.log(`  📊 训练样本数: ${trainingData.X.length}`);
        console.log(`  📐 输入形状: [${trainingData.X.length}, ${trainingData.X[0].length}, ${trainingData.X[0][0].length}]`);
        console.log(`  📐 输出形状: [${trainingData.Y.length}, ${trainingData.Y[0].length}, ${trainingData.Y[0][0].length}]`);
        
        // 创建模型
        const config = {
            inputDim: trainingData.featureCount,
            hiddenDim: 32,
            stateDim: 16,
            numHeads: 2,
            numLayers: 1,
            outputDim: 4,
            lookback: lookback,
            forecast: forecast,
            numSources: 2,
            dropoutRate: 0.1,
            learningRate: 0.001,
            epochs: 5  // 测试用较少轮数
        };
        
        console.log('\n  🧠 创建模型...');
        const model = new SSMAttentionModel(config);
        model.build();
        console.log(`  📊 模型参数量: ${model.model.countParams().toLocaleString()}`);
        
        // 训练模型
        console.log('\n  🏋️ 开始训练 (5 轮)...');
        const startTrain = performance.now();
        
        await model.train(trainingData, {
            onEpochEnd: (epoch, logs) => {
                console.log(`    Epoch ${epoch + 1}: loss=${logs.loss.toFixed(6)}, val_loss=${logs.val_loss.toFixed(6)}`);
            }
        });
        
        const trainTime = performance.now() - startTrain;
        console.log(`  ⏱️  训练时间: ${(trainTime / 1000).toFixed(2)} 秒`);
        
        // 进行预测
        console.log('\n  🔮 进行预测...');
        const lastSequence = trainingData.X[trainingData.X.length - 1];
        const prediction = model.predict(lastSequence);
        
        console.log('  📈 预测结果 (归一化):');
        prediction.forEach((p, i) => {
            console.log(`    Day ${i + 1}: O=${p[0].toFixed(4)}, H=${p[1].toFixed(4)}, L=${p[2].toFixed(4)}, C=${p[3].toFixed(4)}`);
        });
        
        // 反归一化
        const denormalized = dataProcessor.denormalizePrediction(prediction, 'SPY');
        console.log('\n  💰 预测结果 (实际价格):');
        denormalized.forEach((p, i) => {
            console.log(`    Day ${i + 1}: O=${p.open.toFixed(2)}, H=${p.high.toFixed(2)}, L=${p.low.toFixed(2)}, C=${p.close.toFixed(2)}`);
        });
        
        // 清理
        model.dispose();
        
        testResults.push({ name: 'CSV训练预测', status: 'PASS', time: trainTime });
        console.log('\n  ✅ CSV 数据训练测试通过');
        
    } catch (error) {
        console.log(`  ❌ 测试失败: ${error.message}`);
        console.error(error);
        testResults.push({ name: 'CSV训练预测', status: 'FAIL', error: error.message });
    }
}

/**
 * 测试 4: 从本地文件读取 CSV（如果存在）
 */
async function testLocalCSVFile() {
    console.log('\n🔍 测试 4: 本地 CSV 文件读取\n');
    
    const testDataDir = path.join(__dirname, 'data');
    const testFile = path.join(testDataDir, 'SPY.csv');
    
    // 检查测试数据目录
    if (!fs.existsSync(testDataDir)) {
        fs.mkdirSync(testDataDir, { recursive: true });
        console.log(`  📁 创建测试数据目录: ${testDataDir}`);
    }
    
    // 检查是否存在真实的 CSV 文件
    if (fs.existsSync(testFile)) {
        console.log(`  📄 找到本地文件: ${testFile}`);
        
        try {
            const content = fs.readFileSync(testFile, 'utf-8');
            const data = dataProcessor.parseYFinanceCSV(content, 'SPY');
            
            console.log(`  ✅ 解析成功: ${data.length} 条记录`);
            console.log(`  📅 日期范围: ${data[0].date} 至 ${data[data.length - 1].date}`);
            console.log(`  💰 最新收盘价: ${data[data.length - 1].close.toFixed(2)}`);
            
            testResults.push({ name: '本地CSV读取', status: 'PASS' });
        } catch (error) {
            console.log(`  ❌ 读取失败: ${error.message}`);
            testResults.push({ name: '本地CSV读取', status: 'FAIL', error: error.message });
        }
    } else {
        console.log(`  ⚠️  未找到本地测试文件: ${testFile}`);
        console.log('  💡 提示: 您可以使用以下 Python 代码生成测试数据:');
        console.log('');
        console.log('     import yfinance as yf');
        console.log('     spy = yf.download("SPY", start="2023-01-01", end="2024-01-01")');
        console.log(`     spy.to_csv("${testFile}")`);
        console.log('');
        
        // 生成模拟的 CSV 文件用于测试
        console.log('  📝 生成模拟 CSV 文件用于测试...');
        const mockCsv = generateYFinanceCSV('SPY', 252);  // 约1年交易日
        fs.writeFileSync(testFile, mockCsv);
        console.log(`  ✅ 已生成: ${testFile}`);
        
        testResults.push({ name: '本地CSV读取', status: 'SKIP', note: '使用生成的模拟数据' });
    }
}

/**
 * 测试 5: 错误处理
 */
async function testErrorHandling() {
    console.log('\n🔍 测试 5: 错误处理\n');
    
    const errorCases = [
        {
            name: '空文件',
            csv: '',
            expectError: true
        },
        {
            name: '只有表头',
            csv: 'Date,Open,High,Low,Close,Volume',
            expectError: true
        },
        {
            name: '缺少必需列',
            csv: 'Date,Price,Volume\n2024-01-02,100.0,1000000',
            expectError: true
        },
        {
            name: '无效数值',
            csv: 'Date,Open,High,Low,Close,Volume\n2024-01-02,abc,101.5,99.5,101.0,1000000',
            expectError: false  // 应该跳过无效行
        }
    ];
    
    let handled = 0;
    for (const testCase of errorCases) {
        try {
            const data = dataProcessor.parseYFinanceCSV(testCase.csv, 'TEST');
            if (testCase.expectError) {
                console.log(`  ⚠️  ${testCase.name}: 应该抛出错误但没有`);
            } else {
                console.log(`  ✅ ${testCase.name}: 正确处理 (${data.length} 行)`);
                handled++;
            }
        } catch (error) {
            if (testCase.expectError) {
                console.log(`  ✅ ${testCase.name}: 正确抛出错误 - ${error.message}`);
                handled++;
            } else {
                console.log(`  ❌ ${testCase.name}: 意外错误 - ${error.message}`);
            }
        }
    }
    
    console.log(`\n  错误处理: ${handled}/${errorCases.length} 测试通过`);
    testResults.push({ name: '错误处理', status: handled === errorCases.length ? 'PASS' : 'PARTIAL' });
}

/**
 * 运行所有测试
 */
async function runAllTests() {
    await testCSVParsing();
    await testCSVFormats();
    await testTrainingWithCSV();
    await testLocalCSVFile();
    await testErrorHandling();
    
    // 打印测试摘要
    console.log('\n');
    console.log('═'.repeat(60));
    console.log('                    测试摘要');
    console.log('═'.repeat(60));
    console.log('');
    
    testResults.forEach(result => {
        const icon = result.status === 'PASS' ? '✅' : 
                     result.status === 'FAIL' ? '❌' : 
                     result.status === 'PARTIAL' ? '⚠️' : '⏭️';
        console.log(`  ${icon} ${result.name}: ${result.status}`);
        if (result.error) console.log(`     错误: ${result.error}`);
        if (result.time) console.log(`     耗时: ${result.time.toFixed(2)} ms`);
    });
    
    const passed = testResults.filter(r => r.status === 'PASS').length;
    const total = testResults.length;
    
    console.log('');
    console.log(`  总计: ${passed}/${total} 测试通过`);
    console.log('');
    console.log('═'.repeat(60));
}

// 运行测试
runAllTests().catch(console.error);
