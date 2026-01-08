/**
 * Real model test - test same code as browser
 * 
 * Run: node tests/real-model.test.js
 */

import * as tf from '@tensorflow/tfjs';
import { DataProcessor } from '../src/utils/dataProcessor.js';
import { SSMAttentionModel } from '../src/models/model.js';

// Wait for TensorFlow initialization
await tf.ready();

console.log('');
console.log('═'.repeat(60));
console.log('      Real Model Performance Test (identical to browser code)');
console.log('═'.repeat(60));
console.log('');

console.log(`TensorFlow.js version: ${tf.version.tfjs}`);
console.log(`Backend: ${tf.getBackend()}`);
console.log('');

// Performance issues list
const issues = [];

/**
 * Test 1: Model construction performance
 */
async function testModelBuild() {
    console.log('🔍 Test 1: Model construction performance\n');
    
    // Use same optimized default config as browser HTML
    const config = {
        inputDim: 15,        // 3 data sources x 5 features
        hiddenDim: 32,       // Optimized: 64 -> 32
        stateDim: 16,        // Optimized: 32 -> 16
        numHeads: 2,         // Optimized: 4 -> 2
        numLayers: 1,        // Optimized: 2 -> 1
        outputDim: 4,        // OHLC
        lookback: 30,        // Optimized: 60 -> 30
        forecast: 5,         // Default
        numSources: 3,
        dropoutRate: 0.1,
        learningRate: 0.001,
        epochs: 20           // Optimized: 50 -> 20
    };
    
    console.log('  Config:', JSON.stringify(config, null, 2).split('\n').map(l => '  ' + l).join('\n'));
    console.log('');
    
    const startBuild = performance.now();
    const model = new SSMAttentionModel(config);
    
    try {
        model.build();
        const buildTime = performance.now() - startBuild;
        
        console.log(`  ⏱️ 模型构建时间: ${buildTime.toFixed(0)} ms`);
        console.log(`  📊 模型参数量: ${model.model.countParams().toLocaleString()}`);
        console.log(`  🔢 层数: ${model.model.layers.length}`);
        console.log(`  🧠 张量数: ${tf.memory().numTensors}`);
        
        if (buildTime > 5000) {
            issues.push({
                severity: 'HIGH',
                type: 'MODEL_BUILD',
                message: `模型构建时间过长: ${buildTime.toFixed(0)}ms`,
                suggestion: '减少 numLayers 或 hiddenDim'
            });
        }
        
        if (model.model.countParams() > 100000) {
            issues.push({
                severity: 'MEDIUM', 
                type: 'MODEL_SIZE',
                message: `模型参数过多: ${model.model.countParams().toLocaleString()}`,
                suggestion: '考虑减少 hiddenDim 或 numLayers'
            });
        }
        
        return model;
    } catch (error) {
        console.log(`  ❌ 构建失败: ${error.message}`);
        issues.push({
            severity: 'HIGH',
            type: 'BUILD_ERROR',
            message: error.message
        });
        return null;
    }
}

/**
 * 测试2: 前向传播性能
 */
async function testForwardPass(model) {
    if (!model) return;
    
    console.log('\n🔍 测试2: 前向传播性能\n');
    
    const { lookback, inputDim } = model.config;
    const batchSizes = [1, 4, 8, 16];
    
    for (const batchSize of batchSizes) {
        const input = tf.randomNormal([batchSize, lookback, inputDim]);
        
        // 预热
        const warmup = model.model.predict(input);
        warmup.dispose();
        
        // 计时
        const times = [];
        for (let i = 0; i < 5; i++) {
            const start = performance.now();
            const pred = model.model.predict(input);
            pred.dispose();
            times.push(performance.now() - start);
        }
        
        input.dispose();
        
        const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
        console.log(`  批次 ${batchSize.toString().padStart(2)}: ${avgTime.toFixed(1).padStart(8)} ms`);
        
        if (batchSize === 1 && avgTime > 100) {
            issues.push({
                severity: 'HIGH',
                type: 'INFERENCE_SLOW',
                message: `单次推理太慢: ${avgTime.toFixed(0)}ms`,
                suggestion: '模型太复杂，需要简化'
            });
        }
    }
}

/**
 * 测试3: 训练循环性能（关键测试）
 */
async function testTrainingLoop(model) {
    if (!model) return;
    
    console.log('\n🔍 测试3: 训练循环性能 (这是主要卡顿来源)\n');
    
    const { lookback, inputDim, outputDim, forecast } = model.config;
    
    // 生成训练数据 - 模拟实际数据量
    const numSamples = 100;
    const X = [];
    const Y = [];
    
    for (let i = 0; i < numSamples; i++) {
        X.push(Array(lookback).fill(0).map(() => 
            Array(inputDim).fill(0).map(() => Math.random())
        ));
        Y.push(Array(forecast).fill(0).map(() => 
            Array(outputDim).fill(0).map(() => Math.random())
        ));
    }
    
    console.log(`  训练数据: ${numSamples} 样本, lookback=${lookback}, forecast=${forecast}`);
    console.log(`  输入维度: [${numSamples}, ${lookback}, ${inputDim}]`);
    console.log(`  输出维度: [${numSamples}, ${forecast}, ${outputDim}]`);
    console.log('');
    
    // 测试单个 epoch 时间
    const xTensor = tf.tensor3d(X);
    const yTensor = tf.tensor3d(Y);
    
    const splitIdx = Math.floor(numSamples * 0.8);
    const xTrain = xTensor.slice([0, 0, 0], [splitIdx, -1, -1]);
    const yTrain = yTensor.slice([0, 0, 0], [splitIdx, -1, -1]);
    const xVal = xTensor.slice([splitIdx, 0, 0], [-1, -1, -1]);
    const yVal = yTensor.slice([splitIdx, 0, 0], [-1, -1, -1]);
    
    console.log(`  🔄 测试 3 个 epoch...\n`);
    
    const epochTimes = [];
    const startTotal = performance.now();
    
    for (let epoch = 0; epoch < 3; epoch++) {
        const startEpoch = performance.now();
        
        await model.model.fit(xTrain, yTrain, {
            epochs: 1,
            batchSize: 16,
            validationData: [xVal, yVal],
            verbose: 0,
            callbacks: {
                onBatchEnd: async () => {
                    // 这是关键！允许浏览器响应
                    await tf.nextFrame();
                }
            }
        });
        
        const epochTime = performance.now() - startEpoch;
        epochTimes.push(epochTime);
        console.log(`  Epoch ${epoch + 1}: ${epochTime.toFixed(0)} ms`);
    }
    
    const totalTime = performance.now() - startTotal;
    const avgEpochTime = epochTimes.reduce((a, b) => a + b, 0) / epochTimes.length;
    
    console.log('');
    console.log(`  📊 平均每轮: ${avgEpochTime.toFixed(0)} ms`);
    console.log(`  📊 总时间: ${totalTime.toFixed(0)} ms`);
    console.log(`  📊 预估 50 轮: ${(avgEpochTime * 50 / 1000).toFixed(1)} 秒`);
    
    if (avgEpochTime > 2000) {
        issues.push({
            severity: 'HIGH',
            type: 'TRAINING_SLOW',
            message: `每轮训练太慢: ${avgEpochTime.toFixed(0)}ms`,
            suggestion: '1. 减少模型复杂度 2. 增加批次大小 3. 减少训练数据'
        });
    }
    
    // 清理
    xTensor.dispose();
    yTensor.dispose();
    xTrain.dispose();
    yTrain.dispose();
    xVal.dispose();
    yVal.dispose();
}

/**
 * 测试4: 内存泄漏检测
 */
async function testMemoryLeak(model) {
    if (!model) return;
    
    console.log('\n🔍 测试4: 内存泄漏检测\n');
    
    const { lookback, inputDim } = model.config;
    const initialTensors = tf.memory().numTensors;
    
    console.log(`  初始张量数: ${initialTensors}`);
    
    // 模拟多次推理
    for (let i = 0; i < 20; i++) {
        const input = tf.randomNormal([1, lookback, inputDim]);
        const pred = model.model.predict(input);
        
        // 故意不释放，检测泄漏
        if (i % 2 === 0) {
            input.dispose();
            pred.dispose();
        }
    }
    
    const afterTensors = tf.memory().numTensors;
    const leaked = afterTensors - initialTensors;
    
    console.log(`  最终张量数: ${afterTensors}`);
    console.log(`  泄漏张量数: ${leaked}`);
    
    if (leaked > 20) {
        issues.push({
            severity: 'HIGH',
            type: 'MEMORY_LEAK',
            message: `严重内存泄漏: ${leaked} 个张量`,
            suggestion: '确保所有 predict 调用都在 tf.tidy() 中'
        });
    }
}

/**
 * 测试5: 实际数据处理流程
 */
async function testDataProcessing() {
    console.log('\n🔍 测试5: 数据处理流程\n');
    
    const processor = new DataProcessor();
    
    // 模拟实际使用场景
    const symbols = ['SPY', 'VIX', 'TLT'];
    const days = 110; // lookback + 额外数据
    
    const start = performance.now();
    
    const multiSourceData = {};
    for (const symbol of symbols) {
        const data = await processor.fetchData(symbol, days);
        multiSourceData[symbol] = data;
    }
    
    const fetchTime = performance.now() - start;
    console.log(`  数据获取: ${fetchTime.toFixed(0)} ms`);
    
    // 准备训练数据
    const startPrep = performance.now();
    const trainingData = processor.prepareTrainingData(
        multiSourceData,
        'SPY',
        60,  // lookback
        5    // forecast
    );
    const prepTime = performance.now() - startPrep;
    
    console.log(`  数据准备: ${prepTime.toFixed(0)} ms`);
    console.log(`  样本数量: ${trainingData.X.length}`);
    console.log(`  特征数量: ${trainingData.featureCount}`);
    
    if (prepTime > 1000) {
        issues.push({
            severity: 'MEDIUM',
            type: 'DATA_PREP_SLOW',
            message: `数据准备太慢: ${prepTime.toFixed(0)}ms`,
            suggestion: '优化数据处理算法'
        });
    }
    
    return trainingData;
}

/**
 * 打印诊断报告
 */
function printReport() {
    console.log('');
    console.log('═'.repeat(60));
    console.log('                    诊断报告');
    console.log('═'.repeat(60));
    console.log('');
    
    console.log(`最终张量数: ${tf.memory().numTensors}`);
    console.log(`内存使用: ${(tf.memory().numBytes / 1024 / 1024).toFixed(1)} MB`);
    console.log('');
    
    if (issues.length === 0) {
        console.log('✅ 未发现严重性能问题');
    } else {
        const severityOrder = { 'HIGH': 0, 'MEDIUM': 1, 'LOW': 2 };
        issues.sort((a, b) => severityOrder[a.severity] - severityOrder[b.severity]);
        
        console.log(`发现 ${issues.length} 个问题:\n`);
        
        for (const issue of issues) {
            const icon = { 'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢' }[issue.severity];
            console.log(`${icon} [${issue.severity}] ${issue.type}`);
            console.log(`   问题: ${issue.message}`);
            if (issue.suggestion) {
                console.log(`   建议: ${issue.suggestion}`);
            }
            console.log('');
        }
    }
    
    console.log('═'.repeat(60));
    console.log('                  优化建议');
    console.log('═'.repeat(60));
    console.log('');
    console.log('针对当前模型的具体优化:');
    console.log('');
    console.log('1. 【立即】减少模型复杂度:');
    console.log('   - hiddenDim: 64 -> 32');
    console.log('   - stateDim: 32 -> 16');
    console.log('   - numLayers: 2 -> 1');
    console.log('   - numHeads: 4 -> 2');
    console.log('');
    console.log('2. 【立即】减少训练参数:');
    console.log('   - epochs: 50 -> 20');
    console.log('   - lookback: 60 -> 30');
    console.log('');
    console.log('3. 【关键】在训练循环中添加 yield:');
    console.log('   callbacks: { onBatchEnd: async () => await tf.nextFrame() }');
    console.log('');
    console.log('4. 【重要】确保使用 WebGL 后端:');
    console.log('   在浏览器中检查 tf.getBackend() 应该是 "webgl"');
    console.log('');
}

// 运行所有测试
async function runTests() {
    try {
        const model = await testModelBuild();
        await testForwardPass(model);
        await testTrainingLoop(model);
        await testMemoryLeak(model);
        await testDataProcessing();
        
        if (model) {
            model.dispose();
        }
        
        printReport();
    } catch (error) {
        console.error('测试出错:', error);
    }
}

runTests();
