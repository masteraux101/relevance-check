/**
 * Performance analysis and bottleneck diagnosis
 * 
 * Run: node tests/performance.test.js
 */

import * as tf from '@tensorflow/tfjs';  // Use pure JS version
import { DataProcessor } from '../src/utils/dataProcessor.js';

// Wait for TensorFlow initialization
await tf.ready();

console.log('');
console.log('═'.repeat(60));
console.log('           Performance Analysis and Bottleneck Diagnosis');
console.log('═'.repeat(60));
console.log('');

// Collect performance issues
const issues = [];

/**
 * Test 1: Detect tensor leaks
 */
async function detectTensorLeaks() {
    console.log('🔍 Detecting tensor memory leaks...\n');
    
    const iterations = 50;
    const tensorCounts = [];
    
    for (let i = 0; i < iterations; i++) {
        // Simulate common operations
        const input = tf.randomNormal([16, 60, 15]);
        const dense = tf.layers.dense({ units: 64 });
        const output = dense.apply(input);
        
        // Intentionally don't dispose, observe leaks
        if (i % 2 === 0) {
            input.dispose();
            output.dispose();
        }
        
        tensorCounts.push(tf.memory().numTensors);
        
        if (i % 10 === 0) {
            console.log(`  Iteration ${i}: tensor count ${tf.memory().numTensors}`);
        }
    }
    
    // Analyze leak trend
    const firstHalf = tensorCounts.slice(0, 25).reduce((a, b) => a + b, 0) / 25;
    const secondHalf = tensorCounts.slice(25).reduce((a, b) => a + b, 0) / 25;
    
    if (secondHalf > firstHalf * 1.5) {
        issues.push({
            type: 'TENSOR_LEAK',
            severity: 'HIGH',
            message: `Tensor leak detected: tensor count grew from average ${firstHalf.toFixed(0)} to ${secondHalf.toFixed(0)}`,
            suggestion: 'Ensure all intermediate tensors are disposed() or wrapped with tf.tidy()'
        });
    }
    
    // Cleanup
    tf.disposeVariables();
}

/**
 * Test 2: Detect synchronous blocking
 */
async function detectSyncBlocking() {
    console.log('\n🔍 Detecting synchronous blocking operations...\n');
    
    const operations = [
        {
            name: 'arraySync()',
            test: () => {
                const tensor = tf.randomNormal([1000, 1000]);
                const start = performance.now();
                const arr = tensor.arraySync();  // Synchronous operation, will block
                const time = performance.now() - start;
                tensor.dispose();
                return time;
            }
        },
        {
            name: 'dataSync()',
            test: () => {
                const tensor = tf.randomNormal([1000, 1000]);
                const start = performance.now();
                const data = tensor.dataSync();  // Synchronous operation, will block
                const time = performance.now() - start;
                tensor.dispose();
                return time;
            }
        },
        {
            name: 'array() (async)',
            test: async () => {
                const tensor = tf.randomNormal([1000, 1000]);
                const start = performance.now();
                const arr = await tensor.array();  // Asynchronous operation, won't block
                const time = performance.now() - start;
                tensor.dispose();
                return time;
            }
        }
    ];
    
    for (const op of operations) {
        const time = await op.test();
        console.log(`  ${op.name}: ${time.toFixed(2)} ms`);
        
        if (time > 100 && !op.name.includes('async')) {
            issues.push({
                type: 'SYNC_BLOCKING',
                severity: 'MEDIUM',
                message: `Sync operation ${op.name} took ${time.toFixed(0)}ms`,
                suggestion: `Use async version like tensor.array() instead of tensor.arraySync()`
            });
        }
    }
}

/**
 * Test 3: Detect excessive recompilation
 */
async function detectRecompilation() {
    console.log('\n🔍 Detecting model recompilation issues...\n');
    
    // Dynamic input shapes cause recompilation
    const shapes = [
        [1, 30, 15],
        [2, 30, 15],
        [4, 30, 15],
        [8, 30, 15],
        [16, 30, 15],
    ];
    
    const input = tf.input({ shape: [30, 15] });  // Fixed shape
    const output = tf.layers.dense({ units: 32 }).apply(input);
    const model = tf.model({ inputs: input, outputs: output });
    
    const times = [];
    for (const shape of shapes) {
        const testInput = tf.randomNormal(shape);
        const start = performance.now();
        const pred = model.predict(testInput);
        times.push(performance.now() - start);
        testInput.dispose();
        pred.dispose();
    }
    
    console.log('  Inference time for different batch sizes:');
    shapes.forEach((shape, i) => {
        console.log(`    [${shape.join(', ')}]: ${times[i].toFixed(2)} ms`);
    });
    
    // First call is usually slow (compilation), subsequent should be fast
    const firstTime = times[0];
    const avgSubsequent = times.slice(1).reduce((a, b) => a + b, 0) / (times.length - 1);
    
    if (times.some((t, i) => i > 0 && t > firstTime * 0.8)) {
        issues.push({
            type: 'RECOMPILATION',
            severity: 'HIGH',
            message: 'Possible repeated compilation detected, dynamic input shapes cause performance degradation',
            suggestion: 'Use fixed input shapes, avoid frequently changing batch size or sequence length'
        });
    }
    
    model.dispose();
}

/**
 * Test 4: Detect inefficient layer structures
 */
async function detectInefficientLayers() {
    console.log('\n🔍 Detecting inefficient layer structures...\n');
    
    const batchSize = 16;
    const seqLen = 60;
    const features = 64;
    
    const layerTests = [
        {
            name: 'Dense (efficient)',
            create: () => {
                const input = tf.input({ shape: [seqLen, features] });
                const output = tf.layers.dense({ units: 64 }).apply(input);
                return tf.model({ inputs: input, outputs: output });
            }
        },
        {
            name: 'LSTM (medium)',
            create: () => {
                const input = tf.input({ shape: [seqLen, features] });
                const output = tf.layers.lstm({ units: 64, returnSequences: true }).apply(input);
                return tf.model({ inputs: input, outputs: output });
            }
        },
        {
            name: 'GRU (medium)',
            create: () => {
                const input = tf.input({ shape: [seqLen, features] });
                const output = tf.layers.gru({ units: 64, returnSequences: true }).apply(input);
                return tf.model({ inputs: input, outputs: output });
            }
        },
        {
            name: 'Multi-layer stacking',
            create: () => {
                const input = tf.input({ shape: [seqLen, features] });
                let x = tf.layers.dense({ units: 64 }).apply(input);
                x = tf.layers.lstm({ units: 32, returnSequences: true }).apply(x);
                x = tf.layers.gru({ units: 32, returnSequences: true }).apply(x);
                x = tf.layers.dense({ units: 64 }).apply(x);
                return tf.model({ inputs: input, outputs: x });
            }
        }
    ];
    
    for (const test of layerTests) {
        const model = test.create();
        const testInput = tf.randomNormal([batchSize, seqLen, features]);
        
        // Warmup
        model.predict(testInput).dispose();
        
        // Timing
        const times = [];
        for (let i = 0; i < 10; i++) {
            const start = performance.now();
            const pred = model.predict(testInput);
            pred.dispose();
            times.push(performance.now() - start);
        }
        
        const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
        console.log(`  ${test.name}: ${avgTime.toFixed(2)} ms`);
        
        testInput.dispose();
        model.dispose();
    }
}

/**
 * Test 5: Detect data preprocessing bottleneck
 */
async function detectDataProcessingBottleneck() {
    console.log('\n🔍 Detecting data preprocessing bottleneck...\n');
    
    const processor = new DataProcessor();
    
    // Test processing time for different data volumes
    const dataSizes = [100, 500, 1000, 2000, 5000];
    
    for (const size of dataSizes) {
        const start = performance.now();
        const data = processor.generateMockData('SPY', size);
        const genTime = performance.now() - start;
        
        const start2 = performance.now();
        const features = processor.extractFeatures(data);
        const extractTime = performance.now() - start2;
        
        const start3 = performance.now();
        const indicators = processor.calculateIndicators(data);
        const indicatorTime = performance.now() - start3;
        
        console.log(`  ${size} records:`);
        console.log(`    Generation: ${genTime.toFixed(2)} ms`);
        console.log(`    Feature extraction: ${extractTime.toFixed(2)} ms`);
        console.log(`    Indicator calculation: ${indicatorTime.toFixed(2)} ms`);
        
        // Detect super-linear growth
        if (size === 5000 && indicatorTime > 500) {
            issues.push({
                type: 'DATA_PROCESSING',
                severity: 'MEDIUM',
                message: `Data processing takes too long: ${size} records need ${indicatorTime.toFixed(0)}ms`,
                suggestion: 'Consider using vectorized operations or WebWorker for data processing'
            });
        }
    }
}

/**
 * Test 6: Detect WebGL backend issues
 */
async function detectBackendIssues() {
    console.log('\n🔍 Detecting backend configuration...\n');
    
    console.log(`  Current backend: ${tf.getBackend()}`);
    console.log(`  Available backends: ${tf.engine().registryFactory ? Object.keys(tf.engine().registryFactory) : ['cpu', 'webgl']}`);
    
    const memInfo = tf.memory();
    console.log(`  Tensor count: ${memInfo.numTensors}`);
    console.log(`  Data buffers: ${memInfo.numDataBuffers}`);
    console.log(`  Allocated bytes: ${(memInfo.numBytes / 1024 / 1024).toFixed(2)} MB`);
    
    if (tf.getBackend() === 'cpu') {
        issues.push({
            type: 'BACKEND',
            severity: 'HIGH',
            message: 'Using CPU backend, performance will be slow',
            suggestion: 'Use WebGL backend in browsers, use tensorflow-node in Node.js'
        });
    }
}

/**
 * Test 7: Detect batch size issues
 */
async function detectBatchSizeIssues() {
    console.log('\n🔍 Detecting batch size performance...\n');
    
    const input = tf.input({ shape: [60, 15] });
    let x = tf.layers.dense({ units: 32 }).apply(input);
    x = tf.layers.lstm({ units: 32, returnSequences: false }).apply(x);
    const output = tf.layers.dense({ units: 20 }).apply(x);
    const model = tf.model({ inputs: input, outputs: output });
    
    const batchSizes = [1, 2, 4, 8, 16, 32, 64];
    const results = [];
    
    for (const bs of batchSizes) {
        const testInput = tf.randomNormal([bs, 60, 15]);
        
        // 预热
        model.predict(testInput).dispose();
        
        const times = [];
        for (let i = 0; i < 5; i++) {
            const start = performance.now();
            const pred = model.predict(testInput);
            pred.dispose();
            times.push(performance.now() - start);
        }
        
        const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
        const perSample = avgTime / bs;
        
        results.push({ bs, avgTime, perSample });
        console.log(`  批次 ${bs.toString().padStart(2)}: 总 ${avgTime.toFixed(2).padStart(8)} ms, 每样本 ${perSample.toFixed(2).padStart(6)} ms`);
        
        testInput.dispose();
    }
    
    // 找到最优批次大小
    const optimal = results.reduce((min, r) => r.perSample < min.perSample ? r : min);
    console.log(`\n  📊 最优批次大小: ${optimal.bs} (每样本 ${optimal.perSample.toFixed(2)} ms)`);
    
    if (optimal.bs > 1) {
        issues.push({
            type: 'BATCH_SIZE',
            severity: 'LOW',
            message: `建议使用批次大小 ${optimal.bs} 以获得最佳性能`,
            suggestion: '批量处理数据而不是逐个处理'
        });
    }
    
    model.dispose();
}

/**
 * 测试8: 检测训练循环效率
 */
async function detectTrainingEfficiency() {
    console.log('\n🔍 检测训练循环效率...\n');
    
    const input = tf.input({ shape: [30, 10] });
    let x = tf.layers.dense({ units: 16 }).apply(input);
    x = tf.layers.lstm({ units: 16, returnSequences: false }).apply(x);
    const output = tf.layers.dense({ units: 5 }).apply(x);
    const model = tf.model({ inputs: input, outputs: output });
    
    model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
    
    const trainX = tf.randomNormal([100, 30, 10]);
    const trainY = tf.randomNormal([100, 5]);
    
    // 测试不同训练配置
    const configs = [
        { batchSize: 8, epochs: 5, desc: '小批次' },
        { batchSize: 32, epochs: 5, desc: '中批次' },
        { batchSize: 64, epochs: 5, desc: '大批次' }
    ];
    
    for (const config of configs) {
        const start = performance.now();
        await model.fit(trainX, trainY, {
            epochs: config.epochs,
            batchSize: config.batchSize,
            verbose: 0
        });
        const time = performance.now() - start;
        
        console.log(`  ${config.desc} (${config.batchSize}): ${time.toFixed(2)} ms / ${config.epochs} epochs`);
    }
    
    trainX.dispose();
    trainY.dispose();
    model.dispose();
}

/**
 * 输出诊断报告
 */
function printReport() {
    console.log('');
    console.log('═'.repeat(60));
    console.log('                    诊断报告');
    console.log('═'.repeat(60));
    console.log('');
    
    if (issues.length === 0) {
        console.log('✅ 未发现明显性能问题\n');
        return;
    }
    
    // 按严重程度排序
    const severityOrder = { 'HIGH': 0, 'MEDIUM': 1, 'LOW': 2 };
    issues.sort((a, b) => severityOrder[a.severity] - severityOrder[b.severity]);
    
    console.log(`发现 ${issues.length} 个潜在问题:\n`);
    
    for (const issue of issues) {
        const icon = {
            'HIGH': '🔴',
            'MEDIUM': '🟡',
            'LOW': '🟢'
        }[issue.severity];
        
        console.log(`${icon} [${issue.severity}] ${issue.type}`);
        console.log(`   问题: ${issue.message}`);
        console.log(`   建议: ${issue.suggestion}`);
        console.log('');
    }
    
    // 性能优化建议
    console.log('═'.repeat(60));
    console.log('                  性能优化建议');
    console.log('═'.repeat(60));
    console.log('');
    console.log('1. 使用 tf.tidy() 包装所有张量操作，避免内存泄漏');
    console.log('2. 优先使用异步操作 (array/data) 而非同步操作 (arraySync/dataSync)');
    console.log('3. 固定模型输入形状，避免动态重新编译');
    console.log('4. 使用合适的批次大小进行推理和训练');
    console.log('5. 考虑使用 WebWorker 处理数据预处理');
    console.log('6. 在浏览器中确保 WebGL 后端可用');
    console.log('7. 减少模型层数或隐藏维度以提升速度');
    console.log('');
}

// 运行所有诊断
async function runDiagnostics() {
    try {
        await detectBackendIssues();
        await detectTensorLeaks();
        await detectSyncBlocking();
        await detectRecompilation();
        await detectInefficientLayers();
        await detectDataProcessingBottleneck();
        await detectBatchSizeIssues();
        await detectTrainingEfficiency();
        
        printReport();
    } catch (error) {
        console.error('诊断过程出错:', error);
    }
}

runDiagnostics();
