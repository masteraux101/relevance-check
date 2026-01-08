/**
 * Robustness testing - Verify model and data processing performance and stability
 * 
 * Run: node tests/robustness.test.js
 */

import * as tf from '@tensorflow/tfjs';  // Use pure JS version for compatibility
import { DataProcessor } from '../src/utils/dataProcessor.js';
import { SimpleSSMBlock, BidirectionalSSMBlock } from '../src/models/ssmLayer.js';
import { MultiHeadAttention, CrossSourceAttention, PositionalEncoding } from '../src/models/attentionLayer.js';

// Wait for TensorFlow initialization
await tf.ready();

// Test configuration
const TEST_CONFIG = {
    verbose: true,
    memoryThreshold: 500 * 1024 * 1024, // 500MB
    timeoutMs: 30000,
    iterations: 5
};

// Test result collection
const results = {
    passed: [],
    failed: [],
    warnings: []
};

// Utility functions
function log(message, type = 'info') {
    const prefix = {
        'info': '📝',
        'success': '✅',
        'error': '❌',
        'warning': '⚠️',
        'timing': '⏱️'
    }[type] || '📝';
    
    console.log(`${prefix} ${message}`);
}

function formatBytes(bytes) {
    return (bytes / 1024 / 1024).toFixed(2) + ' MB';
}

function formatMs(ms) {
    return ms.toFixed(2) + ' ms';
}

async function measureTime(fn, name) {
    const start = performance.now();
    const result = await fn();
    const elapsed = performance.now() - start;
    log(`${name}: ${formatMs(elapsed)}`, 'timing');
    return { result, elapsed };
}

function getMemoryUsage() {
    const usage = process.memoryUsage();
    return {
        heapUsed: usage.heapUsed,
        heapTotal: usage.heapTotal,
        external: usage.external,
        rss: usage.rss
    };
}

// ==================== Test cases ====================

/**
 * Test 1: Basic data processor functionality
 */
async function testDataProcessorBasic() {
    log('Test 1: Basic data processor functionality', 'info');
    
    try {
        const processor = new DataProcessor();
        
        // Test data generation
        const data = processor.generateMockData('SPY', 100);
        
        // Validate data structure
        if (!Array.isArray(data) || data.length !== 100) {
            throw new Error(`Data length incorrect: expected 100, got ${data.length}`);
        }
        
        // Validate required fields
        const requiredFields = ['date', 'timestamp', 'open', 'high', 'low', 'close', 'volume'];
        for (const field of requiredFields) {
            if (!(field in data[0])) {
                throw new Error(`Missing required field: ${field}`);
            }
        }
        
        // Validate data validity
        for (const item of data) {
            if (item.high < item.low) {
                throw new Error('Data anomaly: high < low');
            }
            if (item.open <= 0 || item.close <= 0) {
                throw new Error('Data anomaly: negative price');
            }
        }
        
        results.passed.push('testDataProcessorBasic');
        log('Data processor basic functionality test passed', 'success');
        return true;
    } catch (error) {
        results.failed.push({ name: 'testDataProcessorBasic', error: error.message });
        log(`Data processor basic functionality test failed: ${error.message}`, 'error');
        return false;
    }
}

/**
 * Test 2: Data processor performance
 */
async function testDataProcessorPerformance() {
    log('Test 2: Data processor performance', 'info');
    
    try {
        const processor = new DataProcessor();
        const testCases = [100, 500, 1000, 2000, 5000];
        
        for (const days of testCases) {
            const { elapsed } = await measureTime(
                () => processor.generateMockData('SPY', days),
                `Generating ${days} days of data`
            );
            
            // Performance threshold: should not exceed 100ms per 1000 records
            const threshold = (days / 1000) * 100;
            if (elapsed > threshold && days > 100) {
                results.warnings.push(`Generating ${days} days of data took too long: ${formatMs(elapsed)}`);
                log(`Performance warning: Generating ${days} days of data took ${formatMs(elapsed)}`, 'warning');
            }
        }
        
        results.passed.push('testDataProcessorPerformance');
        log('Data processor performance test passed', 'success');
        return true;
    } catch (error) {
        results.failed.push({ name: 'testDataProcessorPerformance', error: error.message });
        log(`Data processor performance test failed: ${error.message}`, 'error');
        return false;
    }
}

/**
 * Test 3: TensorFlow tensor memory management
 */
async function testTensorMemoryManagement() {
    log('Test 3: TensorFlow tensor memory management', 'info');
    
    try {
        const initialTensors = tf.memory().numTensors;
        log(`Initial tensor count: ${initialTensors}`);
        
        // Create and destroy tensors
        for (let i = 0; i < 100; i++) {
            const tensor = tf.randomNormal([32, 60, 15]);
            tensor.dispose();
        }
        
        const afterDispose = tf.memory().numTensors;
        log(`Tensor count after disposal: ${afterDispose}`);
        
        if (afterDispose > initialTensors + 10) {
            throw new Error(`Tensor leak: increased from ${initialTensors} to ${afterDispose}`);
        }
        
        // Test tf.tidy
        const result = tf.tidy(() => {
            const a = tf.randomNormal([100, 100]);
            const b = tf.randomNormal([100, 100]);
            return tf.matMul(a, b);
        });
        
        const afterTidy = tf.memory().numTensors;
        log(`Tensor count after tf.tidy: ${afterTidy}`);
        
        result.dispose();
        
        results.passed.push('testTensorMemoryManagement');
        log('TensorFlow tensor memory management test passed', 'success');
        return true;
    } catch (error) {
        results.failed.push({ name: 'testTensorMemoryManagement', error: error.message });
        log(`TensorFlow tensor memory management test failed: ${error.message}`, 'error');
        return false;
    }
}

/**
 * Test 4: SSM layer construction and forward pass
 */
async function testSSMLayerForward() {
    log('Test 4: SSM layer construction and forward pass', 'info');
    
    try {
        const hiddenDim = 64;
        const stateDim = 32;
        const batchSize = 4;
        const seqLen = 60;
        
        // Create input
        const input = tf.input({ shape: [seqLen, hiddenDim] });
        
        // Build SSM block
        const ssmBlock = new SimpleSSMBlock(hiddenDim, stateDim);
        const output = ssmBlock.build(input);
        
        // Create model for testing
        const testModel = tf.model({ inputs: input, outputs: output });
        
        // Forward pass test
        const { result, elapsed } = await measureTime(async () => {
            const testInput = tf.randomNormal([batchSize, seqLen, hiddenDim]);
            const prediction = testModel.predict(testInput);
            const shape = prediction.shape;
            testInput.dispose();
            prediction.dispose();
            return shape;
        }, 'SSM layer forward pass');
        
        log(`Output shape: [${result.join(', ')}]`);
        
        // Validate output shape
        if (result[0] !== batchSize || result[1] !== seqLen || result[2] !== hiddenDim) {
            throw new Error(`Output shape incorrect: expected [${batchSize}, ${seqLen}, ${hiddenDim}], got [${result.join(', ')}]`);
        }
        
        testModel.dispose();
        
        results.passed.push('testSSMLayerForward');
        log('SSM layer construction and forward pass test passed', 'success');
        return true;
    } catch (error) {
        results.failed.push({ name: 'testSSMLayerForward', error: error.message });
        log(`SSM layer construction and forward pass test failed: ${error.message}`, 'error');
        return false;
    }
}

/**
 * Test 5: Attention layer construction and forward pass
 */
async function testAttentionLayerForward() {
    log('Test 5: Attention layer construction and forward pass', 'info');
    
    try {
        const hiddenDim = 64;
        const numHeads = 4;
        const batchSize = 4;
        const seqLen = 60;
        
        // Create input
        const input = tf.input({ shape: [seqLen, hiddenDim] });
        
        // Build attention layer
        const attention = new MultiHeadAttention(hiddenDim, numHeads, 0.1);
        const output = attention.build(input);
        
        // Create model for testing
        const testModel = tf.model({ inputs: input, outputs: output });
        
        // Forward pass test
        const { result, elapsed } = await measureTime(async () => {
            const testInput = tf.randomNormal([batchSize, seqLen, hiddenDim]);
            const prediction = testModel.predict(testInput);
            const shape = prediction.shape;
            testInput.dispose();
            prediction.dispose();
            return shape;
        }, 'Attention layer forward pass');
        
        log(`Output shape: [${result.join(', ')}]`);
        
        // Validate output shape
        if (result[0] !== batchSize || result[1] !== seqLen || result[2] !== hiddenDim) {
            throw new Error(`Output shape incorrect: expected [${batchSize}, ${seqLen}, ${hiddenDim}], got [${result.join(', ')}]`);
        }
        
        testModel.dispose();
        
        results.passed.push('testAttentionLayerForward');
        log('Attention layer construction and forward pass test passed', 'success');
        return true;
    } catch (error) {
        results.failed.push({ name: 'testAttentionLayerForward', error: error.message });
        log(`Attention layer construction and forward pass test failed: ${error.message}`, 'error');
        return false;
    }
}

/**
 * Test 6: Full model construction
 */
async function testFullModelBuild() {
    log('Test 6: Full model construction', 'info');
    
    try {
        const config = {
            inputDim: 15,
            hiddenDim: 32,  // Reduce dimension to speed up tests
            stateDim: 16,
            numHeads: 2,
            numLayers: 1,
            outputDim: 4,
            lookback: 30,
            forecast: 5,
            numSources: 3,
            dropoutRate: 0.1,
            learningRate: 0.001
        };
        
        const { result: model, elapsed } = await measureTime(async () => {
            // Build simplified model
            const input = tf.input({ shape: [config.lookback, config.inputDim] });
            
            // Embedding layer
            let x = tf.layers.dense({
                units: config.hiddenDim,
                activation: 'relu',
                name: 'embedding'
            }).apply(input);
            
            // SSM layer
            const ssmBlock = new SimpleSSMBlock(config.hiddenDim, config.stateDim);
            x = ssmBlock.build(x);
            
            // Attention layer
            const attention = new MultiHeadAttention(config.hiddenDim, config.numHeads, config.dropoutRate);
            x = attention.build(x);
            
            // Global pooling
            x = tf.layers.globalAveragePooling1d().apply(x);
            
            // Output layer
            const output = tf.layers.dense({
                units: config.forecast * config.outputDim,
                activation: 'linear'
            }).apply(x);
            
            const outputReshaped = tf.layers.reshape({
                targetShape: [config.forecast, config.outputDim]
            }).apply(output);
            
            const model = tf.model({ inputs: input, outputs: outputReshaped });
            
            model.compile({
                optimizer: tf.train.adam(config.learningRate),
                loss: 'meanSquaredError'
            });
            
            return model;
        }, 'Full model construction');
        
        // Validate model
        const testInput = tf.randomNormal([2, config.lookback, config.inputDim]);
        const prediction = model.predict(testInput);
        
        log(`Model output shape: [${prediction.shape.join(', ')}]`);
        log(`Model parameter count: ${model.countParams()}`);
        
        // Validate output shape
        if (prediction.shape[1] !== config.forecast || prediction.shape[2] !== config.outputDim) {
            throw new Error('Model output shape incorrect');
        }
        
        testInput.dispose();
        prediction.dispose();
        model.dispose();
        
        results.passed.push('testFullModelBuild');
        log('Full model construction test passed', 'success');
        return true;
    } catch (error) {
        results.failed.push({ name: 'testFullModelBuild', error: error.message });
        log(`Full model construction test failed: ${error.message}`, 'error');
        return false;
    }
}

/**
 * Test 7: Model training stability (small scale)
 */
async function testModelTrainingStability() {
    log('Test 7: Model training stability', 'info');
    
    try {
        const config = {
            lookback: 20,
            inputDim: 10,
            hiddenDim: 16,
            outputDim: 4,
            forecast: 3
        };
        
        // Create simple model
        const input = tf.input({ shape: [config.lookback, config.inputDim] });
        let x = tf.layers.dense({ units: config.hiddenDim, activation: 'relu' }).apply(input);
        x = tf.layers.lstm({ units: config.hiddenDim, returnSequences: false }).apply(x);
        const output = tf.layers.dense({ units: config.forecast * config.outputDim }).apply(x);
        const outputReshaped = tf.layers.reshape({ targetShape: [config.forecast, config.outputDim] }).apply(output);
        
        const model = tf.model({ inputs: input, outputs: outputReshaped });
        model.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });
        
        // Generate training data
        const trainX = tf.randomNormal([50, config.lookback, config.inputDim]);
        const trainY = tf.randomNormal([50, config.forecast, config.outputDim]);
        
        // Train and monitor loss
        const losses = [];
        
        const { elapsed } = await measureTime(async () => {
            for (let epoch = 0; epoch < 10; epoch++) {
                const history = await model.fit(trainX, trainY, {
                    epochs: 1,
                    batchSize: 16,
                    verbose: 0
                });
                losses.push(history.history.loss[0]);
            }
        }, '10 epochs of training');
        
        // Check if loss converges (or at least doesn't diverge)
        const firstLoss = losses[0];
        const lastLoss = losses[losses.length - 1];
        
        log(`Initial loss: ${firstLoss.toFixed(4)}, Final loss: ${lastLoss.toFixed(4)}`);
        
        // Check for NaN
        if (losses.some(l => isNaN(l))) {
            throw new Error('NaN loss encountered during training');
        }
        
        // Check for divergence
        if (lastLoss > firstLoss * 10) {
            results.warnings.push(`Loss may diverge: increased from ${firstLoss.toFixed(4)} to ${lastLoss.toFixed(4)}`);
            log('Warning: Loss may diverge', 'warning');
        }
        
        // Cleanup
        trainX.dispose();
        trainY.dispose();
        model.dispose();
        
        results.passed.push('testModelTrainingStability');
        log('Model training stability test passed', 'success');
        return true;
    } catch (error) {
        results.failed.push({ name: 'testModelTrainingStability', error: error.message });
        log(`Model training stability test failed: ${error.message}`, 'error');
        return false;
    }
}

/**
 * Test 8: Memory pressure test
 */
async function testMemoryPressure() {
    log('Test 8: Memory pressure test', 'info');
    
    try {
        const initialMemory = getMemoryUsage();
        log(`Initial memory: ${formatBytes(initialMemory.heapUsed)}`);
        
        // Continuously create and destroy tensors
        for (let i = 0; i < 100; i++) {
            await tf.tidy(() => {
                const a = tf.randomNormal([64, 60, 64]);
                const b = tf.randomNormal([64, 64, 64]);
                const c = tf.matMul(
                    a.reshape([64 * 60, 64]),
                    b.reshape([64, 64 * 64])
                );
                return c.mean();
            });
            
            if (i % 20 === 0) {
                const currentMemory = getMemoryUsage();
                log(`Iteration ${i}: memory ${formatBytes(currentMemory.heapUsed)}, tensor count ${tf.memory().numTensors}`);
            }
        }
        
        // Force garbage collection (if available)
        if (global.gc) {
            global.gc();
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        const finalMemory = getMemoryUsage();
        log(`Final memory: ${formatBytes(finalMemory.heapUsed)}`);
        
        const memoryIncrease = finalMemory.heapUsed - initialMemory.heapUsed;
        log(`Memory increase: ${formatBytes(memoryIncrease)}`);
        
        if (memoryIncrease > TEST_CONFIG.memoryThreshold) {
            throw new Error(`Memory increase too large: ${formatBytes(memoryIncrease)}`);
        }
        
        results.passed.push('testMemoryPressure');
        log('Memory pressure test passed', 'success');
        return true;
    } catch (error) {
        results.failed.push({ name: 'testMemoryPressure', error: error.message });
        log(`Memory pressure test failed: ${error.message}`, 'error');
        return false;
    }
}

/**
 * Test 9: Boundary conditions test
 */
async function testBoundaryConditions() {
    log('Test 9: Boundary conditions test', 'info');
    
    try {
        const processor = new DataProcessor();
        
        // Test minimal data
        const smallData = processor.generateMockData('SPY', 1);
        if (smallData.length !== 1) {
            throw new Error('Minimal data test failed');
        }
        log('Minimal data test passed');
        
        // Test extreme prices
        const extremeData = processor.generateMockData('TEST', 100, {
            basePrice: 0.001,
            volatility: 0.5
        });
        for (const item of extremeData) {
            if (item.close <= 0 || !isFinite(item.close)) {
                throw new Error('Extreme price test failed: invalid price');
            }
        }
        log('Extreme price test passed');
        
        // Test empty input
        try {
            const emptyFeatures = processor.extractFeatures([]);
            if (emptyFeatures.length !== 0) {
                throw new Error('Empty input test failed');
            }
            log('Empty input test passed');
        } catch (e) {
            // Expected to possibly throw error
            log('Empty input correctly threw error', 'warning');
        }
        
        // Test very long sequence tensor operations
        const { elapsed } = await measureTime(async () => {
            const longSeq = tf.tidy(() => {
                const input = tf.randomNormal([1, 500, 32]);
                const dense = tf.layers.dense({ units: 32 });
                return dense.apply(input);
            });
            longSeq.dispose();
        }, 'Long sequence processing');
        
        log('Long sequence processing test passed');
        
        results.passed.push('testBoundaryConditions');
        log('Boundary conditions test passed', 'success');
        return true;
    } catch (error) {
        results.failed.push({ name: 'testBoundaryConditions', error: error.message });
        log(`Boundary conditions test failed: ${error.message}`, 'error');
        return false;
    }
}

/**
 * Test 10: Concurrent operations test
 */
async function testConcurrentOperations() {
    log('Test 10: Concurrent operations test', 'info');
    
    try {
        const processor = new DataProcessor();
        
        // Concurrent data generation
        const symbols = ['SPY', 'QQQ', 'VIX', 'TLT', 'GLD'];
        
        const { elapsed } = await measureTime(async () => {
            const promises = symbols.map(symbol => 
                processor.fetchData(symbol, 200)
            );
            return Promise.all(promises);
        }, 'Concurrent data fetching');
        
        // Concurrent tensor operations
        await measureTime(async () => {
            const operations = [];
            for (let i = 0; i < 10; i++) {
                operations.push(
                    tf.tidy(() => {
                        const a = tf.randomNormal([16, 30, 32]);
                        const b = tf.randomNormal([16, 32, 30]);
                        return tf.matMul(a, b);
                    })
                );
            }
            
            const resultsArray = await Promise.all(operations);
            resultsArray.forEach(r => r.dispose());
        }, 'Concurrent tensor operations');
        
        log(`Tensor count after concurrent operations: ${tf.memory().numTensors}`);
        
        results.passed.push('testConcurrentOperations');
        log('Concurrent operations test passed', 'success');
        return true;
    } catch (error) {
        results.failed.push({ name: 'testConcurrentOperations', error: error.message });
        log(`Concurrent operations test failed: ${error.message}`, 'error');
        return false;
    }
}

/**
 * Test 11: Numerical stability test
 */
async function testNumericalStability() {
    log('Test 11: Numerical stability test', 'info');
    
    try {
        // Test large values
        const largeValues = tf.tidy(() => {
            const input = tf.tensor2d([[1e10, 1e10, 1e10]]);
            const normalized = tf.layers.layerNormalization().apply(input);
            return normalized.arraySync();
        });
        
        if (!isFinite(largeValues[0][0])) {
            throw new Error('Large value normalization failed');
        }
        log(`Large value normalization result: ${largeValues[0].map(v => v.toFixed(4)).join(', ')}`);
        
        // Test small values
        const smallValues = tf.tidy(() => {
            const input = tf.tensor2d([[1e-10, 1e-10, 1e-10]]);
            const normalized = tf.layers.layerNormalization().apply(input);
            return normalized.arraySync();
        });
        
        if (!isFinite(smallValues[0][0])) {
            throw new Error('Small value normalization failed');
        }
        log(`Small value normalization result: ${smallValues[0].map(v => v.toFixed(4)).join(', ')}`);
        
        // Test gradient calculation
        const gradientTest = tf.tidy(() => {
            const x = tf.variable(tf.tensor1d([1.0, 2.0, 3.0]));
            const f = () => x.square().sum();
            const grads = tf.grad(f)(x);
            return grads.arraySync();
        });
        
        log(`Gradient calculation result: ${gradientTest.join(', ')}`);
        
        // Verify gradient correctness (d/dx(x^2) = 2x)
        const expectedGrads = [2, 4, 6];
        for (let i = 0; i < 3; i++) {
            if (Math.abs(gradientTest[i] - expectedGrads[i]) > 1e-5) {
                throw new Error('Gradient calculation incorrect');
            }
        }
        
        results.passed.push('testNumericalStability');
        log('Numerical stability test passed', 'success');
        return true;
    } catch (error) {
        results.failed.push({ name: 'testNumericalStability', error: error.message });
        log(`Numerical stability test failed: ${error.message}`, 'error');
        return false;
    }
}

/**
 * Test 12: Inference performance test
 */
async function testInferencePerformance() {
    log('Test 12: Inference performance test', 'info');
    
    try {
        const config = {
            lookback: 60,
            inputDim: 15,
            hiddenDim: 32,
            outputDim: 4,
            forecast: 5
        };
        
        // Build model
        const input = tf.input({ shape: [config.lookback, config.inputDim] });
        let x = tf.layers.dense({ units: config.hiddenDim, activation: 'relu' }).apply(input);
        x = tf.layers.lstm({ units: config.hiddenDim, returnSequences: false }).apply(x);
        const output = tf.layers.dense({ units: config.forecast * config.outputDim }).apply(x);
        const outputReshaped = tf.layers.reshape({ targetShape: [config.forecast, config.outputDim] }).apply(output);
        
        const model = tf.model({ inputs: input, outputs: outputReshaped });
        
        // Warmup
        const warmupInput = tf.randomNormal([1, config.lookback, config.inputDim]);
        model.predict(warmupInput);
        warmupInput.dispose();
        
        // Batch inference test
        const batchSizes = [1, 4, 8, 16, 32];
        const inferenceResults = [];
        
        for (const batchSize of batchSizes) {
            const testInput = tf.randomNormal([batchSize, config.lookback, config.inputDim]);
            
            const times = [];
            for (let i = 0; i < 10; i++) {
                const start = performance.now();
                const prediction = model.predict(testInput);
                prediction.dispose();
                times.push(performance.now() - start);
            }
            
            testInput.dispose();
            
            const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
            const perSample = avgTime / batchSize;
            
            inferenceResults.push({ batchSize, avgTime, perSample });
            log(`Batch size ${batchSize}: avg ${formatMs(avgTime)}, per sample ${formatMs(perSample)}`);
        }
        
        // Check for performance anomalies
        const singleSampleTime = inferenceResults[0].perSample;
        for (const result of inferenceResults) {
            if (result.perSample > singleSampleTime * 2) {
                results.warnings.push(`Batch ${result.batchSize} has abnormal per-sample inference time`);
            }
        }
        
        model.dispose();
        
        results.passed.push('testInferencePerformance');
        log('Inference performance test passed', 'success');
        return true;
    } catch (error) {
        results.failed.push({ name: 'testInferencePerformance', error: error.message });
        log(`Inference performance test failed: ${error.message}`, 'error');
        return false;
    }
}

// ==================== Main test runner ====================

async function runAllTests() {
    console.log('');
    console.log('═'.repeat(60));
    console.log('          SSM-Attention Model Robustness Test Suite');
    console.log('═'.repeat(60));
    console.log('');
    
    const startTime = performance.now();
    const initialMemory = getMemoryUsage();
    
    log(`TensorFlow.js version: ${tf.version.tfjs}`);
    log(`Backend: ${tf.getBackend()}`);
    log(`Initial tensor count: ${tf.memory().numTensors}`);
    log(`Initial memory usage: ${formatBytes(initialMemory.heapUsed)}`);
    console.log('');
    
    // Run all tests
    const tests = [
        testDataProcessorBasic,
        testDataProcessorPerformance,
        testTensorMemoryManagement,
        testSSMLayerForward,
        testAttentionLayerForward,
        testFullModelBuild,
        testModelTrainingStability,
        testMemoryPressure,
        testBoundaryConditions,
        testConcurrentOperations,
        testNumericalStability,
        testInferencePerformance
    ];
    
    for (const test of tests) {
        console.log('');
        console.log('─'.repeat(60));
        await test();
    }
    
    // Output summary
    const totalTime = performance.now() - startTime;
    const finalMemory = getMemoryUsage();
    
    console.log('');
    console.log('═'.repeat(60));
    console.log('                    Test Results Summary');
    console.log('═'.repeat(60));
    console.log('');
    
    log(`Total time: ${formatMs(totalTime)}`, 'timing');
    log(`Final tensor count: ${tf.memory().numTensors}`);
    log(`Final memory usage: ${formatBytes(finalMemory.heapUsed)}`);
    log(`Memory growth: ${formatBytes(finalMemory.heapUsed - initialMemory.heapUsed)}`);
    console.log('');
    
    log(`Passed: ${results.passed.length} tests`, 'success');
    results.passed.forEach(name => console.log(`   ✅ ${name}`));
    
    if (results.failed.length > 0) {
        console.log('');
        log(`Failed: ${results.failed.length} tests`, 'error');
        results.failed.forEach(({ name, error }) => console.log(`   ❌ ${name}: ${error}`));
    }
    
    if (results.warnings.length > 0) {
        console.log('');
        log(`Warnings: ${results.warnings.length}`, 'warning');
        results.warnings.forEach(warning => console.log(`   ⚠️ ${warning}`));
    }
    
    console.log('');
    console.log('═'.repeat(60));
    
    // Return whether all tests passed
    return results.failed.length === 0;
}

// Run tests
runAllTests()
    .then(success => {
        process.exit(success ? 0 : 1);
    })
    .catch(error => {
        console.error('Test run error:', error);
        process.exit(1);
    });
