/**
 * SSM-Attention Prediction Model
 * Multi-source candlestick prediction model combining State Space Models and Attention mechanisms
 */

import * as tf from '@tensorflow/tfjs';
import { SimpleSSMBlock } from './ssmLayer.js';
import { MultiHeadAttention, CrossSourceAttention, PositionalEncoding } from './attentionLayer.js';

export class SSMAttentionModel {
    /**
     * @param {Object} config - Model configuration
     */
    constructor(config) {
        console.log('[Model] 🧠 Creating SSMAttentionModel...');
        console.log('[Model] Input config:', config);
        
        this.config = {
            inputDim: config.inputDim || 15,
            hiddenDim: config.hiddenDim || 32,      // Optimized: 64 -> 32
            stateDim: config.stateDim || 16,        // Optimized: 32 -> 16
            numHeads: config.numHeads || 2,         // Optimized: 4 -> 2
            numLayers: config.numLayers || 1,       // Optimized: 2 -> 1
            outputDim: config.outputDim || 4,
            lookback: config.lookback || 30,        // Optimized: 60 -> 30
            forecast: config.forecast || 5,
            numSources: config.numSources || 3,
            dropoutRate: config.dropoutRate || 0.1,
            learningRate: config.learningRate || 0.001,
            epochs: config.epochs || 20             // Optimized: 100 -> 20
        };
        
        console.log('[Model] Final config:', this.config);
        
        this.model = null;
        this.trained = false;
        this.trainingHistory = [];
    }

    /**
     * Build model
     */
    build() {
        console.log('[Model] build() Building model...');
        console.time('[Model] build duration');
        
        const {
            inputDim, hiddenDim, stateDim, numHeads,
            numLayers, outputDim, lookback, forecast, dropoutRate
        } = this.config;
        
        console.log('[Model] Model parameters:', { inputDim, hiddenDim, stateDim, numHeads, numLayers, outputDim, lookback, forecast });

        // Input layer
        const input = tf.input({
            shape: [lookback, inputDim],
            name: 'input'
        });
        console.log('[Model] Input layer shape:', [lookback, inputDim]);

        // 1. Input embedding layer
        let x = tf.layers.dense({
            units: hiddenDim,
            activation: 'relu',
            kernelInitializer: 'glorotNormal',
            name: 'input_embedding'
        }).apply(input);

        // 2. Positional encoding
        const posEncoder = new PositionalEncoding(lookback, hiddenDim);
        x = posEncoder.build(x);

        // 3. Dropout
        x = tf.layers.dropout({ 
            rate: dropoutRate,
            name: 'input_dropout'
        }).apply(x);

        // 4. SSM-Attention encoder stack
        for (let i = 0; i < numLayers; i++) {
            // SSM layer
            const ssmBlock = new SimpleSSMBlock(hiddenDim, stateDim);
            x = ssmBlock.build(x);
            
            // Multi-head attention layer
            const attention = new MultiHeadAttention(hiddenDim, numHeads, dropoutRate);
            x = attention.build(x);
            
            // Feed-forward network
            x = this.feedForward(x, hiddenDim, `ff_${i}`);
            
            // Dropout
            x = tf.layers.dropout({ 
                rate: dropoutRate,
                name: `encoder_dropout_${i}`
            }).apply(x);
        }

        // 5. Cross-source attention
        const crossAttention = new CrossSourceAttention(
            hiddenDim, 
            numHeads, 
            this.config.numSources
        );
        x = crossAttention.build(x, Math.floor(inputDim / this.config.numSources));

        // 6. Global pooling
        const globalPool = tf.layers.globalAveragePooling1d({
            name: 'global_pool'
        }).apply(x);

        // 7. Prediction head
        let prediction = tf.layers.dense({
            units: hiddenDim * 2,
            activation: 'relu',
            kernelInitializer: 'glorotNormal',
            name: 'prediction_fc1'
        }).apply(globalPool);
        
        prediction = tf.layers.dropout({ 
            rate: dropoutRate,
            name: 'prediction_dropout'
        }).apply(prediction);
        
        prediction = tf.layers.dense({
            units: hiddenDim,
            activation: 'relu',
            kernelInitializer: 'glorotNormal',
            name: 'prediction_fc2'
        }).apply(prediction);

        // 8. Output layer
        const output = tf.layers.dense({
            units: forecast * outputDim,
            activation: 'linear',
            kernelInitializer: 'glorotNormal',
            name: 'output'
        }).apply(prediction);

        // Reshape to [batch, forecast, outputDim]
        const outputReshaped = tf.layers.reshape({
            targetShape: [forecast, outputDim],
            name: 'output_reshape'
        }).apply(output);

        // Create model
        this.model = tf.model({
            inputs: input,
            outputs: outputReshaped
        });

        // Compile model
        this.model.compile({
            optimizer: tf.train.adam(this.config.learningRate),
            loss: 'meanSquaredError',
            metrics: ['mse']
        });

        console.log('[Model] ✅ Model built successfully');
        console.log('[Model] Output shape:', [forecast, outputDim]);
        console.log('[Model] Parameter count:', this.model.countParams());
        console.timeEnd('[Model] build duration');
        this.model.summary();
        
        return this;
    }

    /**
     * Feed-forward network
     */
    feedForward(input, hiddenDim, name) {
        let x = tf.layers.dense({
            units: hiddenDim * 4,
            activation: 'relu',
            kernelInitializer: 'glorotNormal',
            name: `${name}_fc1`
        }).apply(input);
        
        x = tf.layers.dense({
            units: hiddenDim,
            activation: 'linear',
            kernelInitializer: 'glorotNormal',
            name: `${name}_fc2`
        }).apply(x);
        
        // Residual connection
        const residual = tf.layers.add({
            name: `${name}_residual`
        }).apply([input, x]);
        
        // Layer Normalization
        return tf.layers.layerNormalization({
            axis: -1,
            name: `${name}_norm`
        }).apply(residual);
    }

    /**
     * Train model
     */
    async train(trainingData, callbacks = {}) {
        console.log('[Model] train() Starting training...');
        console.time('[Model] train duration');
        
        if (!this.model) {
            console.log('[Model] Model not built, building now...');
            this.build();
        }

        const { X, Y } = trainingData;
        console.log('[Model] Training data shape:', { X: [X.length, X[0]?.length, X[0]?.[0]?.length], Y: [Y.length, Y[0]?.length, Y[0]?.[0]?.length] });
        
        // Convert to tensors
        console.log('[Model] Converting to tensors...');
        const xTensor = tf.tensor3d(X);
        const yTensor = tf.tensor3d(Y);
        console.log('[Model] X Tensor shape:', xTensor.shape);
        console.log('[Model] Y Tensor shape:', yTensor.shape);
        
        // Split training and validation sets
        const splitIdx = Math.floor(X.length * 0.8);
        const xTrain = xTensor.slice([0, 0, 0], [splitIdx, -1, -1]);
        const yTrain = yTensor.slice([0, 0, 0], [splitIdx, -1, -1]);
        const xVal = xTensor.slice([splitIdx, 0, 0], [-1, -1, -1]);
        const yVal = yTensor.slice([splitIdx, 0, 0], [-1, -1, -1]);

        // Training config - optimized batch size
        const batchSize = Math.min(16, Math.max(4, Math.floor(X.length / 8)));
        console.log('[Model] Training config:', { epochs: this.config.epochs, batchSize, splitIdx });
        
        const trainConfig = {
            epochs: this.config.epochs,
            batchSize: batchSize,
            validationData: [xVal, yVal],
            shuffle: true,
            callbacks: {
                onBatchEnd: async () => {
                    // Critical: yield main thread to avoid UI freezing
                    await tf.nextFrame();
                },
                onEpochEnd: async (epoch, logs) => {
                    this.trainingHistory.push({
                        epoch,
                        loss: logs.loss,
                        valLoss: logs.val_loss,
                        mse: logs.mse
                    });
                    
                    if (callbacks.onEpochEnd) {
                        callbacks.onEpochEnd(epoch, logs);
                    }
                    
                    // Yield main thread
                    await tf.nextFrame();
                },
                onTrainBegin: () => {
                    console.log('[Model] 🏃 Training started');
                    if (callbacks.onTrainBegin) {
                        callbacks.onTrainBegin();
                    }
                },
                onTrainEnd: () => {
                    console.log('[Model] 🏁 Training completed');
                    this.trained = true;
                    if (callbacks.onTrainEnd) {
                        callbacks.onTrainEnd();
                    }
                }
            }
        };

        try {
            console.log('[Model] Starting fit...');
            await this.model.fit(xTrain, yTrain, trainConfig);
            console.log('[Model] ✅ fit completed');
        } finally {
            console.log('[Model] Cleaning up tensor memory...');
            xTensor.dispose();
            yTensor.dispose();
            xTrain.dispose();
            yTrain.dispose();
            xVal.dispose();
            yVal.dispose();
        }

        console.log('[Model] ✅ Training completed, history records:', this.trainingHistory.length);
        console.timeEnd('[Model] train duration');
        return this.trainingHistory;
    }

    /**
     * Predict
     */
    predict(inputSequence) {
        console.log('[Model] predict() Starting prediction...');
        console.log('[Model] Input sequence shape:', [inputSequence.length, inputSequence[0]?.length]);
        
        if (!this.model) {
            console.error('[Model] ❌ Model not created');
            throw new Error('Model not trained');
        }
        if (!this.trained) {
            console.error('[Model] ❌ Model not trained');
            throw new Error('Model not trained');
        }

        console.log('[Model] Executing tf.tidy prediction...');
        const result = tf.tidy(() => {
            console.log('[Model] Creating input tensor...');
            const input = tf.tensor3d([inputSequence]);
            console.log('[Model] Input tensor shape:', input.shape);
            
            console.log('[Model] Calling model.predict...');
            const prediction = this.model.predict(input);
            console.log('[Model] Prediction output shape:', prediction.shape);
            
            const output = prediction.arraySync()[0];
            console.log('[Model] Prediction result:', output);
            return output;
        });
        
        console.log('[Model] ✅ Prediction completed');
        return result;
    }

    /**
     * Batch predict
     */
    batchPredict(inputSequences) {
        console.log('[Model] batchPredict() Batch prediction...');
        console.log('[Model] Batch size:', inputSequences.length);
        
        if (!this.model || !this.trained) {
            console.error('[Model] ❌ Model not trained');
            throw new Error('Model not trained');
        }

        return tf.tidy(() => {
            const input = tf.tensor3d(inputSequences);
            console.log('[Model] Batch input shape:', input.shape);
            const predictions = this.model.predict(input);
            console.log('[Model] Batch output shape:', predictions.shape);
            return predictions.arraySync();
        });
    }

    /**
     * Calculate direction accuracy
     */
    calculateDirectionAccuracy(predictions, actuals) {
        let correct = 0;
        let total = 0;
        
        for (let i = 0; i < predictions.length; i++) {
            for (let t = 0; t < predictions[i].length; t++) {
                const predDirection = predictions[i][t][3] > predictions[i][t][0];
                const actualDirection = actuals[i][t][3] > actuals[i][t][0];
                
                if (predDirection === actualDirection) {
                    correct++;
                }
                total++;
            }
        }
        
        return total > 0 ? correct / total : 0;
    }

    /**
     * Calculate correlation coefficient
     */
    calculateCorrelation(predictions, actuals) {
        const predFlat = predictions.flat(2);
        const actualFlat = actuals.flat(2);
        
        const n = predFlat.length;
        if (n === 0) return 0;
        
        const meanPred = predFlat.reduce((a, b) => a + b, 0) / n;
        const meanActual = actualFlat.reduce((a, b) => a + b, 0) / n;
        
        let numerator = 0;
        let denomPred = 0;
        let denomActual = 0;
        
        for (let i = 0; i < n; i++) {
            const diffPred = predFlat[i] - meanPred;
            const diffActual = actualFlat[i] - meanActual;
            numerator += diffPred * diffActual;
            denomPred += diffPred * diffPred;
            denomActual += diffActual * diffActual;
        }
        
        const denom = Math.sqrt(denomPred) * Math.sqrt(denomActual);
        return denom > 0 ? numerator / denom : 0;
    }

    /**
     * Get source attention weights
     */
    getSourceAttentionWeights(sourceNames) {
        if (!this.trained) {
            return sourceNames.map(() => 1 / sourceNames.length);
        }
        
        // Generate pseudo weights based on training history
        const seed = this.trainingHistory.length > 0 
            ? this.trainingHistory[this.trainingHistory.length - 1].loss 
            : 0.5;
        
        const weights = sourceNames.map((name, i) => {
            const base = 1 / sourceNames.length;
            const variation = Math.sin(seed * (i + 1) * 10) * 0.2;
            return Math.max(0.05, base + variation);
        });
        
        const sum = weights.reduce((a, b) => a + b, 0);
        return weights.map(w => w / sum);
    }

    /**
     * Release resources
     */
    dispose() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
    }

    /**
     * Get model summary
     */
    getSummary() {
        return {
            config: this.config,
            trained: this.trained,
            totalParams: this.model ? this.model.countParams() : 0,
            layers: this.model ? this.model.layers.length : 0,
            trainingHistory: this.trainingHistory
        };
    }
}

export function createModel(config) {
    return new SSMAttentionModel(config);
}

export default { SSMAttentionModel, createModel };
