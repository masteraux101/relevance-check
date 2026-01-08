/**
 * State Space Model Layer
 * Implements SSM functionality using standard TensorFlow.js layers
 */

import * as tf from '@tensorflow/tfjs';

/**
 * Simplified SSM Block - Using standard TensorFlow.js layers
 */
export class SimpleSSMBlock {
    constructor(hiddenDim, stateDim) {
        this.hiddenDim = hiddenDim;
        this.stateDim = stateDim;
        this.counter = Math.random().toString(36).substr(2, 9);
    }

    /**
     * Build SSM block
     * @param {tf.SymbolicTensor} input - Input tensor
     * @returns {tf.SymbolicTensor} - Output tensor
     */
    build(input) {
        const id = this.counter;
        
        // Input projection
        let x = tf.layers.dense({
            units: this.hiddenDim,
            activation: 'linear',
            kernelInitializer: 'glorotNormal',
            name: `ssm_input_proj_${id}`
        }).apply(input);

        // Project to state space
        const stateProjection = tf.layers.dense({
            units: this.stateDim,
            activation: 'tanh',
            kernelInitializer: 'glorotNormal',
            name: `ssm_state_proj_${id}`
        }).apply(x);

        // Gating mechanism - selective processing
        const gate = tf.layers.dense({
            units: this.stateDim,
            activation: 'sigmoid',
            kernelInitializer: 'glorotNormal',
            name: `ssm_gate_${id}`
        }).apply(x);

        // Apply gating
        const gatedState = tf.layers.multiply({
            name: `ssm_gated_${id}`
        }).apply([stateProjection, gate]);

        // Use GRU for sequence modeling (simulating state transitions)
        const rnnOutput = tf.layers.gru({
            units: this.stateDim,
            returnSequences: true,
            kernelInitializer: 'glorotNormal',
            recurrentInitializer: 'orthogonal',
            name: `ssm_gru_${id}`
        }).apply(gatedState);

        // State to output projection
        const output = tf.layers.dense({
            units: this.hiddenDim,
            activation: 'linear',
            kernelInitializer: 'glorotNormal',
            name: `ssm_output_proj_${id}`
        }).apply(rnnOutput);

        // Residual connection
        const residual = tf.layers.add({
            name: `ssm_residual_${id}`
        }).apply([x, output]);

        // Layer Normalization
        const normalized = tf.layers.layerNormalization({
            axis: -1,
            name: `ssm_layer_norm_${id}`
        }).apply(residual);

        return normalized;
    }
}

/**
 * Bidirectional SSM Block
 */
export class BidirectionalSSMBlock {
    constructor(hiddenDim, stateDim) {
        this.hiddenDim = hiddenDim;
        this.stateDim = stateDim;
        this.counter = Math.random().toString(36).substr(2, 9);
    }

    build(input) {
        const id = this.counter;
        
        // Forward GRU
        const forwardGru = tf.layers.gru({
            units: this.stateDim,
            returnSequences: true,
            goBackwards: false,
            kernelInitializer: 'glorotNormal',
            name: `bidir_forward_${id}`
        }).apply(input);

        // Backward GRU
        const backwardGru = tf.layers.gru({
            units: this.stateDim,
            returnSequences: true,
            goBackwards: true,
            kernelInitializer: 'glorotNormal',
            name: `bidir_backward_${id}`
        }).apply(input);

        // Merge
        const merged = tf.layers.concatenate({
            axis: -1,
            name: `bidir_concat_${id}`
        }).apply([forwardGru, backwardGru]);

        // Project back to original dimension
        const output = tf.layers.dense({
            units: this.hiddenDim,
            activation: 'linear',
            name: `bidir_proj_${id}`
        }).apply(merged);

        return output;
    }
}

export default { SimpleSSMBlock, BidirectionalSSMBlock };
