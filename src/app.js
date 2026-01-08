/**
 * Main application module - integrates all components
 */

import { DataProcessor } from './utils/dataProcessor.js';
import { SSMAttentionModel } from './models/model.js';
import { Visualization } from './utils/visualization.js';

export class App {
    constructor() {
        console.log('[App] 🚀 Initializing app...');
        this.dataProcessor = new DataProcessor();
        this.model = null;
        this.visualization = new Visualization();
        this.multiSourceData = null;
        this.trainingData = null;
        this.lastPrediction = null;
        this.targetSymbol = 'SPY';
        this.selectedSources = ['SPY', 'VIX', 'TLT'];
        this.dataSourceMode = 'mock';  // 'mock' or 'csv'
        this.uploadedFiles = new Map(); // Store uploaded file info
        
        console.log('[App] Config:', { targetSymbol: this.targetSymbol, selectedSources: this.selectedSources });
        this.init();
    }

    /**
     * Initialize application
     */
    init() {
        console.log('[App] init() Starting initialization...');
        // Wait for TensorFlow.js to load
        this.waitForTF().then(() => {
            console.log('[App] ✅ TensorFlow.js loaded');
            this.visualization.init();
            this.bindEvents();
            this.visualization.addLog('System initialized, TensorFlow.js loaded');
            this.visualization.addLog(`TensorFlow.js version: ${tf.version.tfjs}`, 'success');
            this.visualization.addLog('Please select data source mode and load data');
            console.log('[App] ✅ App initialization complete');
        });
    }

    /**
     * Wait for TensorFlow.js to load
     */
    async waitForTF() {
        return new Promise((resolve) => {
            const check = () => {
                if (typeof tf !== 'undefined') {
                    resolve();
                } else {
                    setTimeout(check, 100);
                }
            };
            check();
        });
    }

    /**
     * Bind events
     */
    bindEvents() {
        // Data source mode switch
        document.getElementById('dataSourceMode').addEventListener('change', (e) => {
            this.handleDataSourceModeChange(e.target.value);
        });

        // CSV file upload - target file
        document.getElementById('targetCsvFile').addEventListener('change', (e) => {
            this.handleTargetCSVUpload(e.target.files);
        });

        // CSV file upload - related files
        document.getElementById('relatedCsvFiles').addEventListener('change', (e) => {
            this.handleRelatedCSVUpload(e.target.files);
        });

        // Drag and drop upload
        this.setupDragAndDrop();

        // Load data button
        document.getElementById('loadDataBtn').addEventListener('click', () => {
            this.loadData();
        });

        // Train button
        document.getElementById('trainBtn').addEventListener('click', () => {
            this.train();
        });

        // Predict button
        document.getElementById('predictBtn').addEventListener('click', () => {
            this.predict();
        });

        // Tab switch
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const tab = e.target.dataset.tab;
                this.switchTab(tab);
            });
        });

        // Data source checkboxes
        document.querySelectorAll('.checkbox-group input').forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                this.updateSelectedSources();
            });
        });

        // Target selection
        document.getElementById('targetSymbol').addEventListener('change', (e) => {
            this.targetSymbol = e.target.value;
        });
    }

    /**
     * Setup drag and drop upload
     */
    setupDragAndDrop() {
        const uploadZone = document.getElementById('uploadZone');
        if (!uploadZone) return;

        // Prevent default drag and drop behavior
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
        });

        // Highlight on drag enter
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadZone.addEventListener(eventName, () => {
                uploadZone.classList.add('drag-over');
            });
        });

        // Remove highlight on drag leave
        ['dragleave', 'drop'].forEach(eventName => {
            uploadZone.addEventListener(eventName, () => {
                uploadZone.classList.remove('drag-over');
            });
        });

        // Handle dropped files
        uploadZone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleDroppedFiles(files);
            }
        });
    }

    /**
     * Handle dropped files
     */
    async handleDroppedFiles(files) {
        const csvFiles = Array.from(files).filter(f => f.name.endsWith('.csv'));
        
        if (csvFiles.length === 0) {
            this.visualization.addLog('❌ Please drop CSV format files', 'error');
            return;
        }

        this.visualization.addLog(`Detected ${csvFiles.length} CSV file(s)`, 'info');

        // If only one file and no target file, set as target
        if (csvFiles.length === 1 && !this.hasTargetFile()) {
            await this.handleTargetCSVUpload(csvFiles);
        } else if (!this.hasTargetFile()) {
            // First file as target, rest as related
            await this.handleTargetCSVUpload([csvFiles[0]]);
            if (csvFiles.length > 1) {
                await this.handleRelatedCSVUpload(csvFiles.slice(1));
            }
        } else {
            // Already has target file, all as related files
            await this.handleRelatedCSVUpload(csvFiles);
        }
    }

    /**
     * Check if target file exists
     */
    hasTargetFile() {
        for (const [, info] of this.uploadedFiles) {
            if (info.type === 'target') return true;
        }
        return false;
    }

    /**
     * Handle data source mode change
     */
    handleDataSourceModeChange(mode) {
        console.log('[App] Data source mode changed:', mode);
        this.dataSourceMode = mode;
        
        const csvSection = document.getElementById('csvUploadSection');
        const mockSection = document.getElementById('mockDataSection');
        
        if (mode === 'csv') {
            csvSection.style.display = 'block';
            mockSection.style.display = 'none';
            this.visualization.addLog('Switched to CSV mode. Please upload yfinance format CSV files', 'info');
        } else {
            csvSection.style.display = 'none';
            mockSection.style.display = 'block';
            this.visualization.addLog('Switched to mock data mode', 'info');
            // Clear uploaded CSV data
            this.dataProcessor.clearCSVData();
            this.uploadedFiles.clear();
            this.updateUploadedFilesList();
        }
    }

    /**
     * Handle target CSV file upload
     */
    async handleTargetCSVUpload(files) {
        if (!files || files.length === 0) return;
        
        const file = files[0];
        console.log('[App] Uploading target CSV file:', file.name);
        this.visualization.addLog(`Parsing target file: ${file.name}...`);
        
        try {
            const result = await this.dataProcessor.loadCSVFile(file);
            this.targetSymbol = result.symbol;
            this.uploadedFiles.set(result.symbol, { 
                type: 'target', 
                name: file.name, 
                rows: result.rowCount 
            });
            
            this.visualization.addLog(
                `✅ Target file ${result.symbol} loaded: ${result.rowCount} rows`, 
                'success'
            );
            this.updateUploadedFilesList();
        } catch (error) {
            console.error('[App] CSV parse failed:', error);
            this.visualization.addLog(`❌ Parse failed: ${error.message}`, 'error');
        }
    }

    /**
     * Handle related CSV files upload
     */
    async handleRelatedCSVUpload(files) {
        if (!files || files.length === 0) return;
        
        console.log('[App] Uploading related CSV files:', files.length);
        
        for (const file of files) {
            this.visualization.addLog(`Parsing related file: ${file.name}...`);
            
            try {
                const result = await this.dataProcessor.loadCSVFile(file);
                this.uploadedFiles.set(result.symbol, { 
                    type: 'related', 
                    name: file.name, 
                    rows: result.rowCount 
                });
                
                this.visualization.addLog(
                    `✅ Related file ${result.symbol} loaded: ${result.rowCount} rows`, 
                    'success'
                );
            } catch (error) {
                console.error('[App] CSV parse failed:', error);
                this.visualization.addLog(`❌ ${file.name} parse failed: ${error.message}`, 'error');
            }
        }
        
        this.updateUploadedFilesList();
    }

    /**
     * Update uploaded files list display
     */
    updateUploadedFilesList() {
        const container = document.getElementById('uploadedFilesList');
        container.innerHTML = '';
        
        if (this.uploadedFiles.size === 0) {
            return;
        }
        
        this.uploadedFiles.forEach((info, symbol) => {
            const item = document.createElement('div');
            item.className = 'uploaded-file-item';
            item.innerHTML = `
                <span class="file-name">${info.type === 'target' ? '🎯' : '📊'} ${symbol}</span>
                <span class="file-rows">${info.rows} rows</span>
                <span class="remove-file" data-symbol="${symbol}">✕</span>
            `;
            container.appendChild(item);
        });
        
        // Bind delete events
        container.querySelectorAll('.remove-file').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const symbol = e.target.dataset.symbol;
                this.removeUploadedFile(symbol);
            });
        });
    }

    /**
     * Remove uploaded file
     */
    removeUploadedFile(symbol) {
        console.log('[App] Removing file:', symbol);
        this.uploadedFiles.delete(symbol);
        this.dataProcessor.csvData.delete(symbol);
        this.updateUploadedFilesList();
        this.visualization.addLog(`Removed: ${symbol}`, 'info');
    }

    /**
     * Update selected sources
     */
    updateSelectedSources() {
        const checkboxes = document.querySelectorAll('.checkbox-group input:checked');
        this.selectedSources = [this.targetSymbol];
        checkboxes.forEach(cb => {
            if (cb.value !== this.targetSymbol && !this.selectedSources.includes(cb.value)) {
                this.selectedSources.push(cb.value);
            }
        });
    }

    /**
     * Load data
     */
    async loadData() {
        console.log('[App] loadData() Starting data load...');
        console.log('[App] Data source mode:', this.dataSourceMode);
        console.time('[App] loadData duration');
        
        const loadBtn = document.getElementById('loadDataBtn');
        const trainBtn = document.getElementById('trainBtn');
        
        loadBtn.disabled = true;
        loadBtn.innerHTML = '<span class="loading"></span> Loading...';
        
        try {
            const lookbackDays = parseInt(document.getElementById('lookbackDays').value);
            const forecastDays = parseInt(document.getElementById('forecastDays').value);
            
            if (this.dataSourceMode === 'csv') {
                // CSV mode
                await this.loadDataFromCSV(lookbackDays, forecastDays);
            } else {
                // Mock data mode
                await this.loadDataFromMock(lookbackDays, forecastDays);
            }
            
            // Enable train button
            trainBtn.disabled = false;
            console.log('[App] ✅ Data loading complete');
            console.timeEnd('[App] loadData duration');
            
        } catch (error) {
            console.error('[App] ❌ Data loading failed:', error);
            this.visualization.addLog(`Data loading failed: ${error.message}`, 'error');
            console.error(error);
        } finally {
            loadBtn.disabled = false;
            loadBtn.innerHTML = '📥 Load Data';
        }
    }

    /**
     * Load data from CSV
     */
    async loadDataFromCSV(lookbackDays, forecastDays) {
        console.log('[App] loadDataFromCSV()');
        
        // Check if there are uploaded target files
        const loadedSymbols = this.dataProcessor.getLoadedCSVSymbols();
        console.log('[App] Loaded CSV symbols:', loadedSymbols);
        
        if (loadedSymbols.length === 0) {
            throw new Error('Please upload CSV data files first');
        }
        
        // Find target symbol
        let targetFile = null;
        let relatedFiles = [];
        
        this.uploadedFiles.forEach((info, symbol) => {
            if (info.type === 'target') {
                targetFile = symbol;
            } else {
                relatedFiles.push(symbol);
            }
        });
        
        if (!targetFile) {
            // If no explicit target file, use the first uploaded file
            targetFile = loadedSymbols[0];
            relatedFiles = loadedSymbols.slice(1);
        }
        
        this.targetSymbol = targetFile;
        this.selectedSources = [targetFile, ...relatedFiles];
        
        console.log('[App] CSV data config:', { target: targetFile, related: relatedFiles });
        this.visualization.addLog(`CSV mode - Target: ${targetFile}, Related: ${relatedFiles.join(', ') || 'none'}`);
        
        // Build multi-source data
        this.multiSourceData = {};
        for (const symbol of this.selectedSources) {
            const data = this.dataProcessor.getCSVData(symbol);
            if (data) {
                this.multiSourceData[symbol] = data;
                this.visualization.addLog(`✅ ${symbol}: ${data.length} records`);
            }
        }
        
        // Validate data amount
        const targetData = this.multiSourceData[targetFile];
        if (!targetData || targetData.length < lookbackDays + forecastDays + 10) {
            throw new Error(`Insufficient data. Need at least ${lookbackDays + forecastDays + 10} records, got ${targetData?.length || 0}`);
        }
        
        this.visualization.addLog(`Data loaded: ${targetData.length} records`, 'success');
        
        // Display candlestick chart
        this.visualization.drawCandlestick(targetData, targetFile);
        
        // Prepare training data
        this.trainingData = this.dataProcessor.prepareTrainingData(
            this.multiSourceData,
            targetFile,
            lookbackDays,
            forecastDays
        );
        
        this.visualization.addLog(
            `Training data ready: ${this.trainingData.X.length} samples`,
            'success'
        );
    }

    /**
     * Load data from mock
     */
    async loadDataFromMock(lookbackDays, forecastDays) {
        console.log('[App] loadDataFromMock()');
        
        this.updateSelectedSources();
        const totalDays = lookbackDays + 50; // Extra data for training
        
        console.log('[App] Mock data config:', { lookbackDays, totalDays, sources: this.selectedSources });
        this.visualization.addLog(`Loading mock data: ${this.selectedSources.join(', ')}`);
        
        // First load target data
        await this.dataProcessor.fetchData(this.targetSymbol, totalDays);
        
        // Load multi-source data
        this.multiSourceData = {};
        for (const symbol of this.selectedSources) {
            this.visualization.addLog(`Loading ${symbol} data...`);
            const data = await this.dataProcessor.fetchData(symbol, totalDays);
            this.multiSourceData[symbol] = data;
            await this.delay(100);
        }
        
        this.visualization.addLog(`Data loaded: ${totalDays} days`, 'success');
        
        // Display candlestick chart
        this.visualization.drawCandlestick(
            this.multiSourceData[this.targetSymbol],
            this.targetSymbol
        );
        
        // Prepare training data
        this.trainingData = this.dataProcessor.prepareTrainingData(
            this.multiSourceData,
            this.targetSymbol,
            lookbackDays,
            forecastDays
        );
        
        this.visualization.addLog(
            `Training data ready: ${this.trainingData.X.length} samples`,
            'success'
        );
    }

    /**
     * Train model
     */
    async train() {
        console.log('[App] train() Starting model training...');
        console.time('[App] train duration');
        
        if (!this.trainingData) {
            console.warn('[App] ⚠️ No training data');
            this.visualization.addLog('Please load data first', 'warning');
            return;
        }

        const trainBtn = document.getElementById('trainBtn');
        const predictBtn = document.getElementById('predictBtn');
        
        trainBtn.disabled = true;
        trainBtn.innerHTML = '<span class="loading"></span> Training...';
        
        try {
            // Get configuration
            const config = this.getModelConfig();
            
            this.visualization.addLog('Building model...');
            
            // Create model
            if (this.model) {
                this.model.dispose();
            }
            
            this.model = new SSMAttentionModel(config);
            this.model.build();
            
            this.visualization.addLog(
                `Model built. Parameters: ${this.model.model.countParams().toLocaleString()}`,
                'success'
            );
            
            // Train model
            this.visualization.addLog(`Starting training for ${config.epochs} epochs...`);
            
            const callbacks = {
                onEpochEnd: (epoch, logs) => {
                    if (epoch % 10 === 0 || epoch === config.epochs - 1) {
                        this.visualization.addLog(
                            `Epoch ${epoch + 1}/${config.epochs} - loss: ${logs.loss.toFixed(6)} - val_loss: ${logs.val_loss.toFixed(6)}`
                        );
                    }
                    
                    // Update metrics
                    this.visualization.updateMetrics({
                        trainLoss: logs.loss,
                        valLoss: logs.val_loss
                    });
                },
                onTrainEnd: () => {
                    this.visualization.addLog('Training complete!', 'success');
                    
                    // Update attention weight visualization
                    const weights = this.model.getSourceAttentionWeights(this.selectedSources);
                    this.visualization.updateAttentionVisualization(
                        this.selectedSources,
                        weights
                    );
                }
            };
            
            // Set training epochs
            this.model.config.epochs = config.epochs;
            
            await this.model.train(this.trainingData, callbacks);
            
            // Calculate additional metrics
            await this.calculateMetrics();
            
            // Enable predict button
            predictBtn.disabled = false;
            console.log('[App] ✅ Training complete');
            console.timeEnd('[App] train duration');
            
        } catch (error) {
            console.error('[App] ❌ Training failed:', error);
            this.visualization.addLog(`Training failed: ${error.message}`, 'error');
            console.error(error);
        } finally {
            trainBtn.disabled = false;
            trainBtn.innerHTML = '🚀 Train Model';
        }
    }

    /**
     * Calculate evaluation metrics
     */
    async calculateMetrics() {
        if (!this.model || !this.model.trained) return;
        
        // Use validation set to calculate metrics
        const splitIdx = Math.floor(this.trainingData.X.length * 0.8);
        const valX = this.trainingData.X.slice(splitIdx);
        const valY = this.trainingData.Y.slice(splitIdx);
        
        // Batch prediction
        const predictions = this.model.batchPredict(valX);
        
        // Calculate direction accuracy
        const dirAccuracy = this.model.calculateDirectionAccuracy(predictions, valY);
        
        // Calculate correlation
        const correlation = this.model.calculateCorrelation(predictions, valY);
        
        this.visualization.updateMetrics({
            dirAccuracy,
            correlation
        });
        
        this.visualization.addLog(
            `Direction accuracy: ${(dirAccuracy * 100).toFixed(1)}%, Correlation: ${correlation.toFixed(4)}`,
            'success'
        );
    }

    /**
     * Make prediction
     */
    async predict() {
        console.log('[App] predict() Starting prediction...');
        console.time('[App] predict duration');
        
        if (!this.model || !this.model.trained) {
            console.warn('[App] ⚠️ Model not trained');
            this.visualization.addLog('Please train the model first', 'warning');
            return;
        }

        const predictBtn = document.getElementById('predictBtn');
        predictBtn.disabled = true;
        predictBtn.innerHTML = '<span class="loading"></span> Predicting...';
        
        try {
            this.visualization.addLog('Making prediction...');
            
            // Use recent data for prediction
            const lookback = this.model.config.lookback;
            const lastSequence = this.trainingData.X[this.trainingData.X.length - 1];
            console.log('[App] Prediction input shape:', [lastSequence.length, lastSequence[0]?.length]);
            
            // Predict
            console.log('[App] Calling model predict...');
            const normalizedPrediction = this.model.predict(lastSequence);
            console.log('[App] Normalized prediction:', normalizedPrediction);
            
            // Denormalize
            console.log('[App] Denormalizing prediction...');
            const prediction = this.dataProcessor.denormalizePrediction(
                normalizedPrediction,
                this.targetSymbol
            );
            console.log('[App] Final prediction:', prediction);
            
            // Save prediction result
            this.lastPrediction = prediction;
            
            this.visualization.addLog(
                `Prediction complete: ${prediction.length} days`,
                'success'
            );
            
            // Show prediction details
            prediction.forEach((pred, i) => {
                this.visualization.addLog(
                    `Day ${i + 1}: O=${pred.open.toFixed(2)}, H=${pred.high.toFixed(2)}, ` +
                    `L=${pred.low.toFixed(2)}, C=${pred.close.toFixed(2)}`
                );
            });
            
            // Update chart
            this.visualization.drawPrediction(
                this.multiSourceData[this.targetSymbol],
                prediction
            );
            
            // Switch to prediction tab
            this.switchTab('prediction');
            console.log('[App] ✅ Prediction complete');
            console.timeEnd('[App] predict duration');
            
        } catch (error) {
            console.error('[App] ❌ Prediction failed:', error);
            console.error('[App] Error stack:', error.stack);
            this.visualization.addLog(`Prediction failed: ${error.message}`, 'error');
            console.error(error);
        } finally {
            predictBtn.disabled = false;
            predictBtn.innerHTML = '🔮 Predict';
        }
    }

    /**
     * Switch tab
     */
    switchTab(tab) {
        console.log('[App] switchTab() Switching to:', tab);
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tab);
        });

        if (!this.multiSourceData) {
            console.log('[App] switchTab: No data, skipping');
            return;
        }

        switch (tab) {
            case 'candlestick':
                this.visualization.drawCandlestick(
                    this.multiSourceData[this.targetSymbol],
                    this.targetSymbol
                );
                break;
            case 'comparison':
                this.visualization.drawComparison(this.multiSourceData);
                break;
            case 'prediction':
                if (this.lastPrediction && this.model && this.model.trained) {
                    // Show existing prediction results
                    this.visualization.drawPrediction(
                        this.multiSourceData[this.targetSymbol],
                        this.lastPrediction
                    );
                } else {
                    this.visualization.addLog('Please make a prediction first', 'warning');
                }
                break;
        }
    }

    /**
     * Get model configuration
     */
    getModelConfig() {
        const lookback = parseInt(document.getElementById('lookbackDays').value);
        const forecast = parseInt(document.getElementById('forecastDays').value);
        const hiddenDim = parseInt(document.getElementById('ssmDim').value);
        const stateDim = parseInt(document.getElementById('stateDim').value);
        const numHeads = parseInt(document.getElementById('numHeads').value);
        const epochs = parseInt(document.getElementById('epochs').value);
        const learningRate = parseFloat(document.getElementById('learningRate').value);
        
        return {
            inputDim: this.trainingData.featureCount,
            hiddenDim: hiddenDim,
            stateDim: stateDim,
            numHeads: numHeads,
            numLayers: 2,
            outputDim: 4,
            lookback: lookback,
            forecast: forecast,
            numSources: this.selectedSources.length,
            dropoutRate: 0.1,
            learningRate: learningRate,
            epochs: epochs
        };
    }

    /**
     * Delay function
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

export default App;
