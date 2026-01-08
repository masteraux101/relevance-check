/**
 * Visualization module - Candlestick charts and prediction results display
 */

export class Visualization {
    constructor() {
        console.log('[Visualization] 📊 Initializing visualization module');
        this.mainChart = null;
        this.currentTab = 'candlestick';
        this.chartColors = {
            up: '#10b981',
            down: '#ef4444',
            line: '#6366f1',
            prediction: '#f59e0b',
            volume: '#64748b'
        };
    }

    /**
     * Initialize chart
     */
    init() {
        console.log('[Visualization] init() Initializing chart');
        this.setupTabs();
    }

    /**
     * Setup tab switching
     */
    setupTabs() {
        const tabs = document.querySelectorAll('.tab-btn');
        tabs.forEach(tab => {
            tab.addEventListener('click', (e) => {
                tabs.forEach(t => t.classList.remove('active'));
                e.target.classList.add('active');
                this.currentTab = e.target.dataset.tab;
            });
        });
    }

    /**
     * 绘制K线图
     * @param {Array} data - K线数据
     * @param {string} symbol - 股票代码
     * @param {Array} predictions - 预测数据（可选）
     */
    drawCandlestick(data, symbol, predictions = null) {
        console.log('[Visualization] drawCandlestick():', { symbol, dataLength: data?.length, hasPredictions: !!predictions });
        const ctx = document.getElementById('mainChart').getContext('2d');
        
        // 销毁旧图表
        if (this.mainChart) {
            this.mainChart.destroy();
        }

        // 准备蜡烛图数据
        const candleData = data.map(d => ({
            x: new Date(d.date),
            o: d.open,
            h: d.high,
            l: d.low,
            c: d.close
        }));

        // 计算价格范围
        const prices = data.flatMap(d => [d.high, d.low]);
        const minPrice = Math.min(...prices);
        const maxPrice = Math.max(...prices);
        const padding = (maxPrice - minPrice) * 0.1;

        // 准备数据集
        const datasets = [
            {
                label: `${symbol} K线`,
                data: candleData,
                type: 'candlestick',
                color: {
                    up: this.chartColors.up,
                    down: this.chartColors.down,
                    unchanged: this.chartColors.line
                },
                borderColor: {
                    up: this.chartColors.up,
                    down: this.chartColors.down,
                    unchanged: this.chartColors.line
                }
            }
        ];

        // 添加预测数据
        if (predictions && predictions.length > 0) {
            const lastDate = new Date(data[data.length - 1].date);
            const predictionData = predictions.map((pred, i) => {
                const date = new Date(lastDate);
                date.setDate(date.getDate() + i + 1);
                return {
                    x: date,
                    o: pred.open,
                    h: pred.high,
                    l: pred.low,
                    c: pred.close
                };
            });

            datasets.push({
                label: '预测',
                data: predictionData,
                type: 'candlestick',
                color: {
                    up: this.chartColors.prediction,
                    down: '#ff6b6b',
                    unchanged: this.chartColors.prediction
                },
                borderColor: {
                    up: this.chartColors.prediction,
                    down: '#ff6b6b',
                    unchanged: this.chartColors.prediction
                }
            });
        }

        // 由于chartjs-chart-financial可能不可用，使用备选方案
        this.drawCandlestickFallback(data, symbol, predictions);
    }

    /**
     * 备选K线图绘制方法（使用折线图模拟）
     */
    drawCandlestickFallback(data, symbol, predictions = null) {
        const ctx = document.getElementById('mainChart').getContext('2d');
        
        if (this.mainChart) {
            this.mainChart.destroy();
        }

        // 准备数据
        const labels = data.map(d => d.date);
        const closeData = data.map(d => d.close);
        const highData = data.map(d => d.high);
        const lowData = data.map(d => d.low);

        const datasets = [
            {
                label: `${symbol} Close`,
                data: closeData,
                borderColor: this.chartColors.line,
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                fill: false,
                tension: 0.1,
                pointRadius: 0,
                borderWidth: 2
            },
            {
                label: 'High',
                data: highData,
                borderColor: this.chartColors.up,
                backgroundColor: 'transparent',
                fill: false,
                tension: 0.1,
                pointRadius: 0,
                borderWidth: 1,
                borderDash: [5, 5]
            },
            {
                label: 'Low',
                data: lowData,
                borderColor: this.chartColors.down,
                backgroundColor: 'transparent',
                fill: false,
                tension: 0.1,
                pointRadius: 0,
                borderWidth: 1,
                borderDash: [5, 5]
            }
        ];

        // 添加预测数据
        if (predictions && predictions.length > 0) {
            const lastDate = new Date(data[data.length - 1].date);
            const predLabels = predictions.map((_, i) => {
                const date = new Date(lastDate);
                date.setDate(date.getDate() + i + 1);
                return date.toISOString().split('T')[0];
            });
            
            labels.push(...predLabels);
            
            // 用null填充历史数据部分
            const predClose = [...Array(closeData.length - 1).fill(null), closeData[closeData.length - 1], ...predictions.map(p => p.close)];
            const predHigh = [...Array(highData.length - 1).fill(null), highData[highData.length - 1], ...predictions.map(p => p.high)];
            const predLow = [...Array(lowData.length - 1).fill(null), lowData[lowData.length - 1], ...predictions.map(p => p.low)];
            
            datasets.push({
                label: 'Predicted Close',
                data: predClose,
                borderColor: this.chartColors.prediction,
                backgroundColor: 'rgba(245, 158, 11, 0.1)',
                fill: false,
                tension: 0.1,
                pointRadius: 3,
                borderWidth: 2,
                borderDash: [3, 3]
            });
            
            datasets.push({
                label: 'Predicted High',
                data: predHigh,
                borderColor: '#fcd34d',
                backgroundColor: 'transparent',
                fill: false,
                tension: 0.1,
                pointRadius: 0,
                borderWidth: 1,
                borderDash: [2, 2]
            });
            
            datasets.push({
                label: 'Predicted Low',
                data: predLow,
                borderColor: '#fb923c',
                backgroundColor: 'transparent',
                fill: false,
                tension: 0.1,
                pointRadius: 0,
                borderWidth: 1,
                borderDash: [2, 2]
            });
        }

        this.mainChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#f1f5f9',
                            usePointStyle: true
                        }
                    },
                    tooltip: {
                        backgroundColor: '#1e293b',
                        titleColor: '#f1f5f9',
                        bodyColor: '#94a3b8',
                        borderColor: '#334155',
                        borderWidth: 1
                    }
                },
                scales: {
                    x: {
                        display: true,
                        ticks: {
                            color: '#94a3b8',
                            maxRotation: 45,
                            maxTicksLimit: 10
                        },
                        grid: {
                            color: 'rgba(51, 65, 85, 0.5)'
                        }
                    },
                    y: {
                        display: true,
                        position: 'right',
                        ticks: {
                            color: '#94a3b8'
                        },
                        grid: {
                            color: 'rgba(51, 65, 85, 0.5)'
                        }
                    }
                }
            }
        });
    }

    /**
     * 绘制多源对比图
     * @param {Object} multiSourceData - 多源数据
     */
    drawComparison(multiSourceData) {
        const ctx = document.getElementById('mainChart').getContext('2d');
        
        if (this.mainChart) {
            this.mainChart.destroy();
        }

        const colors = [
            '#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'
        ];

        const symbols = Object.keys(multiSourceData);
        const labels = multiSourceData[symbols[0]].map(d => d.date);

        // 归一化数据用于对比
        const datasets = symbols.map((symbol, i) => {
            const data = multiSourceData[symbol];
            const closes = data.map(d => d.close);
            const min = Math.min(...closes);
            const max = Math.max(...closes);
            const normalized = closes.map(c => (c - min) / (max - min || 1));

            return {
                label: symbol,
                data: normalized,
                borderColor: colors[i % colors.length],
                backgroundColor: 'transparent',
                fill: false,
                tension: 0.1,
                pointRadius: 0,
                borderWidth: 2
            };
        });

        this.mainChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#f1f5f9',
                            usePointStyle: true
                        }
                    },
                    title: {
                        display: true,
                        text: 'Multi-source Normalized Comparison',
                        color: '#f1f5f9'
                    },
                    tooltip: {
                        backgroundColor: '#1e293b',
                        titleColor: '#f1f5f9',
                        bodyColor: '#94a3b8'
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: '#94a3b8',
                            maxTicksLimit: 10
                        },
                        grid: {
                            color: 'rgba(51, 65, 85, 0.5)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Normalized Price (0-1)',
                            color: '#94a3b8'
                        },
                        ticks: {
                            color: '#94a3b8'
                        },
                        grid: {
                            color: 'rgba(51, 65, 85, 0.5)'
                        }
                    }
                }
            }
        });
    }

    /**
     * 绘制预测结果图
     * @param {Array} historical - 历史数据
     * @param {Array} predictions - 预测数据
     * @param {Array} actual - 实际数据（可选，用于验证）
     */
    drawPrediction(historical, predictions, actual = null) {
        console.log('[Visualization] drawPrediction() 绘制预测图表...');
        console.log('[Visualization] 历史数据长度:', historical?.length);
        console.log('[Visualization] 预测数据:', predictions);
        
        const ctx = document.getElementById('mainChart').getContext('2d');
        
        if (this.mainChart) {
            this.mainChart.destroy();
        }

        // 只显示最近的历史数据
        const recentHistory = historical.slice(-30);
        const labels = recentHistory.map(d => d.date);
        
        // 添加预测日期
        const lastDate = new Date(historical[historical.length - 1].date);
        predictions.forEach((_, i) => {
            const date = new Date(lastDate);
            date.setDate(date.getDate() + i + 1);
            labels.push(date.toISOString().split('T')[0]);
        });

        const historicalClose = recentHistory.map(d => d.close);
        const predictionClose = [...Array(recentHistory.length - 1).fill(null), 
                                  historicalClose[historicalClose.length - 1],
                                  ...predictions.map(p => p.close)];

        const datasets = [
            {
                label: 'Historical Close',
                data: [...historicalClose, ...Array(predictions.length).fill(null)],
                borderColor: this.chartColors.line,
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                fill: true,
                tension: 0.1,
                pointRadius: 0,
                borderWidth: 2
            },
            {
                label: 'Predicted Close',
                data: predictionClose,
                borderColor: this.chartColors.prediction,
                backgroundColor: 'rgba(245, 158, 11, 0.2)',
                fill: true,
                tension: 0.1,
                pointRadius: 4,
                pointBackgroundColor: this.chartColors.prediction,
                borderWidth: 2,
                borderDash: [5, 5]
            }
        ];

        // 如果有实际数据，添加对比
        if (actual && actual.length > 0) {
            const actualData = [...Array(recentHistory.length - 1).fill(null),
                                historicalClose[historicalClose.length - 1],
                                ...actual.map(a => a.close)];
            datasets.push({
                label: 'Actual Close',
                data: actualData,
                borderColor: this.chartColors.up,
                backgroundColor: 'transparent',
                fill: false,
                tension: 0.1,
                pointRadius: 4,
                pointBackgroundColor: this.chartColors.up,
                borderWidth: 2
            });
        }

        // 添加预测区间
        const upperBound = [...Array(recentHistory.length).fill(null),
                           ...predictions.map(p => p.high)];
        const lowerBound = [...Array(recentHistory.length).fill(null),
                           ...predictions.map(p => p.low)];

        datasets.push({
            label: 'Prediction Range (High)',
            data: upperBound,
            borderColor: 'rgba(245, 158, 11, 0.5)',
            backgroundColor: 'transparent',
            fill: false,
            tension: 0.1,
            pointRadius: 0,
            borderWidth: 1,
            borderDash: [2, 2]
        });

        datasets.push({
            label: 'Prediction Range (Low)',
            data: lowerBound,
            borderColor: 'rgba(245, 158, 11, 0.5)',
            backgroundColor: 'rgba(245, 158, 11, 0.1)',
            fill: '-1',
            tension: 0.1,
            pointRadius: 0,
            borderWidth: 1,
            borderDash: [2, 2]
        });

        this.mainChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#f1f5f9',
                            usePointStyle: true
                        }
                    },
                    title: {
                        display: true,
                        text: 'Price Prediction Results',
                        color: '#f1f5f9',
                        font: {
                            size: 14
                        }
                    },
                    annotation: {
                        annotations: {
                            predictionStart: {
                                type: 'line',
                                xMin: recentHistory.length - 1,
                                xMax: recentHistory.length - 1,
                                borderColor: 'rgba(255, 255, 255, 0.3)',
                                borderWidth: 2,
                                borderDash: [10, 5],
                                label: {
                                    display: true,
                                    content: 'Prediction Start',
                                    position: 'start'
                                }
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: '#94a3b8',
                            maxTicksLimit: 10
                        },
                        grid: {
                            color: 'rgba(51, 65, 85, 0.5)'
                        }
                    },
                    y: {
                        position: 'right',
                        ticks: {
                            color: '#94a3b8'
                        },
                        grid: {
                            color: 'rgba(51, 65, 85, 0.5)'
                        }
                    }
                }
            }
        });
    }

    /**
     * 更新注意力权重可视化
     * @param {Array} sources - 数据源名称
     * @param {Array} weights - 注意力权重
     */
    updateAttentionVisualization(sources, weights) {
        const container = document.getElementById('attentionViz');
        container.innerHTML = '';

        sources.forEach((source, i) => {
            const weight = weights[i];
            const percentage = (weight * 100).toFixed(1);
            
            const barDiv = document.createElement('div');
            barDiv.className = 'attention-bar';
            barDiv.innerHTML = `
                <div class="attention-bar-label">
                    <span>${source}</span>
                    <span>${percentage}%</span>
                </div>
                <div class="attention-bar-track">
                    <div class="attention-bar-fill" style="width: ${percentage}%"></div>
                </div>
            `;
            container.appendChild(barDiv);
        });
    }

    /**
     * 更新指标显示
     */
    updateMetrics(metrics) {
        console.log('[Visualization] updateMetrics():', metrics);
        if (metrics.trainLoss !== undefined) {
            document.getElementById('trainLoss').textContent = 
                metrics.trainLoss.toFixed(6);
        }
        if (metrics.valLoss !== undefined) {
            document.getElementById('valLoss').textContent = 
                metrics.valLoss.toFixed(6);
        }
        if (metrics.dirAccuracy !== undefined) {
            document.getElementById('dirAccuracy').textContent = 
                (metrics.dirAccuracy * 100).toFixed(1) + '%';
        }
        if (metrics.correlation !== undefined) {
            document.getElementById('correlation').textContent = 
                metrics.correlation.toFixed(4);
        }
    }

    /**
     * 添加日志
     */
    addLog(message, type = 'info') {
        console.log(`[Visualization] addLog(${type}):`, message);
        const container = document.getElementById('logContainer');
        const entry = document.createElement('p');
        entry.className = `log-entry ${type}`;
        entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        container.appendChild(entry);
        container.scrollTop = container.scrollHeight;
    }

    /**
     * 清除日志
     */
    clearLogs() {
        const container = document.getElementById('logContainer');
        container.innerHTML = '';
    }

    /**
     * 销毁图表
     */
    destroy() {
        if (this.mainChart) {
            this.mainChart.destroy();
            this.mainChart = null;
        }
    }
}

export default Visualization;
