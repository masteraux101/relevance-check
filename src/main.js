/**
 * Main entry file
 */

import * as tf from '@tensorflow/tfjs';
import { Chart, registerables } from 'chart.js';
import 'chartjs-adapter-date-fns';

// Register Chart.js components
Chart.register(...registerables);

// Export tf globally for other modules
window.tf = tf;
window.Chart = Chart;

// Import app module
import { App } from './app.js';

// Start application
document.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
});
