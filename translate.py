#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Translate Chinese comments to English in JS files
"""

import os
import re

# Chinese to English comment translations
translations = {
    # Comments
    '/**\n     * 主应用模块 - 整合所有组件\n     */': '/**\n     * Main application module - integrates all components\n     */',
    '/**\n     * 初始化应用\n     */': '/**\n     * Initialize application\n     */',
    '/**\n     * 等待TensorFlow.js加载\n     */': '/**\n     * Wait for TensorFlow.js to load\n     */',
    '/**\n     * 绑定事件\n     */': '/**\n     * Bind events\n     */',
    '/**\n     * 数据源模式切换\n     */': '/**\n     * Data source mode switch\n     */',
    '/**\n     * 从CSV数据加载\n     */': '/**\n     * Load data from CSV\n     */',
    '/**\n     * 从模拟数据加载\n     */': '/**\n     * Load data from mock\n     */',
    '/**\n     * 训练模型\n     */': '/**\n     * Train model\n     */',
    '/**\n     * 计算评估指标\n     */': '/**\n     * Calculate metrics\n     */',
    '/**\n     * 进行预测\n     */': '/**\n     * Make prediction\n     */',
    '/**\n     * 切换标签页\n     */': '/**\n     * Switch tab\n     */',
    '/**\n     * 获取模型配置\n     */': '/**\n     * Get model config\n     */',
    '/**\n     * 延迟函数\n     */': '/**\n     * Delay function\n     */',
    
    # Console logs
    'console.log(\'[App] 🚀 初始化应用...\');': 'console.log(\'[App] 🚀 Initializing app...\');',
    'console.log(\'[App] 配置:\', { targetSymbol: this.targetSymbol, selectedSources: this.selectedSources });': 'console.log(\'[App] Config:\', { targetSymbol: this.targetSymbol, selectedSources: this.selectedSources });',
    'console.log(\'[App] init() 开始初始化...\');': 'console.log(\'[App] init() Starting initialization...\');',
    'console.log(\'[App] ✅ TensorFlow.js 加载完成\');': 'console.log(\'[App] ✅ TensorFlow.js loaded\');',
    'console.log(\'[App] ✅ 应用初始化完成\');': 'console.log(\'[App] ✅ App initialization complete\');',
    
    # Inline comments
    '// 数据源模式切换': '// Data source mode switch',
    '// CSV 文件上传 - 目标文件': '// CSV file upload - target file',
    '// CSV 文件上传 - 关联文件': '// CSV file upload - related files',
    '// 拖放上传功能': '// Drag and drop upload',
    '// 加载数据按钮': '// Load data button',
    '// 训练按钮': '// Train button',
    '// 预测按钮': '// Predict button',
    '// 标签页切换': '// Tab switch',
    '// 数据源复选框': '// Data source checkboxes',
    '// 目标选择': '// Target selection',
    '// 阻止默认拖放行为': '// Prevent default drag and drop',
    '// 拖入时高亮': '// Highlight on drag enter',
    '// 拖出时取消高亮': '// Remove highlight on drag leave',
    '// 处理拖放的文件': '// Handle dropped files',
    '// 绑定删除事件': '// Bind delete events',
    '// CSV 模式': '// CSV mode',
    '// 模拟数据模式': '// Mock data mode',
    '// 清除已上传的CSV数据': '// Clear uploaded CSV data',
    '// 启用训练按钮': '// Enable train button',
    '// 检查是否有上传的目标文件': '// Check if target file uploaded',
    '// 找到目标符号': '// Find target symbol',
    '// 如果没有明确的目标文件，使用第一个上传的文件': '// If no explicit target file, use first uploaded',
    '// 第一个设为目标，其余设为关联': '// First as target, rest as related',
    '// 已有目标文件，全部作为关联文件': '// Already has target file, all as related',
    '// 构建多源数据': '// Build multi-source data',
    '// 验证数据量': '// Validate data amount',
    '// 显示K线图': '// Show candlestick chart',
    '// 准备训练数据': '// Prepare training data',
}

def translate_file(filepath):
    """Translate a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply all translations
        for chinese, english in translations.items():
            content = content.replace(chinese, english)
        
        # If content changed, write back
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Translated: {filepath}")
            return True
        else:
            print(f"⏭️  No changes: {filepath}")
            return False
    except Exception as e:
        print(f"❌ Error in {filepath}: {e}")
        return False

def main():
    """Main function"""
    src_dir = '/Users/masteraux1/code/thoughts/relevance-check/src'
    
    # Find all JS files
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.js'):
                filepath = os.path.join(root, file)
                translate_file(filepath)
    
    # Also translate index.html
    html_file = '/Users/masteraux1/code/thoughts/relevance-check/index.html'
    if os.path.exists(html_file):
        translate_file(html_file)

if __name__ == '__main__':
    main()
