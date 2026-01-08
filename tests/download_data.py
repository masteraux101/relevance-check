#!/usr/bin/env python3
"""
下载 yfinance 数据的示例脚本

使用方法:
1. 安装 yfinance: pip install yfinance
2. 运行脚本: python download_data.py

这会在 tests/data/ 目录下生成 CSV 文件
"""

import os
import yfinance as yf
from datetime import datetime, timedelta

# 配置
# 注意: VIX 需要使用 ^VIX 符号，但文件名保存为 VIX.csv
SYMBOLS = {
    'SPY': 'SPY',
    '^VIX': 'VIX',      # VIX 恐慌指数
    'TLT': 'TLT',
    'GLD': 'GLD',
    'QQQ': 'QQQ'
}
OUTPUT_DIR = 'data'
DAYS = 365  # 下载过去一年的数据

def download_data():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 计算日期范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=DAYS)
    
    print(f"下载日期范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
    print(f"下载标的: {', '.join(SYMBOLS.values())}")
    print()
    
    for ticker_symbol, file_name in SYMBOLS.items():
        try:
            print(f"正在下载 {file_name} ({ticker_symbol})...")
            
            # 使用 yfinance 下载数据
            ticker = yf.Ticker(ticker_symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                print(f"  ⚠️ {file_name} 没有数据")
                continue
            
            # 重置索引使 Date 成为列
            df = df.reset_index()
            
            # 只保留需要的列
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            # 格式化日期
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            
            # 添加 Adj Close 列（与 Close 相同，简化处理）
            df['Adj Close'] = df['Close']
            
            # 重新排列列顺序
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
            
            # 保存到 CSV (使用 file_name 而不是 ticker_symbol)
            output_file = os.path.join(OUTPUT_DIR, f'{file_name}.csv')
            df.to_csv(output_file, index=False)
            
            print(f"  ✅ {file_name}: {len(df)} 行数据已保存到 {output_file}")
            print(f"     日期范围: {df['Date'].iloc[0]} 至 {df['Date'].iloc[-1]}")
            print(f"     价格范围: {df['Low'].min():.2f} - {df['High'].max():.2f}")
            print()
            
        except Exception as e:
            print(f"  ❌ 下载 {file_name} 失败: {e}")
            print()
    
    print("下载完成!")
    print(f"CSV 文件保存在: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == '__main__':
    download_data()
