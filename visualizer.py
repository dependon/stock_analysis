import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import numpy as np
import matplotlib as mpl

class StockVisualizer:
    def __init__(self):
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        plt.style.use('ggplot')  # 使用matplotlib内置的ggplot样式，它提供了类似seaborn的美观效果
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'positive': '#2ecc71',
            'negative': '#e74c3c'
        }

    def plot_stock_data(self, df, title='股票走势分析'):
        """绘制股票K线图和主要技术指标"""
        # 创建子图
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), height_ratios=[3, 1, 1])
        fig.suptitle(title, fontsize=18)

        # 设置日期格式
        ax1.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.get_xticklabels(), rotation=45)

        # 绘制K线图和均线
        ax1.plot(df['date'], df['close'], label='收盘价', color=self.colors['primary'])
        ax1.plot(df['date'], df['MA5'], label='MA5', color=self.colors['secondary'])
        ax1.plot(df['date'], df['MA20'], label='MA20', color='green')
        ax1.set_title('价格走势')
        ax1.legend()
        ax1.grid(True)

        # 设置MACD图日期格式
        ax2.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax2.get_xticklabels(), rotation=45)

        # 绘制MACD
        ax2.plot(df['date'], df['MACD'], label='MACD', color=self.colors['primary'])
        ax2.plot(df['date'], df['Signal'], label='Signal', color=self.colors['secondary'])
        ax2.set_title('MACD指标')
        ax2.legend()
        ax2.grid(True)

        # 设置RSI图日期格式
        ax3.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax3.get_xticklabels(), rotation=45)

        # 绘制RSI
        ax3.plot(df['date'], df['RSI'], label='RSI', color=self.colors['primary'])
        ax3.axhline(y=70, color='r', linestyle='--')
        ax3.axhline(y=30, color='g', linestyle='--')
        ax3.set_title('RSI指标')
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        return fig

    def plot_prediction(self, actual_values, predicted_values, dates=None, title='趋势预测'):
        """绘制实际值和预测值的对比图"""
        plt.figure(figsize=(30, 15))
        x_values = dates if dates is not None else np.arange(len(predicted_values))
        
        # 设置全局字体大小
        plt.rcParams.update({'font.size': 14})
        
        # 绘制当前实际值（最后一个已知价格点）
        plt.scatter(x_values[0], actual_values[0], color=self.colors['primary'], s=150, label='当前价格')
        
        # 绘制预测值
        plt.plot(x_values, predicted_values, label='预测趋势', color=self.colors['secondary'], linestyle='--')
        
        plt.title(title)
        plt.gca().xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mpl.dates.MonthLocator())
        plt.xticks(rotation=45)
        plt.xlabel('日期')
        plt.ylabel('价格')
        plt.legend()
        plt.grid(True)
        
        # 添加当前价格标注
        plt.annotate(f'当前价格: {actual_values[0]:.2f}', 
                     xy=(x_values[0], actual_values[0]), 
                     xytext=(10, 10), 
                     textcoords='offset points')
        
        plt.tight_layout()
        return plt.gcf()

    def save_plots(self, fig, filename):
        """保存图表到文件"""
        fig.savefig(filename)
        plt.close(fig)

    def show_plots(self):
        """显示所有图表"""
        plt.show()