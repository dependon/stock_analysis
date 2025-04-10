import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLineEdit, QPushButton, QLabel,
                             QTabWidget, QMessageBox, QProgressBar)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from stock_analysis import StockDataFetcher, StockAnalyzer
from data_processor import DataProcessor
from visualizer import StockVisualizer
from datetime import timedelta
import numpy as np
from analysis_worker import AnalysisWorker

class StockAnalysisGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('股票分析系统')
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建主窗口部件和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # 创建输入区域
        input_widget = QWidget()
        input_layout = QHBoxLayout(input_widget)
        
        self.stock_code_input = QLineEdit()
        self.stock_code_input.setPlaceholderText('请输入股票代码（例如：000001）')
        self.analyze_button = QPushButton('分析')
        self.analyze_button.clicked.connect(self.analyze_stock)
        
        input_layout.addWidget(QLabel('股票代码：'))
        input_layout.addWidget(self.stock_code_input)
        
        # 添加训练天数输入框
        self.look_back_input = QLineEdit()
        self.look_back_input.setPlaceholderText('训练天数(30-1825)')
        self.look_back_input.setText('365')
        input_layout.addWidget(QLabel('训练天数：'))
        input_layout.addWidget(self.look_back_input)
        
        input_layout.addWidget(self.analyze_button)
        
        layout.addWidget(input_widget)
        
        # 添加进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        
        # 技术指标分析标签页
        self.analysis_tab = QWidget()
        self.analysis_layout = QVBoxLayout(self.analysis_tab)
        self.analysis_canvas = None
        
        # 预测结果标签页
        self.prediction_tab = QWidget()
        self.prediction_layout = QVBoxLayout(self.prediction_tab)
        self.prediction_canvas = None
        
        # 预测数据标签页
        self.data_tab = QWidget()
        self.data_layout = QVBoxLayout(self.data_tab)
        self.prediction_label = QLabel('请先进行分析...')
        self.data_layout.addWidget(self.prediction_label)
        
        self.tab_widget.addTab(self.analysis_tab, '技术指标分析')
        self.tab_widget.addTab(self.prediction_tab, '趋势预测')
        self.tab_widget.addTab(self.data_tab, '预测数据')
        
        layout.addWidget(self.tab_widget)
        
        # 添加免责声明
        disclaimer_label = QLabel('免责声明：本预测纯属娱乐，请不要用该预测用于投资等危险事宜，股市有风险，本软件不对任何结果负责！')
        disclaimer_label.setStyleSheet('color: red; font-weight: bold; padding: 10px;')
        disclaimer_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(disclaimer_label)
        
        # 初始化分析器
        self.fetcher = StockDataFetcher()
        self.processor = DataProcessor()
        self.analyzer = StockAnalyzer()
        self.visualizer = StockVisualizer()
    


    def analyze_stock(self):
        stock_code = self.stock_code_input.text().strip()
        if not stock_code:
            QMessageBox.warning(self, '警告', '请输入股票代码')
            return
        
        # 验证训练天数
        try:
            look_back = int(self.look_back_input.text())
            if look_back < 30 or look_back > 1825:
                QMessageBox.warning(self, '警告', '训练天数必须在30-1825天之间')
                return
        except ValueError:
            QMessageBox.warning(self, '警告', '请输入有效的训练天数')
            return
        
        # 禁用分析按钮
        self.analyze_button.setEnabled(False)
        
        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 创建并启动工作线程
        self.worker = AnalysisWorker(
            self.fetcher,
            self.processor,
            self.analyzer,
            self.visualizer,
            stock_code,
            look_back  # 传递训练天数参数
        )
        
        # 连接信号
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.analysis_completed.connect(self.handle_analysis_completed)
        self.worker.error_occurred.connect(self.handle_error)
        
        # 启动工作线程
        self.worker.start()
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def handle_analysis_completed(self, results):
        # 更新技术指标图
        if self.analysis_canvas is not None:
            self.analysis_layout.removeWidget(self.analysis_canvas)
            self.analysis_canvas.deleteLater()
        
        analysis_fig = plt.figure()
        img = plt.imread(results['analysis_filename'])
        plt.imshow(img)
        plt.axis('off')
        self.analysis_canvas = FigureCanvas(analysis_fig)
        self.analysis_layout.addWidget(self.analysis_canvas)
        
        # 更新预测图
        if self.prediction_canvas is not None:
            self.prediction_layout.removeWidget(self.prediction_canvas)
            self.prediction_canvas.deleteLater()
        
        prediction_fig = plt.figure()
        img = plt.imread(results['prediction_filename'])
        plt.imshow(img)
        plt.axis('off')
        self.prediction_canvas = FigureCanvas(prediction_fig)
        self.prediction_layout.addWidget(self.prediction_canvas)
        
        # 更新预测数据
        current_price = results['current_price']
        future_price = results['future_price']
        week_final_price = results['week_final_price']
        month_final_price = results['month_final_price']
        year_final_price = results['year_final_price']
        
        prediction_text = f'''
        当前价格: {current_price:.2f}


        明日预测:
        预测价格: {future_price:.2f}
        预计{'上涨' if future_price > current_price else '下跌'}: \
        {abs((future_price/current_price - 1) * 100):.2f}%


        一周预测:
        预测价格: {week_final_price:.2f}
        预计{'上涨' if week_final_price > current_price else '下跌'}: \
        {abs((week_final_price/current_price - 1) * 100):.2f}%


        一月预测:
        预测价格: {month_final_price:.2f}
        预计{'上涨' if month_final_price > current_price else '下跌'}: \
        {abs((month_final_price/current_price - 1) * 100):.2f}%


        一年预测:
        预测价格: {year_final_price:.2f}
        预计{'上涨' if year_final_price > current_price else '下跌'}: \
        {abs((year_final_price/current_price - 1) * 100):.2f}%
        '''
        
        self.prediction_label.setText(prediction_text)
        
        # 重置UI状态
        self.analyze_button.setEnabled(True)
        self.progress_bar.setVisible(False)
    
    def handle_error(self, error_message):
        QMessageBox.critical(self, '错误', error_message)
        self.analyze_button.setEnabled(True)
        self.progress_bar.setVisible(False)
    
    def update_analysis_chart(self, df, stock_code):
        # 清除旧图表
        if self.analysis_canvas is not None:
            self.analysis_layout.removeWidget(self.analysis_canvas)
            self.analysis_canvas.deleteLater()
        
        # 创建新图表
        fig = self.visualizer.plot_stock_data(df, f'股票{stock_code}技术指标分析')
        self.analysis_canvas = FigureCanvas(fig)
        self.analysis_layout.addWidget(self.analysis_canvas)
    
    def update_prediction_chart(self, df, stock_code):
        # 清除旧图表
        if self.prediction_canvas is not None:
            self.prediction_layout.removeWidget(self.prediction_canvas)
            self.prediction_canvas.deleteLater()
        
        # 获取预测数据
        last_sequence = self.analyzer.X[-1:]
        year_prices = self.analyzer.predict_year(last_sequence)
        
        # 生成日期序列
        last_date = df['date'].iloc[-1]
        future_dates = [last_date + timedelta(days=x) for x in range(1, 366)]
        
        # 获取当前价格
        current_price = df['close'].iloc[-1]
        actual_values = np.array([current_price])
        
        # 创建新图表
        fig = self.visualizer.plot_prediction(actual_values, year_prices.flatten(),
                                            future_dates, f'股票{stock_code}未来一年趋势预测')
        self.prediction_canvas = FigureCanvas(fig)
        self.prediction_layout.addWidget(self.prediction_canvas)
    
    def update_prediction_data(self, df):
        current_price = df['close'].iloc[-1]
        last_sequence = self.analyzer.X[-1:]
        
        # 预测各个时间段的价格
        future_pred = self.analyzer.predict(last_sequence)
        future_price = self.analyzer.scaler.inverse_transform(future_pred.reshape(-1, 1))[0][0]
        
        week_prices = self.analyzer.predict_week(last_sequence)
        week_final_price = week_prices[-1][0]
        
        month_prices = self.analyzer.predict_month(last_sequence)
        month_final_price = month_prices[-1][0]
        
        year_prices = self.analyzer.predict_year(last_sequence)
        year_final_price = year_prices[-1][0]
        
        # 更新预测数据显示
        prediction_text = f'''
        当前价格: {current_price:.2f}


        明日预测:
        预测价格: {future_price:.2f}
        预计{'上涨' if future_price > current_price else '下跌'}: \
        {abs((future_price/current_price - 1) * 100):.2f}%


        一周预测:
        预测价格: {week_final_price:.2f}
        预计{'上涨' if week_final_price > current_price else '下跌'}: \
        {abs((week_final_price/current_price - 1) * 100):.2f}%


        一月预测:
        预测价格: {month_final_price:.2f}
        预计{'上涨' if month_final_price > current_price else '下跌'}: \
        {abs((month_final_price/current_price - 1) * 100):.2f}%


        一年预测:
        预测价格: {year_final_price:.2f}
        预计{'上涨' if year_final_price > current_price else '下跌'}: \
        {abs((year_final_price/current_price - 1) * 100):.2f}%
        '''
        
        self.prediction_label.setText(prediction_text)

def main():
    app = QApplication(sys.argv)
    window = StockAnalysisGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()