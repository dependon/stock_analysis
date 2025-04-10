from PyQt5.QtCore import QThread, pyqtSignal
from datetime import datetime, timedelta
import os

class AnalysisWorker(QThread):
    progress_updated = pyqtSignal(int)
    analysis_completed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, fetcher, processor, analyzer, visualizer, stock_code, look_back):
        super().__init__()
        self.fetcher = fetcher
        self.processor = processor
        self.analyzer = analyzer
        self.visualizer = visualizer
        self.stock_code = stock_code
        self.look_back = look_back
        
    def run(self):
        try:
            # 创建保存目录
            save_dir = 'analysis_results'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            # 生成时间戳
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            self.progress_updated.emit(10)
            
            # 获取数据
            sohu_data = self.fetcher.fetch_sohu_data(self.stock_code)
            sohu_df = self.processor.process_sohu_data(sohu_data)
            
            if sohu_df is None or len(sohu_df) == 0:
                self.error_occurred.emit('获取数据失败，请检查股票代码是否正确')
                return
                
            self.progress_updated.emit(30)
            
            # 数据处理
            df = self.processor.clean_data(sohu_df)
            if df is None or len(df) == 0:
                self.error_occurred.emit('数据处理失败')
                return
                
            self.progress_updated.emit(50)
            
            # 计算技术指标
            df = self.analyzer.calculate_technical_indicators(df)
            
            # 准备训练数据
            X, y = self.analyzer.prepare_data(df, self.look_back)
            
            if len(X) == 0:
                self.error_occurred.emit('数据量不足，无法进行预测分析')
                return
                
            self.progress_updated.emit(70)
            
            # 训练模型
            train_size = int(len(X) * 0.8)
            X_train = X[:train_size]
            y_train = y[:train_size]
            self.analyzer.train_model(X_train, y_train)
            
            self.progress_updated.emit(80)
            
            # 生成并保存技术指标图
            analysis_fig = self.visualizer.plot_stock_data(df, f'股票{self.stock_code}技术指标分析')
            analysis_filename = os.path.join(save_dir, f'stock_{self.stock_code}_analysis_{timestamp}.png')
            self.visualizer.save_plots(analysis_fig, analysis_filename)
            
            self.progress_updated.emit(90)
            
            # 生成并保存预测图
            last_sequence = self.analyzer.X[-1:]
            year_prices = self.analyzer.predict_year(last_sequence)
            current_price = df['close'].iloc[-1]
            
            # 预测各个时间段的价格
            future_pred = self.analyzer.predict(last_sequence)
            future_price = self.analyzer.scaler.inverse_transform(future_pred.reshape(-1, 1))[0][0]
            
            week_prices = self.analyzer.predict_week(last_sequence)
            week_final_price = week_prices[-1][0]
            
            month_prices = self.analyzer.predict_month(last_sequence)
            month_final_price = month_prices[-1][0]
            
            year_final_price = year_prices[-1][0]
            
            prediction_fig = self.visualizer.plot_prediction(
                [current_price],
                year_prices.flatten(),
                [df['date'].iloc[-1] + timedelta(days=x) for x in range(1, 366)],
                f'股票{self.stock_code}未来一年趋势预测'
            )
            prediction_filename = os.path.join(save_dir, f'stock_{self.stock_code}_prediction_{timestamp}.png')
            self.visualizer.save_plots(prediction_fig, prediction_filename)
            
            self.progress_updated.emit(100)
            
            # 保存预测结果到JSON文件
            prediction_results = {
                'stock_code': self.stock_code,
                'timestamp': timestamp,
                'current_price': float(current_price),
                'future_price': float(future_price),
                'week_final_price': float(week_final_price),
                'month_final_price': float(month_final_price),
                'year_final_price': float(year_final_price),
                'prediction_changes': {
                    'tomorrow': float((future_price/current_price - 1) * 100),
                    'week': float((week_final_price/current_price - 1) * 100),
                    'month': float((month_final_price/current_price - 1) * 100),
                    'year': float((year_final_price/current_price - 1) * 100)
                }
            }
            
            # 保存预测结果到JSON文件
            import json
            prediction_json = os.path.join(save_dir, f'stock_{self.stock_code}_prediction_{timestamp}.json')
            with open(prediction_json, 'w', encoding='utf-8') as f:
                json.dump(prediction_results, f, ensure_ascii=False, indent=4)
            
            # 返回分析结果
            results = {
                'df': df,
                'current_price': current_price,
                'future_price': future_price,
                'week_final_price': week_final_price,
                'month_final_price': month_final_price,
                'year_final_price': year_final_price,
                'analysis_filename': analysis_filename,
                'prediction_filename': prediction_filename
            }
            
            self.analysis_completed.emit(results)
            
        except Exception as e:
            self.error_occurred.emit(str(e))