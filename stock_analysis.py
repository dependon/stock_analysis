import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from data_processor import DataProcessor

class StockDataFetcher:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def fetch_sohu_data(self, stock_code):
        url = f'https://q.stock.sohu.com/hisHq?code=cn_{stock_code}&start=20140101&end={datetime.now().strftime("%Y%m%d")}'
        try:
            response = requests.get(url, headers=self.headers)
            return response.json()
        except Exception as e:
            print(f'获取搜狐数据失败: {e}')
            return None

    def fetch_qq_data(self, stock_code):
        # 根据股票代码判断是上海还是深圳的股票
        prefix = 'sh' if stock_code.startswith('6') else 'sz'
        url = f'https://data.gtimg.cn/flashdata/hushen/weekly/{prefix}{stock_code}.js'
        try:
            response = requests.get(url, headers=self.headers)
            return response.text
        except Exception as e:
            print(f'获取腾讯数据失败: {e}')
            return None

class StockAnalyzer:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def prepare_data(self, df, look_back=60):
        data = df['close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i-look_back:i, 0])
            y.append(scaled_data[i, 0])
        
        # 保存训练数据的引用
        self.X = np.array(X)
        self.y = np.array(y)
        return self.X, self.y

    def build_model(self, look_back):
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        return model

    def train_model(self, X, y):
        # 重塑数据以适应RandomForest（从3D转为2D）
        X_reshaped = X.reshape(X.shape[0], -1)
        self.model = self.build_model(X.shape[1])
        self.model.fit(X_reshaped, y)
        return self.model

    def predict(self, X):
        # 重塑输入数据
        X_reshaped = X.reshape(X.shape[0], -1)
        predictions = self.model.predict(X_reshaped)
        return predictions

    def get_prediction_dates(self, df, test_size):
        """获取预测日期序列"""
        # 获取最后test_size天的日期
        prediction_dates = df['date'].iloc[-test_size:].values
        return prediction_dates

    def predict_week(self, last_sequence):
        # 预测未来一周的价格
        week_predictions = []
        current_sequence = last_sequence.copy()
        
        # 逐天预测7天
        for _ in range(7):
            # 预测下一天
            pred = self.predict(current_sequence)
            week_predictions.append(pred[0])
            
            # 更新序列，移除最早的数据点，添加新预测的数据点
            current_sequence = current_sequence.reshape(-1)
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = pred[0]
            current_sequence = current_sequence.reshape(1, -1)
        
        # 转换预测结果
        week_predictions = np.array(week_predictions).reshape(-1, 1)
        return self.scaler.inverse_transform(week_predictions)

    def predict_month(self, last_sequence):
        # 预测未来一个月的价格
        month_predictions = []
        current_sequence = last_sequence.copy()
        
        # 逐天预测30天
        for _ in range(30):
            # 预测下一天
            pred = self.predict(current_sequence)
            month_predictions.append(pred[0])
            
            # 更新序列，移除最早的数据点，添加新预测的数据点
            current_sequence = current_sequence.reshape(-1)
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = pred[0]
            current_sequence = current_sequence.reshape(1, -1)
        
        # 转换预测结果
        month_predictions = np.array(month_predictions).reshape(-1, 1)
        return self.scaler.inverse_transform(month_predictions)

    def predict_year(self, last_sequence):
        # 预测未来一年的价格
        year_predictions = []
        current_sequence = last_sequence.copy()
        
        # 逐天预测365天
        for _ in range(365):
            # 预测下一天
            pred = self.predict(current_sequence)
            year_predictions.append(pred[0])
            
            # 更新序列，移除最早的数据点，添加新预测的数据点
            current_sequence = current_sequence.reshape(-1)
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = pred[0]
            current_sequence = current_sequence.reshape(1, -1)
        
        # 转换预测结果
        year_predictions = np.array(year_predictions).reshape(-1, 1)
        return self.scaler.inverse_transform(year_predictions)

    def calculate_technical_indicators(self, df):
        # 计算MA
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        
        # 计算MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # 计算RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df

def main():
    stock_code = input('请输入股票代码（例如：000001）：')
    
    # 获取数据
    print('正在获取股票数据...')
    fetcher = StockDataFetcher()
    processor = DataProcessor()
    
    # 尝试从搜狐获取数据
    sohu_data = fetcher.fetch_sohu_data(stock_code)
    sohu_df = processor.process_sohu_data(sohu_data)
    
    # 只使用搜狐数据源
    if sohu_df is None or len(sohu_df) == 0:
        print('搜狐数据源获取失败，请检查股票代码是否正确或稍后重试。')
        return
    df = sohu_df
    
    # 清理数据
    df = processor.clean_data(df)
    
    if df is None or len(df) == 0:
        print('未能获取到有效数据，请检查股票代码是否正确。')
        return
    
    # 数据分析
    print('正在进行数据分析...')
    analyzer = StockAnalyzer()
    df = analyzer.calculate_technical_indicators(df)
    
    # 准备训练数据
    look_back = 365  # 使用365天的数据预测下一天
    X, y = analyzer.prepare_data(df, look_back)
    
    if len(X) == 0:
        print('数据量不足，无法进行预测分析。')
        return
    
    # 划分训练集和测试集
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 训练模型
    print('正在训练预测模型...')
    model = analyzer.train_model(X_train, y_train)
    
    # 预测未来一年的趋势
    last_sequence = X[-1:]
    year_prices = analyzer.predict_year(last_sequence)
    
    # 生成未来一年的日期序列
    last_date = df['date'].iloc[-1]
    future_dates = [last_date + timedelta(days=x) for x in range(1, 366)]
    
    # 获取当前价格作为实际值
    current_price = df['close'].iloc[-1]
    actual_values = np.array([current_price])
    
    # 可视化
    print('正在生成分析图表...')
    from visualizer import StockVisualizer
    visualizer = StockVisualizer()
    
    # 绘制技术指标图
    fig1 = visualizer.plot_stock_data(df, f'股票{stock_code}技术指标分析')
    visualizer.save_plots(fig1, f'stock_{stock_code}_analysis.png')
    
    # 绘制预测结果图
    fig2 = visualizer.plot_prediction(actual_values, year_prices.flatten(), future_dates, f'股票{stock_code}未来一年趋势预测')
    visualizer.save_plots(fig2, f'stock_{stock_code}_prediction.png')
    
    print(f'分析完成！图表已保存为 stock_{stock_code}_analysis.png 和 stock_{stock_code}_prediction.png')
    
    # 预测未来趋势
    last_sequence = X[-1:]
    # 预测明天的价格
    future_pred = analyzer.predict(last_sequence)
    future_price = analyzer.scaler.inverse_transform(future_pred.reshape(-1, 1))[0][0]
    current_price = df['close'].iloc[-1]
    
    # 预测一周的价格趋势
    week_prices = analyzer.predict_week(last_sequence)
    week_final_price = week_prices[-1][0]
    
    # 预测一个月的价格趋势
    month_prices = analyzer.predict_month(last_sequence)
    month_final_price = month_prices[-1][0]
    
    # 预测一年的价格趋势
    year_prices = analyzer.predict_year(last_sequence)
    year_final_price = year_prices[-1][0]
    
    print('\n趋势预测结果：')
    print('====== 明日预测 ======')
    print(f'当前价格: {current_price:.2f}')
    print(f'预测价格: {future_price:.2f}')
    if future_price > current_price:
        print(f'预计上涨: {((future_price/current_price - 1) * 100):.2f}%')
    else:
        print(f'预计下跌: {((1 - future_price/current_price) * 100):.2f}%')
    
    print('\n====== 一周预测 ======')
    print(f'当前价格: {current_price:.2f}')
    print(f'一周后预测价格: {week_final_price:.2f}')
    if week_final_price > current_price:
        print(f'预计上涨: {((week_final_price/current_price - 1) * 100):.2f}%')
    else:
        print(f'预计下跌: {((1 - week_final_price/current_price) * 100):.2f}%')
    
    print('\n====== 一月预测 ======')
    print(f'当前价格: {current_price:.2f}')
    print(f'一月后预测价格: {month_final_price:.2f}')
    if month_final_price > current_price:
        print(f'预计上涨: {((month_final_price/current_price - 1) * 100):.2f}%')
    else:
        print(f'预计下跌: {((1 - month_final_price/current_price) * 100):.2f}%')
    
    print('\n====== 一年预测 ======')
    print(f'当前价格: {current_price:.2f}')
    print(f'一年后预测价格: {year_final_price:.2f}')
    if year_final_price > current_price:
        print(f'预计上涨: {((year_final_price/current_price - 1) * 100):.2f}%')
    else:
        print(f'预计下跌: {((1 - year_final_price/current_price) * 100):.2f}%')

if __name__ == '__main__':
    main()