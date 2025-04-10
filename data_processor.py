import pandas as pd
import numpy as np
from datetime import datetime

class DataProcessor:
    @staticmethod
    def process_sohu_data(raw_data):
        if not raw_data or not isinstance(raw_data, list):
            return None
        
        try:
            data = raw_data[0]['hq']
            # 检查数据列数
            if not data or not data[0]:
                print('搜狐数据源返回空数据')
                return None
                
            columns_count = len(data[0])
            print(f'搜狐数据列数: {columns_count}')
            
            # 根据数据列数动态分配列名
            if columns_count == 10:
                columns = ['date', 'open', 'close', 'change', 'change_percent',
                          'low', 'high', 'volume', 'amount', 'turnover']
            elif columns_count == 11:
                columns = ['date', 'open', 'close', 'change', 'change_percent',
                          'low', 'high', 'volume', 'amount', 'turnover', 'extra']
            else:
                print(f'搜狐数据源返回未知格式：{columns_count}列')
                return None
            
            df = pd.DataFrame(data, columns=columns)
            
            # 转换数据类型
            numeric_columns = ['open', 'close', 'low', 'high', 'volume', 'amount']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 将YYYY-MM-DD格式的日期字符串转换为datetime
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
            
            # 只保留需要的列
            keep_columns = ['date', 'open', 'close', 'low', 'high', 'volume', 'amount', 'turnover']
            df = df[keep_columns]
            
            return df
        except Exception as e:
            print(f'处理搜狐数据失败: {e}')
            return None

    @staticmethod
    def process_qq_data(raw_data):
        if not raw_data:
            print('腾讯数据源返回空数据')
            return None
            
        # 检查数据是否包含HTML错误页面
        if '<html' in raw_data.lower() or '404' in raw_data:
            print('腾讯数据源返回错误页面，可能是股票代码无效')
            return None
        
        try:
            # 解析腾讯数据格式
            lines = raw_data.split('\n')
            if len(lines) < 2:
                print('腾讯数据格式无效：数据行数不足')
                return None
                
            data = []
            valid_data_count = 0
            invalid_data_count = 0
            
            for line in lines[1:-1]:  # 跳过第一行和最后一行
                if line.strip():
                    try:
                        # 预处理：清理行中的转义字符
                        line = line.replace('\\n', '').replace('\\', '')
                        # 尝试不同的数据格式
                        parts = line.strip().split()
                        if len(parts) >= 6:
                            date_str, open_price, close_price, high_price, low_price, volume = parts[:6]
                        elif len(parts) >= 3:
                            # 简化格式：日期、收盘价、成交量
                            date_str, close_price, volume = parts[:3]
                            open_price = close_price
                            high_price = close_price
                            low_price = close_price
                        else:
                            invalid_data_count += 1
                            continue

                        # 验证数据有效性
                        if not date_str.isdigit():
                            print(f'无效日期格式: {date_str}')
                            invalid_data_count += 1
                            continue

                        # 处理不同长度的日期格式
                        if len(date_str) == 6:  # YYMMDD格式
                            year = int(date_str[:2])
                            # 处理年份，假设20xx年和19xx年
                            year = 2000 + year if year < 50 else 1900 + year
                            month = int(date_str[2:4])
                            day = int(date_str[4:6])
                            try:
                                date = datetime(year, month, day)
                            except ValueError:
                                print(f'无效日期值: {date_str}')
                                invalid_data_count += 1
                                continue
                        elif len(date_str) == 8:  # YYYYMMDD格式
                            try:
                                date = datetime.strptime(date_str, '%Y%m%d')
                            except ValueError:
                                print(f'无效日期值: {date_str}')
                                invalid_data_count += 1
                                continue
                        else:
                            print(f'无效日期格式长度: {date_str}')
                            invalid_data_count += 1
                            continue

                        # 清理数据中的特殊字符
                        volume = volume.rstrip('\n\\').strip()
                        
                        data.append({
                            'date': date,
                            'open': float(open_price),
                            'close': float(close_price),
                            'high': float(high_price),
                            'low': float(low_price),
                            'volume': float(volume)
                        })
                        valid_data_count += 1
                    except (ValueError, IndexError) as e:
                        print(f'跳过无效数据行: {line.strip()}, 错误: {e}')
                        invalid_data_count += 1
                        continue
            
            if valid_data_count == 0:
                print('没有找到有效的数据记录')
                return None
            
            if invalid_data_count > 0:
                print(f'数据处理完成：成功解析{valid_data_count}条记录，跳过{invalid_data_count}条无效记录')
            else:
                print(f'数据处理完成：成功解析{valid_data_count}条记录')
                
            return pd.DataFrame(data)
        except Exception as e:
            print(f'处理腾讯数据失败: {e}')
            return None

    @staticmethod
    def merge_data(sohu_df, qq_df):
        """合并来自不同数据源的数据"""
        if sohu_df is None and qq_df is None:
            return None
        
        if sohu_df is None:
            return qq_df
        if qq_df is None:
            return sohu_df
        
        # 以日期为索引合并数据
        merged_df = pd.concat([sohu_df, qq_df]).drop_duplicates(subset=['date'])
        merged_df.sort_values('date', inplace=True)
        merged_df.reset_index(drop=True, inplace=True)
        
        return merged_df

    @staticmethod
    def clean_data(df):
        """清理数据：处理缺失值、异常值等"""
        if df is None:
            return None
        
        # 删除缺失值
        df.dropna(inplace=True)
        
        # 删除重复数据
        df.drop_duplicates(inplace=True)
        
        # 确保数据按日期排序
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        return df