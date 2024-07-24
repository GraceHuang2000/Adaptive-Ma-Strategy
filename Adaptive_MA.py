import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt


def cal_return(df, period):
    # 获取收盘价序列
    close = df['close']

    # 使用shift函数平移收盘价列获得开始结束价格
    start_price = close
    end_price = df['close'].shift(-period)

    # 计算回报率公式
    df['return' + '_' + str(period)] = (end_price - start_price) / start_price

    # 可以直接得出所有日期区间的回报序列

    return df



def generate_signals(df):
    high = df['high'].rolling(window=10).mean()
    low = df['low'].rolling(window=10).mean()

    result = ols("high ~ low", data=pd.DataFrame({'high': high, 'low': low})).fit()

    df['rsrs_slope_value'] = result.params[1]

    signals = df
    signals.loc[signals['rsrs_slope_value'] > 0.95, 'signal'] = 1
    # 产生延迟2天的卖出信号
    signals.loc[signals['signal'].shift(-2) == 1, 'signal'] = -1
    # signals.loc[signals['rsrs_slope_value'] < 0.85, 'signal'] = -1

    # 生成交易记录数据
    trades = signals.copy()
    trades['pos'] = np.where(trades['signal'] == 1, 1,
                             np.where(trades['signal'] == -1, 0, np.nan))

    # 计算持仓净值曲线
    trades['pnl'] = trades['pos'].shift(1) * df['close'].pct_change()
    trades['capital'] = 1 + trades['pnl'].cumsum()

    trades['strategy'] = trades['capital'] * trades.loc[0, 'close']

    # 绘制净值走势对比图
    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(211)
    ax1.plot(trades['close'], label='Close')
    ax1.plot(trades['strategy'], label='Strategy')
    ax1.legend()
    ax1.set_title('Price and Strategy Equity')

    ax2 = fig.add_subplot(212)
    trades['strategy_ret'] = trades['strategy'].pct_change()
    trades['close_ret'] = trades['close'].pct_change()
    ax2.plot(trades['strategy_ret'], label='Strategy Returns')
    ax2.plot(trades['close_ret'], label='Price Returns')
    ax2.legend()
    ax2.set_title('Returns Comparison')
    plt.show()

    return trades





def ma_break(df, ma_list, ma_diff_list,ma_params=10, sell_ma_parmas=5, weight_step=0.05, buy_weight='Y', sell_weight='Y'):
    signal = 0
    records = []
    investment = 10000

    ## 生成weight arr
    d = weight_step
    ##buy weight arr
    buy_weights_arr = np.arange(1 - (ma_params - 1) * weight_step, 1 + weight_step, weight_step)
    sell_weights_arr = np.arange(1 - (sell_ma_parmas - 1) * weight_step, 1 + weight_step, weight_step)[:sell_ma_parmas]

    # buy_weights_arr = np.arange(1, 1 - (ma_params) * d, -d)
    # sell_weights_arr = np.arange(1, 1 - (sell_ma_parmas) * d, -d)

    print(len(buy_weights_arr), len(sell_weights_arr))
    # print(buy_weights_arr)
    # print(sell_weights_arr)
    for i in range(ma_params, len(df)):

        if buy_weight == 'N':
            buy_ma_arr = np.array(df['adjust_close'].iloc[i - ma_params:i])
        elif buy_weight == 'Y':
            buy_ma_arr = np.array(df['adjust_close'].iloc[i - ma_params:i]) * buy_weights_arr

        sell_ma_days = sell_ma_parmas_cal(data, i, ma_list, ma_diff_list)

        if sell_weight == 'N':
            sell_ma_arr = np.array(df['adjust_close'].iloc[i - sell_ma_days:i])
        elif sell_weight == 'Y':
            sell_ma_arr = np.array(df['adjust_close'].iloc[i - sell_ma_days:i]) * sell_weights_arr

        # temp_arr = np.mean(temp_arr1)
        buy_ma = np.mean(buy_ma_arr)
        # print(f"Index: {i}, Adjusted Close: {df.loc[i, 'adjust_close']}, MA: {temp_arr}, Signal: {signal}")

        if signal == 0 and df.loc[i, 'adjust_close'] > buy_ma * 1:
            temp_buy_price = df.loc[i, 'adjust_close']
            signal = i
            # buy_date = df.loc[i, 'date']
            # shares = investment/temp_buy_price
            # print(f"Buy Signal at {buy_date} with price {temp_buy_price}")

        elif signal > 0 and df.loc[i, 'adjust_close'] < np.mean(sell_ma_arr) * 1.01:

            temp_buy_price = df.loc[signal, 'adjust_close']
            # signal = i
            buy_date = df.loc[signal, 'date']
            shares = investment / temp_buy_price
            temp_sell_price = df.loc[i, 'adjust_close']
            # temp_profit = (temp_sell_price - temp_buy_price) * shares
            investment = temp_sell_price * shares
            earn_percent = (temp_sell_price - temp_buy_price) / temp_buy_price
            sell_date = df.loc[i, 'date']
            record = {
                'sell_ma_days':sell_ma_days,
                'earn/loss percent': earn_percent,
                'pnl': temp_sell_price - temp_buy_price,
                'cum_profit': investment / 10000,
                'investment': investment,
                'buy_date': buy_date,
                'buy_price': temp_buy_price,
                'sell_date': sell_date,
                'sell_price': temp_sell_price,
            }
            records.append(record)
            print(record)
            signal = 0
            # print(f"Sell Signal at {sell_date} with price {temp_sell_price}. Profit: {temp_profit}, Cum. Profit: {cum_profit}")

    print(len(records))
    record_result = pd.DataFrame(records)
    result(record_result)
    return records


def result(records):
    trades = records.copy()
    win_trades = len(trades[trades['pnl'] > 0])
    total_trades = len(trades)
    win_rate = win_trades / total_trades

    profits = trades['pnl'][trades['pnl'] > 0].sum()
    losses = abs(trades['pnl'][trades['pnl'] < 0].sum())
    profit_factor = profits / (profits - losses)

    avg_return = trades['pnl'].mean()
    std_return = trades['pnl'].std()
    sharp_ratio = np.sqrt(252) * (avg_return / std_return)

    print('win_rate', win_rate, 'profit_factor', profit_factor, 'sharp_ratio', sharp_ratio)

    return


def ma_diff_cal(data, window_list):
    ma_list = []
    ma_diff_list = []
    for i in window_list:
        data['ma' + "_" + str(i)] = data['adjust_close'].rolling(window=i).mean()
        ma_list.append('ma' + "_" + str(i))
    ##计算均线差
    for i in range(0, len(window_list) - 2):
        for j in range(i+1 ,len(window_list) - 1 ):
            data['ma_diff' + '_' + str(window_list[i]) + '_' + str(window_list[j])] = (
                        data['ma' + "_" + str(window_list[i])] - data['ma' + "_" + str(window_list[j])])
            ma_diff_list.append('ma_diff' + '_' + str(window_list[i]) + '_' + str(window_list[i + 1]))

    return ma_list, ma_diff_list


def sell_ma_parmas_cal(data, i, ma_list, ma_diff_list):
    ma_up_num_list = []
    ma_diff_up_num = []

    for item in ma_list:
        ma_arr = np.array(data[item][:i])[::-1]  ##转置倒序
        up_num = 0
        for k in range(len(ma_arr)-1):
            if ma_arr[k] > ma_arr[k+1]:
                up_num +=1
            if ma_arr[k] < ma_arr[k+1]:
                break
        ma_up_num_list.append(up_num)

    for item in ma_diff_list:
        ma_diff_arr = np.array(data[item][:i])[::-1]  ##转置倒序
        up_num = 0
        for k in range(len(ma_diff_arr) - 1):
            if ma_diff_arr[k] > ma_diff_arr[k + 1]:
                up_num += 1
            if ma_diff_arr[k] < ma_diff_arr[k + 1]:
                break
        ma_diff_up_num.append(up_num)

    ave_ma_up_days = np.mean(np.array(ma_up_num_list))
    # ave_ma_diff_up_days = np.mean(np.array(ma_diff_up_num))
    sell_ma_days = int((ave_ma_up_days )/2)
    # buy_ma_days = min((ave_ma_up_days ) -10

    return sell_ma_days


if __name__ == '__main__':
    # 读取数据
    file_path = '***'
    data = pd.read_excel(file_path)
    print(data.columns)
    data.columns = ['date', 'adjust_open', 'adjust_high', 'adjust_low', 'adjust_close', 'Volume', 'Turnover',
                    'Transactions', 'MA1', 'MA2', 'MA3', 'MA4', 'MA5', 'MA6']
    data['high'] = data['adjust_high']
    data['close'] = data['adjust_close']
    data['low'] = data['adjust_low']

    window_list = [3, 5, 8, 10, 15, 20, 30]
    ma_list, ma_diff_list = ma_diff_cal(data, window_list)
    print( ma_list, ma_diff_list )

    ma_break(data, ma_list, ma_diff_list,ma_params=5, sell_ma_parmas=5, weight_step=0.05, buy_weight='N', sell_weight='N')

