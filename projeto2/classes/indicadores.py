import numpy as np
import pandas as pd


# Função de cálculo dos indicadores
def calcula_indicadores(dados):
    dados = ind_williams_percent_r(dados, 14)
    dados = ind_roc(dados, 14)
    dados = ind_rsi(dados, 7)
    dados = ind_rsi(dados, 14)
    dados = ind_rsi(dados, 28)
    dados = ind_macd(dados, 8, 21)
    dados = ind_bbands(dados, 20)
    dados = ind_ichimoku_cloud(dados)
    dados = ind_ema(dados, 3)
    dados = ind_ema(dados, 8)
    dados = ind_ema(dados, 15)
    dados = ind_ema(dados, 50)
    dados = ind_ema(dados, 100)
    dados = ind_adx(dados, 14)
    dados = ind_donchian(dados, 10)
    dados = ind_donchian(dados, 20)
    dados = ind_alma(dados, 10)
    dados = ind_tsi(dados, 13, 25)
    dados = ind_zscore(dados, 20)
    dados = ind_log_return(dados, 10)
    dados = ind_log_return(dados, 20)
    dados = ind_vortex(dados, 7)
    dados = ind_aroon(dados, 16)
    dados = ind_ebsw(dados, 14)
    dados = ind_accbands(dados, 20)
    dados = ind_short_run(dados, 14)
    dados = ind_bias(dados, 26)
    dados = ind_ttm_trend(dados, 5, 20)
    dados = ind_percent_return(dados, 10)
    dados = ind_percent_return(dados, 20)
    dados = ind_kurtosis(dados, 5)
    dados = ind_kurtosis(dados, 10)
    dados = ind_kurtosis(dados, 20)
    dados = ind_eri(dados, 13)
    dados = ind_atr(dados, 14)
    dados = ind_keltner_channels(dados, 20)
    dados = ind_chaikin_volatility(dados, 10)
    dados = ind_stdev(dados, 5)
    dados = ind_stdev(dados, 10)
    dados = ind_stdev(dados, 20)
    dados = ta_vix(dados, 21)
    dados = ind_obv(dados, 10)
    dados = ind_chaikin_money_flow(dados, 5)
    dados = ind_volume_price_trend(dados, 7)
    dados = ind_accumulation_distribution_line(dados, 3)
    dados = ind_ease_of_movement(dados, 14)

    return dados


# Williams %R
def ind_williams_percent_r(dados_entrada, window=14):
    highest_high = dados_entrada["High"].rolling(window=window).max()
    lowest_low = dados_entrada["Low"].rolling(window=window).min()
    dados_entrada["Williams_%R{}".format(window)] = (
        -((highest_high - dados_entrada["Close"]) / (highest_high - lowest_low)) * 100
    )
    return dados_entrada


# Rate of Change
def ind_roc(dados_entrada, window=14):
    dados_entrada["ROC_{}".format(window)] = (
        dados_entrada["Close"] / dados_entrada["Close"].shift(window) - 1
    ) * 100
    return dados_entrada


# RSI
def ind_rsi(dados_entrada, window=14):
    delta = dados_entrada["Close"].diff(1)
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    avg_gain = gains.rolling(window=window, min_periods=1).mean()
    avg_loss = losses.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    dados_entrada["rsi_{}".format(window)] = 100 - (100 / (1 + rs))
    return dados_entrada


# MACD
def ind_macd(dados_entrada, short_window=8, long_window=21, signal_window=9):
    short_ema = dados_entrada["Close"].ewm(span=short_window, adjust=False).mean()
    long_ema = dados_entrada["Close"].ewm(span=long_window, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    dados_entrada["MACD_Line"] = macd_line
    dados_entrada["Signal_Line"] = signal_line
    dados_entrada["MACD_Histogram"] = macd_histogram
    return dados_entrada


# Bollinger Bands
def ind_bbands(dados_entrada, window=20, num_std_dev=2):
    dados_entrada["midlle_band"] = dados_entrada["Close"].rolling(window=window).mean()
    dados_entrada["std"] = dados_entrada["Close"].rolling(window=window).std()
    dados_entrada["upper_band{}".format(window)] = dados_entrada["midlle_band"] + (
        num_std_dev * dados_entrada["std"]
    )
    dados_entrada["lower_band{}".format(window)] = dados_entrada["midlle_band"] - (
        num_std_dev * dados_entrada["std"]
    )
    dados_entrada.drop(["std"], axis=1, inplace=True)
    return dados_entrada


# Ichimoku Cloud
def ind_ichimoku_cloud(
    dados_entrada,
    window_tenkan=9,
    window_kijun=26,
    window_senkou_span_b=52,
    window_chikou=26,
):
    tenkan_sen = (
        dados_entrada["Close"].rolling(window=window_tenkan).max()
        + dados_entrada["Close"].rolling(window=window_tenkan).min()
    ) / 2
    kijun_sen = (
        dados_entrada["Close"].rolling(window=window_kijun).max()
        + dados_entrada["Close"].rolling(window=window_kijun).min()
    ) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(window_kijun)
    senkou_span_b = (
        dados_entrada["Close"].rolling(window=window_senkou_span_b).max()
        + dados_entrada["Close"].rolling(window=window_senkou_span_b).min()
    ) / 2
    chikou_span = dados_entrada["Close"].shift(-window_chikou)
    dados_entrada["Tenkan_sen"] = tenkan_sen
    dados_entrada["Kijun_sen"] = kijun_sen
    dados_entrada["Senkou_Span_A"] = senkou_span_a
    dados_entrada["Senkou_Span_B"] = senkou_span_b
    dados_entrada["Chikou_Span"] = chikou_span
    return dados_entrada


# Moving Average (EMA)
def ind_ema(dados_entrada, window=8):
    dados_entrada["ema_{}".format(window)] = (
        dados_entrada["Close"].ewm(span=window, adjust=False).mean()
    )
    return dados_entrada


# ADX
def ind_adx(dados_entrada, window=14):  # 14
    dados_entrada["TR"] = (
        abs(dados_entrada["High"] - dados_entrada["Low"])
        .combine_first(abs(dados_entrada["High"] - dados_entrada["Close"].shift(1)))
        .combine_first(abs(dados_entrada["Low"] - dados_entrada["Close"].shift(1)))
    )
    dados_entrada["DMplus"] = (
        dados_entrada["High"] - dados_entrada["High"].shift(1)
    ).apply(lambda x: x if x > 0 else 0)
    dados_entrada["DMminus"] = (
        dados_entrada["Low"].shift(1) - dados_entrada["Low"]
    ).apply(lambda x: x if x > 0 else 0)
    dados_entrada["ATR"] = dados_entrada["TR"].rolling(window=window).mean()
    dados_entrada["DIplus"] = (
        dados_entrada["DMplus"].rolling(window=window).mean() / dados_entrada["ATR"]
    ) * 100
    dados_entrada["DIminus"] = (
        dados_entrada["DMminus"].rolling(window=window).mean() / dados_entrada["ATR"]
    ) * 100
    dados_entrada["DX"] = (
        abs(dados_entrada["DIplus"] - dados_entrada["DIminus"])
        / (dados_entrada["DIplus"] + dados_entrada["DIminus"])
        * 100
    )
    dados_entrada["ADX_{}".format(window)] = (
        dados_entrada["DX"].rolling(window=window).mean()
    )
    dados_entrada.drop(
        ["TR", "DMplus", "DMminus", "ATR", "DIplus", "DIminus", "DX"],
        axis=1,
        inplace=True,
    )
    return dados_entrada


# Donchian Channel
def ind_donchian(dados_entrada, window=10):
    highest_high = dados_entrada["Close"].rolling(window=window).max()
    lowest_low = dados_entrada["Close"].rolling(window=window).min()
    dados_entrada["Donchian_Upper_{}".format(window)] = highest_high
    dados_entrada["Donchian_Lower_{}".format(window)] = lowest_low
    return dados_entrada


# Arnaud Legoux Moving Average (ALMA)
def ind_alma(dados_entrada, window=10, sigma=6, offset=0.85):
    m = np.linspace(-offset * (window - 1), offset * (window - 1), window)
    w = np.exp(-0.5 * (m / sigma) ** 2)
    w /= w.sum()
    alma_values = np.convolve(dados_entrada["Close"].values, w, mode="valid")
    alma_values = np.concatenate([np.full(window - 1, np.nan), alma_values])
    dados_entrada["ALMA_{}".format(window)] = alma_values
    return dados_entrada


# True Strength Index (TSI)
def ind_tsi(dados_entrada, short_period=13, long_period=25):
    price_diff = dados_entrada["Close"].diff(1)
    double_smoothed = (
        price_diff.ewm(span=short_period, min_periods=1, adjust=False)
        .mean()
        .ewm(span=long_period, min_periods=1, adjust=False)
        .mean()
    )
    double_smoothed_abs = (
        price_diff.abs()
        .ewm(span=short_period, min_periods=1, adjust=False)
        .mean()
        .ewm(span=long_period, min_periods=1, adjust=False)
        .mean()
    )
    tsi_values = 100 * double_smoothed / double_smoothed_abs
    dados_entrada["TSI_{}_{}".format(short_period, long_period)] = tsi_values
    return dados_entrada


# Z-Score
def ind_zscore(dados_entrada, window=20):
    rolling_mean = dados_entrada["Close"].rolling(window=window).mean()
    rolling_std = dados_entrada["Close"].rolling(window=window).std()
    z_score = (dados_entrada["Close"] - rolling_mean) / rolling_std
    dados_entrada["Z_Score_{}".format(window)] = z_score
    return dados_entrada


# Log Return
def ind_log_return(dados_entrada, window=5):
    dados_entrada["LogReturn_{}".format(window)] = (
        dados_entrada["Close"]
        .pct_change(window)
        .apply(lambda x: 0 if pd.isna(x) else x)
    )
    return dados_entrada


# Vortex Indicator
def ind_vortex(dados_entrada, window=7):
    high_low = dados_entrada["High"] - dados_entrada["Low"]
    high_close_previous = abs(dados_entrada["High"] - dados_entrada["Close"].shift(1))
    low_close_previous = abs(dados_entrada["Low"] - dados_entrada["Close"].shift(1))
    true_range = pd.concat(
        [high_low, high_close_previous, low_close_previous], axis=1
    ).max(axis=1)
    positive_vm = abs(dados_entrada["High"].shift(1) - dados_entrada["Low"])
    negative_vm = abs(dados_entrada["Low"].shift(1) - dados_entrada["High"])
    true_range_sum = true_range.rolling(window=window).sum()
    positive_vm_sum = positive_vm.rolling(window=window).sum()
    negative_vm_sum = negative_vm.rolling(window=window).sum()
    positive_vi = positive_vm_sum / true_range_sum
    negative_vi = negative_vm_sum / true_range_sum
    dados_entrada["Positive_VI_{}".format(window)] = positive_vi
    dados_entrada["Negative_VI_{}".format(window)] = negative_vi
    return dados_entrada


# Aroon Indicator
def ind_aroon(dados_entrada, window=16):
    high_prices = dados_entrada["High"]
    low_prices = dados_entrada["Low"]
    aroon_up = []
    aroon_down = []
    for i in range(window, len(high_prices)):
        high_period = high_prices[i - window : i + 1]
        low_period = low_prices[i - window : i + 1]
        high_index = window - high_period.values.argmax() - 1
        low_index = window - low_period.values.argmin() - 1
        aroon_up.append((window - high_index) / window * 100)
        aroon_down.append((window - low_index) / window * 100)
    aroon_up = [None] * window + aroon_up
    aroon_down = [None] * window + aroon_down
    dados_entrada["Aroon_Up_{}".format(window)] = aroon_up
    dados_entrada["Aroon_Down_{}".format(window)] = aroon_down
    return dados_entrada


# Elder"s Bull Power e Bear Power
def ind_ebsw(dados_entrada, window=14):
    ema = dados_entrada["Close"].ewm(span=window, adjust=False).mean()
    bull_power = dados_entrada["High"] - ema
    bear_power = dados_entrada["Low"] - ema
    dados_entrada["Bull_Power_{}".format(window)] = bull_power
    dados_entrada["Bear_Power_{}".format(window)] = bear_power
    return dados_entrada


# Acceleration Bands
def ind_accbands(dados_entrada, window=20, acceleration_factor=0.02):
    sma = dados_entrada["Close"].rolling(window=window).mean()
    band_difference = dados_entrada["Close"] * acceleration_factor
    upper_band = sma + band_difference
    lower_band = sma - band_difference
    dados_entrada["Upper_Band_{}".format(window)] = upper_band
    dados_entrada["Lower_Band_{}".format(window)] = lower_band
    dados_entrada["Middle_Band_{}".format(window)] = sma
    return dados_entrada


# Short Run
def ind_short_run(dados_entrada, window=14):
    short_run = (
        dados_entrada["Close"] - dados_entrada["Close"].rolling(window=window).min()
    )
    dados_entrada["Short_Run_{}".format(window)] = short_run
    return dados_entrada


# Bias
def ind_bias(dados_entrada, window=26):
    moving_average = dados_entrada["Close"].rolling(window=window).mean()
    bias = ((dados_entrada["Close"] - moving_average) / moving_average) * 100
    dados_entrada["Bias_{}".format(window)] = bias
    return dados_entrada


# TTM Trend
def ind_ttm_trend(dados_entrada, short_window=5, long_window=20):
    short_ema = dados_entrada["Close"].ewm(span=short_window, adjust=False).mean()
    long_ema = dados_entrada["Close"].ewm(span=long_window, adjust=False).mean()
    ttm_trend = short_ema - long_ema
    dados_entrada["TTM_Trend_{}_{}".format(short_window, long_window)] = ttm_trend
    return dados_entrada


# Percent Return
def ind_percent_return(dados_entrada, window=1):
    percent_return = (
        dados_entrada["Close"].pct_change().rolling(window=window).mean() * 100
    )
    dados_entrada["Percent_Return_{}".format(window)] = percent_return
    return dados_entrada


# Kurtosis
def ind_kurtosis(dados_entrada, window=20):
    dados_entrada["kurtosis_{}".format(window)] = (
        dados_entrada["Close"]
        .rolling(window=window)
        .apply(lambda x: np.nan if x.isnull().any() else x.kurt())
    )
    return dados_entrada


# Elder's Force Index (ERI)
def ind_eri(dados_entrada, window=13):
    price_change = dados_entrada["Close"].diff()
    force_index = price_change * dados_entrada["Volume"]
    eri = force_index.ewm(span=window, adjust=False).mean()
    dados_entrada["ERI_{}".format(window)] = eri
    return dados_entrada


# ATR
def ind_atr(dados_entrada, window=14):
    dados_entrada["High-Low"] = dados_entrada["High"] - dados_entrada["Low"]
    dados_entrada["High-PrevClose"] = abs(
        dados_entrada["High"] - dados_entrada["Close"].shift(1)
    )
    dados_entrada["Low-PrevClose"] = abs(
        dados_entrada["Low"] - dados_entrada["Close"].shift(1)
    )
    dados_entrada["TrueRange"] = dados_entrada[
        ["High-Low", "High-PrevClose", "Low-PrevClose"]
    ].max(axis=1)
    dados_entrada["atr_{}".format(window)] = (
        dados_entrada["TrueRange"].rolling(window=window, min_periods=1).mean()
    )
    dados_entrada.drop(
        ["High-Low", "High-PrevClose", "Low-PrevClose", "TrueRange"],
        axis=1,
        inplace=True,
    )
    return dados_entrada


# Keltner Channels
def ind_keltner_channels(dados_entrada, period=20, multiplier=2):
    dados_entrada["TR"] = dados_entrada.apply(
        lambda row: max(
            row["High"] - row["Low"],
            abs(row["High"] - row["Close"]),
            abs(row["Low"] - row["Close"]),
        ),
        axis=1,
    )
    dados_entrada["ATR"] = dados_entrada["TR"].rolling(window=period).mean()
    dados_entrada["Middle Band"] = dados_entrada["Close"].rolling(window=period).mean()
    dados_entrada["Upper Band"] = (
        dados_entrada["Middle Band"] + multiplier * dados_entrada["ATR"]
    )
    dados_entrada["Lower Band"] = (
        dados_entrada["Middle Band"] - multiplier * dados_entrada["ATR"]
    )
    return dados_entrada


# Chaikin Volatility
def ind_chaikin_volatility(dados_entrada, window=10):
    daily_returns = dados_entrada["Close"].pct_change()
    chaikin_volatility = daily_returns.rolling(window=window).std() * (252**0.5)
    dados_entrada["Chaikin_Volatility_{}".format(window)] = chaikin_volatility
    return dados_entrada


# Standard Deviation
def ind_stdev(dados_entrada, window=1):
    stdev_column = dados_entrada["Close"].rolling(window=window).std()
    dados_entrada["Stdev_{}".format(window)] = stdev_column
    return dados_entrada


# Volatility Index (VIX)
def ta_vix(dados_entrada, window=21):
    returns = dados_entrada["Close"].pct_change().dropna()
    rolling_std = returns.rolling(window=window).std()
    vix = rolling_std * np.sqrt(252) * 100
    dados_entrada["VIX_{}".format(window)] = vix
    return dados_entrada


# On-Balance Volume (OBV)
def ind_obv(dados_entrada, window=10):
    price_changes = dados_entrada["Close"].diff()
    volume_direction = pd.Series(1, index=price_changes.index)
    volume_direction[price_changes < 0] = -1
    obv = (dados_entrada["Volume"] * volume_direction).cumsum()
    obv_smoothed = obv.rolling(window=window).mean()
    dados_entrada["OBV_{}".format(window)] = obv_smoothed
    return dados_entrada


# Chaikin Money Flow (CMF)
def ind_chaikin_money_flow(dados_entrada, window=10):
    mf_multiplier = (
        (dados_entrada["Close"] - dados_entrada["Close"].shift(1))
        + (dados_entrada["Close"] - dados_entrada["Close"].shift(1)).abs()
    ) / 2
    mf_volume = mf_multiplier * dados_entrada["Volume"]
    adl = mf_volume.cumsum()
    cmf = (
        adl.rolling(window=window).mean()
        / dados_entrada["Volume"].rolling(window=window).mean()
    )
    dados_entrada["CMF_{}".format(window)] = cmf
    return dados_entrada


# Volume Price Trend (VPT)
def ind_volume_price_trend(dados_entrada, window=10):
    price_change = dados_entrada["Close"].pct_change()
    vpt = (price_change * dados_entrada["Volume"].shift(window)).cumsum()
    dados_entrada["VPT_{}".format(window)] = vpt
    return dados_entrada


# Accumulation/Distribution Line
def ind_accumulation_distribution_line(dados_entrada, window=10):
    money_flow_multiplier = (
        (dados_entrada["Close"] - dados_entrada["Close"].shift(1))
        - (dados_entrada["Close"].shift(1) - dados_entrada["Close"])
    ) / (dados_entrada["Close"].shift(1) - dados_entrada["Close"])
    money_flow_volume = money_flow_multiplier * dados_entrada["Volume"]
    ad_line = money_flow_volume.cumsum()
    ad_line_smoothed = ad_line.rolling(window=window, min_periods=1).mean()
    dados_entrada["A/D Line_{}".format(window)] = ad_line_smoothed
    return dados_entrada


# Ease of Movement (EOM)
def ind_ease_of_movement(dados_entrada, window=14):
    midpoint_move = ((dados_entrada["High"] + dados_entrada["Low"]) / 2).diff(1)
    box_ratio = (
        dados_entrada["Volume"]
        / 1000000
        / (dados_entrada["High"] - dados_entrada["Low"])
    )
    eom = midpoint_move / box_ratio
    eom_smoothed = eom.rolling(window=window, min_periods=1).mean()
    dados_entrada["EOM_{}".format(window)] = eom_smoothed
    return dados_entrada
