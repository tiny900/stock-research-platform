"""
Custom Tool: Technical Indicator Calculator
============================================
Purpose: Calculate key technical indicators for stock analysis.
This is the CUSTOM TOOL required by the assignment.

Inputs:
    - price_data: List of historical closing prices (list[float])
    - indicator: Name of the indicator to calculate (str)
    - params: Optional parameters for the indicator (dict)

Outputs:
    - Dictionary containing indicator name, values, signal, and interpretation

Supported Indicators:
    - SMA (Simple Moving Average)
    - EMA (Exponential Moving Average)
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Bollinger Bands

Limitations:
    - Requires sufficient historical data (minimum varies by indicator)
    - Does not account for volume or other non-price data
    - Signals are rule-based, not predictive
"""

import json
from typing import Any, Type, Optional
from pydantic import BaseModel, Field
from crewai.tools import BaseTool


class TechnicalIndicatorInput(BaseModel):
    """Input schema for the Technical Indicator Calculator."""
    price_data: str = Field(
        ...,
        description="JSON string of historical closing prices, e.g. '[150.0, 152.3, 148.7, ...]'"
    )
    indicator: str = Field(
        ...,
        description="Indicator to calculate: 'SMA', 'EMA', 'RSI', 'MACD', or 'BOLLINGER'"
    )
    params: Optional[str] = Field(
        default=None,
        description="Optional JSON string of parameters, e.g. '{\"period\": 14}'"
    )


class TechnicalIndicatorCalculator(BaseTool):
    """
    A custom tool that calculates technical indicators for stock analysis.
    Supports SMA, EMA, RSI, MACD, and Bollinger Bands.
    """
    name: str = "technical_indicator_calculator"
    description: str = (
        "Calculates technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands) "
        "from historical price data. Input price_data as a JSON list of floats, "
        "indicator as one of 'SMA','EMA','RSI','MACD','BOLLINGER', and optionally "
        "params as a JSON dict for custom parameters."
    )
    args_schema: Type[BaseModel] = TechnicalIndicatorInput

    def _run(self, price_data: str, indicator: str, params: Optional[str] = None) -> str:
        """Execute the technical indicator calculation."""
        try:
            prices = json.loads(price_data)
            if not isinstance(prices, list) or len(prices) < 2:
                return json.dumps({"error": "price_data must be a JSON list with at least 2 prices"})

            prices = [float(p) for p in prices]
            parameters = json.loads(params) if params else {}

            indicator = indicator.upper().strip()
            calculators = {
                "SMA": self._calculate_sma,
                "EMA": self._calculate_ema,
                "RSI": self._calculate_rsi,
                "MACD": self._calculate_macd,
                "BOLLINGER": self._calculate_bollinger,
            }

            if indicator not in calculators:
                return json.dumps({
                    "error": f"Unknown indicator '{indicator}'. Supported: {list(calculators.keys())}"
                })

            result = calculators[indicator](prices, parameters)
            return json.dumps(result, indent=2)

        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON input: {str(e)}"})
        except Exception as e:
            return json.dumps({"error": f"Calculation error: {str(e)}"})

    def _calculate_sma(self, prices: list[float], params: dict) -> dict:
        """Simple Moving Average."""
        period = params.get("period", 20)
        if len(prices) < period:
            return {"error": f"Need at least {period} prices for SMA-{period}, got {len(prices)}"}

        sma_values = []
        for i in range(period - 1, len(prices)):
            window = prices[i - period + 1: i + 1]
            sma_values.append(round(sum(window) / period, 4))

        current_price = prices[-1]
        current_sma = sma_values[-1]
        signal = "BULLISH" if current_price > current_sma else "BEARISH"

        return {
            "indicator": f"SMA-{period}",
            "current_value": current_sma,
            "latest_values": sma_values[-5:],
            "current_price": current_price,
            "signal": signal,
            "interpretation": (
                f"Price (${current_price:.2f}) is {'above' if signal == 'BULLISH' else 'below'} "
                f"SMA-{period} (${current_sma:.2f}), suggesting {signal.lower()} momentum."
            )
        }

    def _calculate_ema(self, prices: list[float], params: dict) -> dict:
        """Exponential Moving Average."""
        period = params.get("period", 20)
        if len(prices) < period:
            return {"error": f"Need at least {period} prices for EMA-{period}"}

        multiplier = 2 / (period + 1)
        ema_values = [sum(prices[:period]) / period]

        for price in prices[period:]:
            ema_values.append((price - ema_values[-1]) * multiplier + ema_values[-1])

        current_price = prices[-1]
        current_ema = round(ema_values[-1], 4)
        signal = "BULLISH" if current_price > current_ema else "BEARISH"

        return {
            "indicator": f"EMA-{period}",
            "current_value": current_ema,
            "latest_values": [round(v, 4) for v in ema_values[-5:]],
            "current_price": current_price,
            "signal": signal,
            "interpretation": (
                f"Price (${current_price:.2f}) is {'above' if signal == 'BULLISH' else 'below'} "
                f"EMA-{period} (${current_ema:.2f}), indicating {signal.lower()} trend."
            )
        }

    def _calculate_rsi(self, prices: list[float], params: dict) -> dict:
        """Relative Strength Index."""
        period = params.get("period", 14)
        if len(prices) < period + 1:
            return {"error": f"Need at least {period + 1} prices for RSI-{period}"}

        deltas = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]

        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        rsi_values = []
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss == 0:
                rsi_values.append(100.0)
            else:
                rs = avg_gain / avg_loss
                rsi_values.append(round(100 - (100 / (1 + rs)), 4))

        current_rsi = rsi_values[-1] if rsi_values else 50.0

        if current_rsi > 70:
            signal = "OVERBOUGHT"
            interpretation = f"RSI at {current_rsi:.1f} indicates overbought conditions. Potential pullback ahead."
        elif current_rsi < 30:
            signal = "OVERSOLD"
            interpretation = f"RSI at {current_rsi:.1f} indicates oversold conditions. Potential bounce ahead."
        else:
            signal = "NEUTRAL"
            interpretation = f"RSI at {current_rsi:.1f} is in neutral territory."

        return {
            "indicator": f"RSI-{period}",
            "current_value": current_rsi,
            "latest_values": [round(v, 2) for v in rsi_values[-5:]],
            "signal": signal,
            "interpretation": interpretation
        }

    def _calculate_macd(self, prices: list[float], params: dict) -> dict:
        """Moving Average Convergence Divergence."""
        fast = params.get("fast_period", 12)
        slow = params.get("slow_period", 26)
        signal_period = params.get("signal_period", 9)

        if len(prices) < slow + signal_period:
            return {"error": f"Need at least {slow + signal_period} prices for MACD"}

        def ema_series(data, period):
            multiplier = 2 / (period + 1)
            ema = [sum(data[:period]) / period]
            for val in data[period:]:
                ema.append((val - ema[-1]) * multiplier + ema[-1])
            return ema

        ema_fast = ema_series(prices, fast)
        ema_slow = ema_series(prices, slow)

        offset = slow - fast
        macd_line = [ema_fast[i + offset] - ema_slow[i] for i in range(len(ema_slow))]
        signal_line = ema_series(macd_line, signal_period)

        hist_offset = signal_period - 1
        histogram = [macd_line[i + hist_offset] - signal_line[i] for i in range(len(signal_line))]

        current_macd = round(macd_line[-1], 4)
        current_signal = round(signal_line[-1], 4)
        current_histogram = round(histogram[-1], 4)

        if current_macd > current_signal:
            signal = "BULLISH"
            interpretation = "MACD is above signal line, suggesting bullish momentum."
        else:
            signal = "BEARISH"
            interpretation = "MACD is below signal line, suggesting bearish momentum."

        if len(histogram) >= 2:
            if histogram[-2] < 0 and histogram[-1] > 0:
                signal = "BULLISH_CROSSOVER"
                interpretation = "MACD just crossed above signal line — bullish crossover!"
            elif histogram[-2] > 0 and histogram[-1] < 0:
                signal = "BEARISH_CROSSOVER"
                interpretation = "MACD just crossed below signal line — bearish crossover!"

        return {
            "indicator": f"MACD({fast},{slow},{signal_period})",
            "macd_line": current_macd,
            "signal_line": current_signal,
            "histogram": current_histogram,
            "signal": signal,
            "interpretation": interpretation
        }

    def _calculate_bollinger(self, prices: list[float], params: dict) -> dict:
        """Bollinger Bands."""
        period = params.get("period", 20)
        num_std = params.get("num_std", 2)

        if len(prices) < period:
            return {"error": f"Need at least {period} prices for Bollinger Bands"}

        window = prices[-period:]
        sma = sum(window) / period
        variance = sum((p - sma) ** 2 for p in window) / period
        std_dev = variance ** 0.5

        upper_band = round(sma + num_std * std_dev, 4)
        lower_band = round(sma - num_std * std_dev, 4)
        middle_band = round(sma, 4)
        current_price = prices[-1]
        bandwidth = round((upper_band - lower_band) / middle_band * 100, 4)

        if current_price > upper_band:
            signal = "OVERBOUGHT"
            interpretation = f"Price (${current_price:.2f}) is above upper band (${upper_band:.2f}). Potential pullback."
        elif current_price < lower_band:
            signal = "OVERSOLD"
            interpretation = f"Price (${current_price:.2f}) is below lower band (${lower_band:.2f}). Potential bounce."
        else:
            signal = "WITHIN_BANDS"
            pct_b = round((current_price - lower_band) / (upper_band - lower_band) * 100, 2)
            interpretation = f"Price is within bands. %B = {pct_b}%. Bandwidth = {bandwidth}%."

        return {
            "indicator": f"Bollinger({period},{num_std})",
            "upper_band": upper_band,
            "middle_band": middle_band,
            "lower_band": lower_band,
            "current_price": current_price,
            "bandwidth": bandwidth,
            "signal": signal,
            "interpretation": interpretation
        }
