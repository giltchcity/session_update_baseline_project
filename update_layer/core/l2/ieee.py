"""exp / log / log1p with C++ <cmath> (IEEE 754) results where Python's math raises: std::exp overflows to +inf,
std::log(0) = -inf, std::log(x < 0) = NaN, std::log1p(-1) = -inf, std::log1p(x < -1) = NaN. Inside the normal range
these are math.exp / math.log / math.log1p (libm, as std::), so the port stays bit-equal to the C++."""
import math


def exp(x: float) -> float:
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


def log(x: float) -> float:
    if x > 0.0:
        return math.log(x)
    return -math.inf if x == 0.0 else math.nan


def log1p(x: float) -> float:
    if x > -1.0:
        return math.log1p(x)
    return -math.inf if x == -1.0 else math.nan
