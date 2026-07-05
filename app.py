"""
NIFTY GAMMA EXPOSURE (GEX) — STREAMLIT DASHBOARD
================================================
Interactive dashboard version of nifty_gex_gamma_analysis.py.

Data sources
------------
- SPOT           : tvdatafeed  ->  NSE:NIFTY  (last 1-minute close)      [as requested]
- HISTORICAL VOL : tvdatafeed  ->  NSE:NIFTY daily bars (RV + fallback IV)
- OPTION CHAIN   : NSE option-chain API (same source as the reference script/skill)

All GEX analytics (Black-Scholes greeks, DgammaDtime/"Color", 3 gravity
methods, gamma pin, floor/ceiling, upside & downside hedge walls, K*) are the
exact functions from nifty_gex_gamma_analysis.py, unchanged.

The two hardcoded "gotchas" the skill file calls out are fixed here:
  * expiry is now chosen from a live dropdown (NSE `records.expiryDates`)
  * strike range now auto-centres on spot (with a slider override)

Run locally:   streamlit run app.py
"""

import matplotlib
matplotlib.use("Agg")  # headless backend for servers / Streamlit Cloud

import time
from datetime import datetime, timedelta
from math import log, sqrt, exp

import numpy as np
import pandas as pd
import pytz
import requests
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d

# =============================================================================
# CONFIGURATION
# =============================================================================

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    ),
    "Accept-Language":  "en-US,en;q=0.9",
    "Accept-Encoding":  "gzip, deflate",  # NOT br — avoids brotli decode failures
    "Connection":       "keep-alive",
    "Accept":           "application/json",
    "Referer":          "https://www.nseindia.com/option-chain",
    "Host":             "www.nseindia.com",
    "X-Requested-With": "XMLHttpRequest",
}

FIELD_MAPPINGS = {
    "strikePrice":          "strike",
    "impliedVolatility":    "impliedVolatility",
    "openInterest":         "openInterest",
    "lastPrice":            "lastPrice",
    "changeinOpenInterest": "changeInOpenInterest",
    "totalTradedVolume":    "totalTradedVolume",
}

# Sole chain source: option-chain-v3. A call WITH a valid expiry returns both the
# chain for that expiry AND records.expiryDates (the full list of all expiries),
# so v3 is used for everything. Expiry is appended as &expiry=DD-FullMonth-YYYY
# e.g.  07-July-2026
NSE_OC_V3 = "https://www.nseindia.com/api/option-chain-v3?type=Indices&symbol=NIFTY"
RISK_FREE_RATE_DEFAULT = 0.05
MIN_T = 1 / (24 * 60)  # one-minute floor on time-to-expiry (in years)


# =============================================================================
# BLACK-SCHOLES GREEKS  (verbatim from nifty_gex_gamma_analysis.py)
# =============================================================================

def bs_greeks(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0, 0
    try:
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        delta = norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1)
        gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
        return delta, gamma
    except Exception:
        return 0, 0


def bs_dgamma_dtime(S, K, T, r, sigma):
    """Returns (color_raw, color_per_day). color_per_day = -color_raw / 365."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0, 0
    try:
        sqrtT = sqrt(T)
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        term = 1 + d1 * (2 * r * T - d2 * sigma * sqrtT) / (sigma * sqrtT)
        color_raw = -(norm.pdf(d1) / (2 * S * T * sigma * sqrtT)) * term
        color_per_day = -color_raw / 365.0
        return color_raw, color_per_day
    except Exception:
        return 0, 0


# =============================================================================
# GEX / DENSITY / DETONATION / SIGMA  (verbatim)
# =============================================================================

def compute_gex_with_fallback(df_options, option_type, spot_price,
                              fallback_iv, t_expiry, r):
    total, count = 0, 0
    for _, row in df_options.iterrows():
        K = row["strike"]
        iv = row["impliedVolatility"]
        OI = row["openInterest"]
        if iv <= 0 and fallback_iv is not None:
            iv = fallback_iv
        if iv <= 0 or OI <= 0:
            continue
        _, gamma = bs_greeks(spot_price, K, t_expiry, r, iv, option_type)
        total += gamma * OI * spot_price * 100
        count += 1
    return total, count


def compute_gex_density(calls_df, puts_df, spot_price, time_to_expiry,
                        risk_free_rate, fallback_vol):
    all_strikes = sorted(set(calls_df["strike"].tolist() + puts_df["strike"].tolist()))
    rows = []
    for K in all_strikes:
        cgex = pgex = coi = poi = cp = pp = 0
        ccolor = pcolor = 0

        cd = calls_df[calls_df["strike"] == K]
        if not cd.empty:
            r = cd.iloc[0]
            iv = r["impliedVolatility"]
            iv = iv / 100.0 if iv > 3 else iv     # NSE IV is in % (e.g. 12.3 -> 0.123)
            OI = r["openInterest"]
            coi = OI
            cp = r["lastPrice"]
            if iv <= 0 and fallback_vol:
                iv = fallback_vol
            if iv > 0 and OI > 0:
                _, g = bs_greeks(spot_price, K, time_to_expiry, risk_free_rate, iv, "call")
                cgex = g * OI * spot_price * 100
                _, color_pd = bs_dgamma_dtime(spot_price, K, time_to_expiry, risk_free_rate, iv)
                ccolor = color_pd * OI * spot_price * 100

        pd_ = puts_df[puts_df["strike"] == K]
        if not pd_.empty:
            r = pd_.iloc[0]
            iv = r["impliedVolatility"]
            iv = iv / 100.0 if iv > 3 else iv     # NSE IV is in % (e.g. 12.3 -> 0.123)
            OI = r["openInterest"]
            poi = OI
            pp = r["lastPrice"]
            if iv <= 0 and fallback_vol:
                iv = fallback_vol
            if iv > 0 and OI > 0:
                _, g = bs_greeks(spot_price, K, time_to_expiry, risk_free_rate, iv, "put")
                pgex = g * OI * spot_price * 100
                _, color_pd = bs_dgamma_dtime(spot_price, K, time_to_expiry, risk_free_rate, iv)
                pcolor = color_pd * OI * spot_price * 100

        avg_prem = (cp + pp) / 2 if (cp + pp) > 0 else 0
        rows.append({
            "strike": K, "call_gex": cgex, "put_gex": pgex,
            "net_gex": cgex - pgex, "sell_gamma": -cgex, "buy_gamma": pgex,
            "total_oi": coi + poi, "avg_premium": avg_prem,
            "call_oi": coi, "put_oi": poi,
            "call_premium": cp, "put_premium": pp,
            "ce_price": cp, "pe_price": pp,
            "call_color": ccolor, "put_color": pcolor,
            "net_color": ccolor - pcolor, "total_color": ccolor + pcolor,
        })
    return pd.DataFrame(rows)


def compute_gamma_detonation(oi, prem, spot, rv, iv, n, dte):
    if oi == 0 or prem == 0:
        return 0
    dte = max(dte, 1)
    num = (oi * prem * spot) * np.sqrt(252 / dte)
    den = rv + n * (iv - rv)
    if abs(den) < 1e-6:
        den = 1e-6
    return num / den


def compute_sigma_levels(spot, iv, T, n=1):
    if iv <= 0 or T <= 0:
        return spot, spot
    m = iv * sqrt(T) * n
    return spot * exp(+m), spot * exp(-m)


def compute_volatility_trigger(gex_df, spot_price):
    strikes = gex_df["strike"].values
    net_gex = gex_df["net_gex"].values
    crossings = []
    for i in range(len(net_gex) - 1):
        if net_gex[i] * net_gex[i + 1] < 0:
            x1, x2 = strikes[i], strikes[i + 1]
            y1, y2 = net_gex[i], net_gex[i + 1]
            if (y2 - y1) != 0:
                crossings.append(x1 - y1 * (x2 - x1) / (y2 - y1))
    if not crossings:
        return None
    return min(crossings, key=lambda x: abs(x - spot_price))


# =============================================================================
# GRAVITY CENTERS — 3 METHODS  (verbatim)
# =============================================================================

def compute_gravity_centers(gex_df, spot_price, vol_trigger):
    flip = vol_trigger if vol_trigger is not None else spot_price
    above = gex_df[gex_df["strike"] > flip].copy()
    below = gex_df[gex_df["strike"] < flip].copy()

    call_wall = (above.loc[above["call_gex"].idxmax(), "strike"]
                 if not above.empty else spot_price * 1.03)
    put_wall = (below.loc[below["put_gex"].idxmax(), "strike"]
                if not below.empty else spot_price * 0.97)

    call_ch = call_wall - flip
    put_ch = flip - put_wall

    call_fixed = flip + 0.30 * call_ch
    put_fixed = flip - 0.35 * put_ch

    if not above.empty and above["call_gex"].sum() > 0:
        call_cen = (above["strike"] * above["call_gex"]).sum() / above["call_gex"].sum()
    else:
        call_cen = call_fixed
    if not below.empty and below["put_gex"].sum() > 0:
        put_cen = (below["strike"] * below["put_gex"]).sum() / below["put_gex"].sum()
    else:
        put_cen = put_fixed

    call_cen_ratio = (call_cen - flip) / call_ch if call_ch > 0 else 0.30
    put_cen_ratio = (flip - put_cen) / put_ch if put_ch > 0 else 0.35

    if not above.empty:
        ca = above.sort_values("strike").copy()
        tot = ca["call_gex"].sum()
        if tot > 0:
            ca["cum"] = ca["call_gex"].cumsum()
            idx = (ca["cum"] >= tot * 0.50).idxmax()
            call_med = ca.loc[idx, "strike"]
        else:
            call_med = call_fixed
    else:
        call_med = call_fixed

    if not below.empty:
        pb = below.sort_values("strike", ascending=False).copy()
        tot = pb["put_gex"].sum()
        if tot > 0:
            pb["cum"] = pb["put_gex"].cumsum()
            idx = (pb["cum"] >= tot * 0.50).idxmax()
            put_med = pb.loc[idx, "strike"]
        else:
            put_med = put_fixed
    else:
        put_med = put_fixed

    call_med_ratio = (call_med - flip) / call_ch if call_ch > 0 else 0.30
    put_med_ratio = (flip - put_med) / put_ch if put_ch > 0 else 0.35

    return {
        "gex_flip": flip, "call_wall": call_wall, "put_wall": put_wall,
        "call_channel": call_ch, "put_channel": put_ch,
        "call_gravity_fixed": call_fixed, "put_gravity_fixed": put_fixed,
        "call_ratio": 0.30, "put_ratio": 0.35,
        "call_centroid": call_cen, "put_centroid": put_cen,
        "call_centroid_ratio": call_cen_ratio, "put_centroid_ratio": put_cen_ratio,
        "call_median": call_med, "put_median": put_med,
        "call_median_ratio": call_med_ratio, "put_median_ratio": put_med_ratio,
    }


# =============================================================================
# GAMMA PIN / FLOOR-CEILING / HEDGE WALLS / K*  (verbatim)
# =============================================================================

def compute_gamma_pin_level(gex_df, spot_price, vol_trigger, gravity):
    in_positive_gamma = (vol_trigger is not None and spot_price > vol_trigger)
    max_gex_idx = gex_df["total_gamma"].idxmax()
    max_gex_strike = gex_df.loc[max_gex_idx, "strike"]
    max_gex_value = gex_df.loc[max_gex_idx, "total_gamma"]
    max_oi_idx = gex_df["total_oi"].idxmax()
    max_oi_strike = gex_df.loc[max_oi_idx, "strike"]
    max_oi_value = gex_df.loc[max_oi_idx, "total_oi"]

    call_gravities = [gravity["call_gravity_fixed"], gravity["call_centroid"], gravity["call_median"]]
    put_gravities = [gravity["put_gravity_fixed"], gravity["put_centroid"], gravity["put_median"]]
    call_spread = max(call_gravities) - min(call_gravities)
    put_spread = max(put_gravities) - min(put_gravities)
    gravity_consensus = (call_spread < 200) and (put_spread < 200)
    gex_oi_gap = abs(max_gex_strike - max_oi_strike)
    strong_convergence = gex_oi_gap <= 100

    if strong_convergence:
        pin_strike_exact = max_gex_strike
    else:
        total_weight = max_gex_value + max_oi_value + 1e-9
        pin_strike_exact = (max_gex_value / total_weight * max_gex_strike
                            + max_oi_value / total_weight * max_oi_strike)

    pin_strike = round(pin_strike_exact / 50) * 50
    pin_distance = pin_strike - spot_price
    pin_distance_pct = pin_distance / spot_price * 100

    score = 0
    score += 40 if in_positive_gamma else 0
    score += 25 if strong_convergence else max(0, 25 - int(gex_oi_gap / 20))
    score += 20 if gravity_consensus else max(0, 20 - int((call_spread + put_spread) / 40))
    score += 15 if abs(pin_distance_pct) < 1.0 else max(0, 15 - int(abs(pin_distance_pct) * 5))

    if score >= 70:
        strength_label = "STRONG PIN"
    elif score >= 45:
        strength_label = "MODERATE PIN"
    elif score >= 25:
        strength_label = "WEAK PIN"
    else:
        strength_label = "NO PIN  (negative gamma — trending market)"

    return {
        "pin_strike": pin_strike, "pin_strike_exact": pin_strike_exact,
        "pin_distance": pin_distance, "pin_distance_pct": pin_distance_pct,
        "max_gex_strike": max_gex_strike, "max_oi_strike": max_oi_strike,
        "gex_oi_gap": gex_oi_gap, "in_positive_gamma": in_positive_gamma,
        "strong_convergence": strong_convergence, "gravity_consensus": gravity_consensus,
        "strength_score": score, "strength_label": strength_label,
        "call_gravity_spread": call_spread, "put_gravity_spread": put_spread,
    }


def compute_floor_ceiling(gex_df, spot_price, sigma_upper=None, sigma_lower=None):
    above = gex_df[gex_df["strike"] > spot_price].copy()
    below = gex_df[gex_df["strike"] < spot_price].copy()

    ceil_above = above[above["net_gex"] > 0].copy()
    if not ceil_above.empty:
        ceil_above["gex_oi_score"] = ceil_above["call_gex"].abs() * ceil_above["call_oi"]
        ceil_primary = ceil_above.loc[ceil_above["gex_oi_score"].idxmax(), "strike"]
        ceil_oi_strike = (ceil_above.loc[ceil_above["call_oi"].idxmax(), "strike"]
                          if ceil_above["call_oi"].sum() > 0 else ceil_primary)
        ceil_gex_strike = (ceil_above.loc[ceil_above["call_gex"].idxmax(), "strike"]
                           if ceil_above["call_gex"].sum() > 0 else ceil_primary)
        ceil_spread = (max(ceil_primary, ceil_oi_strike, ceil_gex_strike)
                       - min(ceil_primary, ceil_oi_strike, ceil_gex_strike))
        ceil_strength = ("STRONG" if ceil_spread <= 100 else
                         "MODERATE" if ceil_spread <= 200 else "WEAK")
        ceil_used_fallback = False
    else:
        ceil_primary = sigma_upper if sigma_upper else spot_price * 1.02
        ceil_oi_strike = ceil_gex_strike = ceil_primary
        ceil_spread = 0
        ceil_strength = "WEAK (sigma fallback)"
        ceil_used_fallback = True

    floor_below = below[below["net_gex"] > 0].copy()
    if not floor_below.empty:
        floor_below["gex_oi_score"] = floor_below["put_gex"].abs() * floor_below["put_oi"]
        floor_primary = floor_below.loc[floor_below["gex_oi_score"].idxmax(), "strike"]
        floor_oi_strike = (floor_below.loc[floor_below["put_oi"].idxmax(), "strike"]
                           if floor_below["put_oi"].sum() > 0 else floor_primary)
        floor_gex_strike = (floor_below.loc[floor_below["put_gex"].idxmax(), "strike"]
                            if floor_below["put_gex"].sum() > 0 else floor_primary)
        floor_spread = (max(floor_primary, floor_oi_strike, floor_gex_strike)
                        - min(floor_primary, floor_oi_strike, floor_gex_strike))
        floor_strength = ("STRONG" if floor_spread <= 100 else
                          "MODERATE" if floor_spread <= 200 else "WEAK")
        floor_used_fallback = False
    else:
        floor_primary = sigma_lower if sigma_lower else spot_price * 0.98
        floor_oi_strike = floor_gex_strike = floor_primary
        floor_spread = 0
        floor_strength = "WEAK (sigma fallback)"
        floor_used_fallback = True

    def _r(v):
        return round(v / 50) * 50

    return {
        "ceiling": _r(ceil_primary), "ceiling_exact": ceil_primary,
        "ceiling_oi": _r(ceil_oi_strike), "ceiling_gex": _r(ceil_gex_strike),
        "ceiling_distance": ceil_primary - spot_price,
        "ceiling_distance_pct": (ceil_primary - spot_price) / spot_price * 100,
        "ceiling_spread": ceil_spread, "ceiling_strength": ceil_strength,
        "ceiling_fallback": ceil_used_fallback,
        "floor": _r(floor_primary), "floor_exact": floor_primary,
        "floor_oi": _r(floor_oi_strike), "floor_gex": _r(floor_gex_strike),
        "floor_distance": floor_primary - spot_price,
        "floor_distance_pct": (floor_primary - spot_price) / spot_price * 100,
        "floor_spread": floor_spread, "floor_strength": floor_strength,
        "floor_fallback": floor_used_fallback,
        "trading_range": _r(ceil_primary) - _r(floor_primary),
        "range_pct": (_r(ceil_primary) - _r(floor_primary)) / spot_price * 100,
    }


def compute_hedge_wall(gex_df, spot_price, call_wall):
    above_cw = gex_df[gex_df["strike"] > call_wall].copy()
    if above_cw.empty:
        return None
    above_cw = above_cw.sort_values("strike").reset_index(drop=True)
    above_cw["gex_oi_score"] = above_cw["call_gex"] * above_cw["call_oi"]
    alpha = 5.0
    above_cw["dist_weight"] = np.exp(-alpha * (above_cw["strike"] - call_wall) / (call_wall + 1e-9))
    raw_vanna = above_cw["call_oi"] * above_cw["call_gex"]
    above_cw["vanna_norm"] = raw_vanna / (raw_vanna.max() + 1e-9)
    above_cw["hedge_pressure"] = (above_cw["gex_oi_score"] * above_cw["dist_weight"]
                                  * (1.0 + above_cw["vanna_norm"]))
    hw_idx = above_cw["hedge_pressure"].idxmax()
    hw_exact = above_cw.loc[hw_idx, "strike"]
    hw = round(hw_exact / 50) * 50
    hw_pressure = above_cw.loc[hw_idx, "hedge_pressure"]
    total_pressure = above_cw["hedge_pressure"].sum()
    pressure_pct = hw_pressure / (total_pressure + 1e-9) * 100
    gap_from_cw = hw - call_wall
    dist_from_spot = hw - spot_price
    gap_label = ("TIGHT" if gap_from_cw <= 100 else "NORMAL" if gap_from_cw <= 250 else "WIDE")
    return {
        "hedge_wall": hw, "hedge_wall_exact": hw_exact,
        "gap_from_call_wall": gap_from_cw, "gap_label": gap_label,
        "distance_from_spot": dist_from_spot,
        "distance_from_spot_pct": dist_from_spot / spot_price * 100,
        "peak_pressure": hw_pressure, "pressure_concentration": pressure_pct,
        "above_cw_df": above_cw,
    }


def compute_downside_hedge_wall(gex_df, spot_price, put_wall):
    below_pw = gex_df[gex_df["strike"] < put_wall].copy()
    if below_pw.empty:
        return None
    below_pw = below_pw.sort_values("strike", ascending=False).reset_index(drop=True)
    below_pw["gex_oi_score"] = below_pw["put_gex"] * below_pw["put_oi"]
    alpha = 5.0
    below_pw["dist_weight"] = np.exp(-alpha * (put_wall - below_pw["strike"]) / (put_wall + 1e-9))
    raw_vanna = below_pw["put_oi"] * below_pw["put_gex"]
    below_pw["vanna_norm"] = raw_vanna / (raw_vanna.max() + 1e-9)
    below_pw["hedge_pressure"] = (below_pw["gex_oi_score"] * below_pw["dist_weight"]
                                  * (1.0 + below_pw["vanna_norm"]))
    dhw_idx = below_pw["hedge_pressure"].idxmax()
    dhw_exact = below_pw.loc[dhw_idx, "strike"]
    dhw = round(dhw_exact / 50) * 50
    dhw_pressure = below_pw.loc[dhw_idx, "hedge_pressure"]
    total_pressure = below_pw["hedge_pressure"].sum()
    pressure_pct = dhw_pressure / (total_pressure + 1e-9) * 100
    gap_from_pw = put_wall - dhw
    dist_from_spot = dhw - spot_price
    gap_label = ("TIGHT" if gap_from_pw <= 100 else "NORMAL" if gap_from_pw <= 250 else "WIDE")
    return {
        "downside_hedge_wall": dhw, "dhw_exact": dhw_exact,
        "gap_from_put_wall": gap_from_pw, "gap_label": gap_label,
        "distance_from_spot": dist_from_spot,
        "distance_from_spot_pct": dist_from_spot / spot_price * 100,
        "peak_pressure": dhw_pressure, "pressure_concentration": pressure_pct,
        "below_pw_df": below_pw,
    }


def find_optimal_strike_K_star(gex_df):
    rows = []
    for _, row in gex_df.iterrows():
        K = row["strike"]
        cg = abs(row["call_gex"])
        pg = abs(row["put_gex"])
        cp = row.get("ce_price", 0)
        pp = row.get("pe_price", 0)
        fwd = cp + K - pp
        fx = abs(pg - cg)
        fmn, fmx = abs(pg - cg), pg + cg
        has = (cp != 0 or pp != 0)
        con = has and not (fmn <= fwd <= fmx)
        rows.append({"strike": K, "call_gex": cg, "put_gex": pg,
                     "forward_price": fwd, "fx": fx,
                     "f_min": fmn, "f_max": fmx, "contradiction": con})
    df = pd.DataFrame(rows)

    # Eligibility mask: a K* candidate must have REAL option prices AND non-trivial
    # gamma. Without this, empty deep-OTM strikes (call_gex≈put_gex≈0) win with a
    # fake-zero fx and drag K* to the edge (e.g. 22,350). K* should sit near the
    # flip where call/put GEX actually balance.
    gtot = df["call_gex"] + df["put_gex"]
    gmax = gtot.max()
    eligible = df[
        (df["call_gex"] > 0) & (df["put_gex"] > 0) &  # both sides have gamma
        (gtot >= 0.05 * gmax if gmax > 0 else False)  # non-trivial magnitude
    ]
    valid = eligible[~eligible["contradiction"]]
    if valid.empty:
        valid = eligible if not eligible.empty else df[~df["contradiction"]]
    if valid.empty:
        valid = df
    best = valid.loc[valid["fx"].idxmin()]
    return best["strike"], best["forward_price"], df


# =============================================================================
# DATA LAYER
# =============================================================================

@st.cache_resource(show_spinner=False)
def get_tv():
    """Create (and cache) a TvDatafeed client using the no-login method."""
    from tvDatafeed import TvDatafeed
    return TvDatafeed()


@st.cache_data(ttl=30, show_spinner=False)
def fetch_spot(symbol, exchange, _tv, token):
    """Spot = last 1-minute close of NSE:NIFTY from tvdatafeed."""
    from tvDatafeed import Interval
    df = _tv.get_hist(symbol=symbol, exchange=exchange,
                      interval=Interval.in_1_minute, n_bars=5)
    if df is None or df.empty:
        # fall back to daily close if 1-min is empty (market closed)
        df = _tv.get_hist(symbol=symbol, exchange=exchange,
                          interval=Interval.in_daily, n_bars=5)
    if df is None or df.empty:
        raise RuntimeError("tvdatafeed returned no bars for spot.")
    return float(df["close"].iloc[-1]), df.index[-1]


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_vol_from_tv(symbol, exchange, _tv, token):
    """
    Realized vol + fallback IV from tvdatafeed daily bars (replaces yfinance).
      fallback_iv : 60-day log-return std * sqrt(252)
      rv          : 30-day log-return std * sqrt(252)
    """
    from tvDatafeed import Interval
    df = _tv.get_hist(symbol=symbol, exchange=exchange,
                      interval=Interval.in_daily, n_bars=90)
    if df is None or len(df) < 5:
        return 0.20, 0.20
    close = df["close"]
    logret = np.log(close / close.shift(1)).dropna()
    iv = float(logret.tail(60).std() * np.sqrt(252)) if len(logret) >= 5 else 0.20
    rv = float(logret.tail(30).std() * np.sqrt(252)) if len(logret) >= 5 else iv
    return (iv or 0.20), (rv or iv or 0.20)


# Friendly-name -> tvDatafeed.Interval attribute
_TV_INTERVALS = {
    "1 min":  "in_1_minute",
    "5 min":  "in_5_minute",
    "15 min": "in_15_minute",
    "1 hour": "in_1_hour",
    "1 day":  "in_daily",
}


@st.cache_data(ttl=60, show_spinner=False)
def fetch_candles(symbol, exchange, _tv, interval_name, n_bars, token):
    """OHLC candles for the price view (time-series) from tvdatafeed."""
    from tvDatafeed import Interval
    iv = getattr(Interval, _TV_INTERVALS.get(interval_name, "in_15_minute"))
    df = _tv.get_hist(symbol=symbol, exchange=exchange, interval=iv, n_bars=n_bars)
    if df is None or df.empty:
        raise RuntimeError("tvdatafeed returned no candles.")
    return df


def _nse_session():
    s = requests.Session()
    s.headers.update(NSE_HEADERS)
    s.get("https://www.nseindia.com", timeout=10)
    s.get("https://www.nseindia.com/option-chain", timeout=10)
    return s


def _to_v3_expiry(exp):
    """Convert an NSE expiry string ('07-Jul-2026') to v3 URL format ('07-July-2026')."""
    for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(exp, fmt).strftime("%d-%B-%Y")
        except (ValueError, TypeError):
            continue
    return exp


def next_n_days_expiries(n=20):
    """Next n calendar days as NSE expiry strings ('DD-Mon-YYYY') for the manual picker."""
    today = datetime.now().date()
    return [(today + timedelta(days=i)).strftime("%d-%b-%Y") for i in range(n)]


def default_expiry_index(options):
    """Index of the nearest Tuesday (usual NIFTY weekly expiry) in options; else 0."""
    for i, s in enumerate(options):
        try:
            if datetime.strptime(s, "%d-%b-%Y").weekday() == 1:  # Tuesday
                return i
        except ValueError:
            continue
    return 0


def _expiry_candidates(days_ahead=50):
    """Upcoming Tuesdays & Thursdays — backup seeds covering current & legacy NIFTY expiry regimes."""
    today = datetime.now().date()
    return [today + timedelta(days=i) for i in range(days_ahead)
            if (today + timedelta(days=i)).weekday() in (1, 3)]


def _expiries_from_json(data):
    return (data or {}).get("records", {}).get("expiryDates", []) or []


def _seed_expiry_from_v3(session):
    """
    v3 with no expiry returns a default chain but NO expiryDates list. Grab any
    expiry string from its rows to use as a seed for the real (full-list) call.
    """
    try:
        r = session.get(NSE_OC_V3, timeout=15)
        if r.status_code != 200:
            return None
        for rec in r.json().get("records", {}).get("data", []):
            for leg in ("CE", "PE"):
                e = rec.get(leg, {}).get("expiryDate")
                if e:
                    return e
    except Exception:
        pass
    return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_expiry_list(token):
    """
    Get the full NIFTY expiry list from the v3 endpoint (the only NSE endpoint
    that still works for this and that carries `records.expiryDates`).

    A v3 call *with* a valid expiry returns the complete expiryDates list, so we
    only need one good expiry to seed it: first from v3's default chain, then
    from upcoming Tue/Thu dates as backup.
    """
    last_err = None
    sess = _nse_session()

    seeds = []
    seed = _seed_expiry_from_v3(sess)
    if seed:
        seeds.append(seed)
    seeds += [d.strftime("%d-%b-%Y") for d in _expiry_candidates()]

    for seed in seeds:
        try:
            r = sess.get(f"{NSE_OC_V3}&expiry={_to_v3_expiry(seed)}", timeout=15)
            if r.status_code != 200:
                continue
            exps = _expiries_from_json(r.json())
            if exps:
                return exps
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Could not resolve NIFTY expiries from NSE v3. Last error: {last_err}")


@st.cache_data(ttl=60, show_spinner=False)
def fetch_option_chain_v3(expiry, token):
    """
    NIFTY option chain for a single expiry from the NSE v3 endpoint:
      api/option-chain-v3?type=Indices&symbol=NIFTY&expiry=07-July-2026
    """
    url = f"{NSE_OC_V3}&expiry={_to_v3_expiry(expiry)}"
    last_err = None
    for _ in range(3):
        try:
            s = _nse_session()
            r = s.get(url, timeout=15)
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}"
                time.sleep(1.0)
                continue
            ct = r.headers.get("Content-Type", "")
            if "json" not in ct.lower() and not r.text.lstrip().startswith("{"):
                last_err = "NSE returned non-JSON (likely a bot/IP block page)"
                time.sleep(1.0)
                continue
            data = r.json()
            if data and "records" in data:
                return data
            last_err = "response had no 'records'"
        except Exception as e:
            last_err = e
            time.sleep(1.0)
    raise RuntimeError(f"NSE v3 option-chain fetch failed: {last_err}")


def parse_options(json_data, field_mappings, expiry_filter=None):
    calls, puts = [], []
    if json_data and "records" in json_data and "data" in json_data["records"]:
        for record in json_data["records"]["data"]:
            if "CE" in record:
                opt = record["CE"]
                if expiry_filter and opt.get("expiryDate") != expiry_filter:
                    pass
                else:
                    row = {"expiryDate": opt.get("expiryDate")}
                    for nf, sf in field_mappings.items():
                        row[sf] = opt.get(nf, 0)
                    calls.append(row)
            if "PE" in record:
                opt = record["PE"]
                if expiry_filter and opt.get("expiryDate") != expiry_filter:
                    pass
                else:
                    row = {"expiryDate": opt.get("expiryDate")}
                    for nf, sf in field_mappings.items():
                        row[sf] = opt.get(nf, 0)
                    puts.append(row)
    return calls, puts


def _clean_df(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ["strike", "impliedVolatility", "openInterest", "lastPrice"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def parse_expiry_to_years(expiry_str):
    """NSE expiry strings look like '31-Jul-2026'. Return time-to-expiry in years."""
    dt = None
    for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%d-%m-%Y"):
        try:
            dt = datetime.strptime(expiry_str, fmt)
            break
        except (ValueError, TypeError):
            continue
    if dt is None:
        dt = pd.to_datetime(expiry_str, errors="coerce").to_pydatetime()
    # expiry is at 15:30 IST; keep it simple with end-of-day
    dt = dt.replace(hour=15, minute=30)
    years = (dt - datetime.now()).total_seconds() / (365 * 24 * 3600)
    dte_days = (dt.date() - datetime.now().date()).days
    return max(MIN_T, years), dt, dte_days


# =============================================================================
# VISUALIZATION  (from create_gamma_visualization — refactored to RETURN fig)
# =============================================================================

def build_gamma_figure(gex_df, spot, K_star, F_at_Kstar, ticker, exp_str,
                       exp_label, time_str, min_strike, max_strike,
                       sigma_upper=None, sigma_lower=None, vol_trigger=None,
                       gravity=None, pin=None, fc=None, hw=None, dhw=None):
    BG, FG = "white", "#111111"
    GRID, SPINE = "#cccccc", "#aaaaaa"
    C_SELL, C_BUY = "#cc0000", "#007700"
    C_NET, C_TOTAL = "#b8860b", "#cc6600"
    C_GDET, C_SPOT = "#8B4513", "#c400c4"
    C_KSTAR, C_FWD = "#0077bb", "#444444"
    C_SU, C_SD = "#3a7d00", "#b8860b"
    C_VT, C_NF = "#0066cc", "#7b2d8b"
    C_PIN = "#FF6600"
    C_CEIL, C_FLOOR = "#D35400", "#148F77"
    C_COLOR = "#FF1493"

    fig, ax = plt.subplots(figsize=(20, 11))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    S = gex_df["strike"].values
    if len(S) > 3:
        sg = 2.0
        sell_s = gaussian_filter1d(np.abs(gex_df["sell_gamma"].values), sigma=sg)
        buy_s = gaussian_filter1d(np.abs(gex_df["buy_gamma"].values), sigma=sg)
        net_s = gaussian_filter1d(gex_df["net_gex"].values, sigma=sg)
        tot_s = gaussian_filter1d(gex_df["total_gamma"].values, sigma=sg)
        gd_s = gaussian_filter1d(gex_df["gamma_detonation"].values, sigma=sg)
        color_s = (gaussian_filter1d(gex_df["total_color"].values, sigma=sg)
                   if "total_color" in gex_df.columns else np.zeros_like(net_s))
    else:
        sell_s = np.abs(gex_df["sell_gamma"].values)
        buy_s = np.abs(gex_df["buy_gamma"].values)
        net_s = gex_df["net_gex"].values
        tot_s = gex_df["total_gamma"].values
        gd_s = gex_df["gamma_detonation"].values
        color_s = (gex_df["total_color"].values
                   if "total_color" in gex_df.columns else np.zeros_like(net_s))

    ax.plot(S, sell_s, color=C_SELL, lw=2.5, label="Sell Gamma", alpha=0.9)
    ax.plot(S, buy_s, color=C_BUY, lw=2.5, label="Buy Gamma", alpha=0.9)
    ax.plot(S, net_s, color=C_NET, lw=3.5, label="Neutral Gamma", alpha=0.95)
    ax.plot(S, tot_s, color=C_TOTAL, lw=2.5, label="Total Gamma", alpha=0.9)

    ax2 = ax.twinx()
    ax2.set_facecolor(BG)
    ax2.plot(S, gd_s, color=C_GDET, lw=2.0, ls="--", label="Gamma Detonation", alpha=0.85)
    ax2.set_ylabel("Gamma Detonation", color=C_GDET, fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=C_GDET, colors=C_GDET)

    ax3 = ax.twinx()
    ax3.set_facecolor(BG)
    ax3.spines["right"].set_position(("outward", 60))
    ax3.plot(S, color_s, color=C_COLOR, lw=2.0, ls=":", label="DgammaDtime (Color)", alpha=0.90)
    ax3.axhline(y=0, color=C_COLOR, ls=":", lw=0.8, alpha=0.35)
    ax3.fill_between(S, 0, color_s, where=(color_s < 0), facecolor=C_COLOR, alpha=0.10, interpolate=True)
    ax3.fill_between(S, 0, color_s, where=(color_s >= 0), facecolor=C_COLOR, alpha=0.20, interpolate=True)
    ax3.set_ylabel("DgammaDtime — Color (per day)", color=C_COLOR, fontsize=11, fontweight="bold")
    ax3.tick_params(axis="y", labelcolor=C_COLOR, colors=C_COLOR)
    ax3.spines["right"].set_edgecolor(C_COLOR)

    if len(S) > 2:
        peaks = [(S[i], color_s[i]) for i in range(1, len(color_s) - 1)
                 if abs(color_s[i]) > abs(color_s[i - 1]) and abs(color_s[i]) > abs(color_s[i + 1])]
        for i, (sk, cv) in enumerate(sorted(peaks, key=lambda x: abs(x[1]), reverse=True)[:2]):
            ax3.plot(sk, cv, marker="o", markersize=9, color=C_COLOR,
                     markeredgecolor="white", markeredgewidth=1.2, zorder=15,
                     label="Color Peak" if i == 0 else "")
            ax3.text(sk, cv, f"K={sk:.0f}", ha="center", va="bottom" if cv >= 0 else "top",
                     color=C_COLOR, fontsize=8, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", fc="#ffe6f2", ec=C_COLOR, lw=1.1, alpha=0.9),
                     zorder=16)

    ymax = max(float(np.max(sell_s)), float(np.max(buy_s)), float(np.max(tot_s)))
    ylbl = ymax * 0.92

    diff = buy_s - net_s
    nf_done = False
    for i in range(len(diff) - 1):
        if diff[i] * diff[i + 1] < 0 and (diff[i + 1] - diff[i]) != 0:
            xc = S[i] - diff[i] * (S[i + 1] - S[i]) / (diff[i + 1] - diff[i])
            ax.axvline(x=xc, color=C_NF, ls="-.", lw=1.8, alpha=0.85, zorder=14,
                       label="Neutral/Buy Flip" if not nf_done else "")
            ax.text(xc, ylbl, f"K={xc:.0f}", rotation=90, va="top", ha="right",
                    color=C_NF, fontsize=8, fontweight="bold")
            nf_done = True

    net_spot = float(np.interp(spot, S, net_s))
    ax.axvline(x=spot, color=C_SPOT, ls="--", lw=2.5, alpha=0.95, zorder=20, label=f"Spot Rs{spot:.0f}")
    ax.text(spot, ymax * 0.87, f"Spot=Rs{spot:.0f}\nNetGEX={net_spot:.2e}",
            va="top", ha="center", color="white", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc=C_SPOT, ec="white", lw=1.5, alpha=0.93), zorder=21)

    ax.axvline(x=K_star, color=C_KSTAR, ls=":", lw=2.0, alpha=0.80, label=f"K* Rs{K_star:.0f}")
    ax.axvline(x=F_at_Kstar, color=C_FWD, ls="-.", lw=1.8, alpha=0.70, label=f"Fwd Rs{F_at_Kstar:.2f}")

    if sigma_upper and sigma_lower:
        ax.axvline(x=sigma_upper, color=C_SU, ls=":", lw=1.8, alpha=0.90, zorder=13,
                   label=f"+1s Rs{round(sigma_upper)}")
        ax.text(sigma_upper, ylbl, f"K={round(sigma_upper)}", rotation=90, va="top", ha="right",
                color=C_SU, fontsize=8)
        ax.axvline(x=sigma_lower, color=C_SD, ls=":", lw=1.8, alpha=0.90, zorder=13,
                   label=f"-1s Rs{round(sigma_lower)}")
        ax.text(sigma_lower, ylbl, f"K={round(sigma_lower)}", rotation=90, va="top", ha="right",
                color=C_SD, fontsize=8)

    if vol_trigger is not None:
        ax.axvline(x=vol_trigger, color=C_VT, ls="-", lw=2.5, alpha=0.92, zorder=15,
                   label=f"Vol Trigger Rs{vol_trigger:.0f}")
        ax.text(vol_trigger, ymax * 0.62, f"VOL TRIGGER\nRs{vol_trigger:.0f}",
                va="top", ha="center", color="white", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc=C_VT, ec="white", lw=1.2, alpha=0.92), zorder=16)
        ax.axvspan(min_strike, vol_trigger, alpha=0.04, color="red")
        ax.axvspan(vol_trigger, max_strike, alpha=0.04, color="green")

    if gravity is not None:
        ax.set_xlim(min_strike, max_strike)

        def _vl(x_val, text, lc, fc_box, font_c, y_frac, ls="--", lw=2.0, zl=12, za=18):
            ax.axvline(x=x_val, color=lc, linestyle=ls, linewidth=lw, alpha=0.90, zorder=zl)
            xlo, xhi = ax.get_xlim()
            x_frac = (x_val - xlo) / (xhi - xlo + 1e-9)
            ylo, yhi = ax.get_ylim()
            tip_y = ylo + 0.14 * (yhi - ylo)
            ax.annotate(text, xy=(x_val, tip_y), xytext=(x_frac, y_frac),
                        xycoords=("data", "data"), textcoords="axes fraction",
                        fontsize=8, fontweight="bold", color=font_c, ha="center", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.30", fc=fc_box, ec=lc, lw=1.6, alpha=0.96),
                        arrowprops=dict(arrowstyle="-|>", color=lc, lw=1.4,
                                        connectionstyle="arc3,rad=0.0"), zorder=za)

        _vl(gravity["put_wall"], f"PUT WALL\nRs{gravity['put_wall']:.0f}",
            "#1a6bbf", "#d6e8fb", "#0a3d70", y_frac=0.90, ls="--", lw=2.0, zl=12, za=19)
        _vl(gravity["call_wall"], f"CALL WALL\nRs{gravity['call_wall']:.0f}",
            "#e03000", "#ffe5df", "#7a1500", y_frac=0.90, ls="--", lw=2.0, zl=12, za=19)
        _vl(gravity["put_gravity_fixed"],
            f"PUT FIXED\nRs{gravity['put_gravity_fixed']:.0f}  ({gravity['put_ratio']:.0%})",
            "#0044aa", "#b8d0f5", "#001f5b", y_frac=0.76, ls="-", lw=2.2, zl=15, za=20)
        _vl(gravity["call_gravity_fixed"],
            f"CALL FIXED\nRs{gravity['call_gravity_fixed']:.0f}  ({gravity['call_ratio']:.0%})",
            "#cc2200", "#fdd5cf", "#6b0a00", y_frac=0.76, ls="-", lw=2.2, zl=15, za=20)
        _vl(gravity["put_centroid"],
            f"PUT CENTROID\nRs{gravity['put_centroid']:.0f}  ({gravity['put_centroid_ratio']:.0%})",
            "#006699", "#cce8f4", "#003355", y_frac=0.60, ls="-", lw=2.2, zl=15, za=20)
        _vl(gravity["call_centroid"],
            f"CALL CENTROID\nRs{gravity['call_centroid']:.0f}  ({gravity['call_centroid_ratio']:.0%})",
            "#aa3300", "#fce8d8", "#551500", y_frac=0.60, ls="-", lw=2.2, zl=15, za=20)
        _vl(gravity["put_median"],
            f"PUT MEDIAN\nRs{gravity['put_median']:.0f}  ({gravity['put_median_ratio']:.0%})",
            "#005577", "#c2dce8", "#002233", y_frac=0.46, ls="-.", lw=2.2, zl=15, za=20)
        _vl(gravity["call_median"],
            f"CALL MEDIAN\nRs{gravity['call_median']:.0f}  ({gravity['call_median_ratio']:.0%})",
            "#882200", "#f5cfc0", "#3a0a00", y_frac=0.46, ls="-.", lw=2.2, zl=15, za=20)

        for lbl, lc, ls in [("Walls (peak GEX)", "#1a6bbf", "--"),
                            ("Fixed Ratio [30/35%]", "#0044aa", "-"),
                            ("Centroid [GEX-weighted avg]", "#006699", "-"),
                            ("Median [Cum. GEX 50%]", "#005577", "-.")]:
            ax.plot([], [], color=lc, ls=ls, lw=2.0, label=lbl)

    if pin is not None:
        ax.axvline(x=pin["pin_strike"], color=C_PIN, ls="-", lw=3.5, alpha=0.97, zorder=30,
                   label=f"PIN Rs{pin['pin_strike']:.0f}")
        ax.axvspan(pin["pin_strike"] - 50, pin["pin_strike"] + 50, alpha=0.08, color=C_PIN, zorder=5)
        ax.text(pin["pin_strike"], ymax * 0.98,
                f"PIN LEVEL\nRs{pin['pin_strike']:.0f}\n{pin['strength_label']}\nScore: {pin['strength_score']}/100",
                va="top", ha="center", color="white", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", fc=C_PIN, ec="white", lw=2.0, alpha=0.97), zorder=31)

    if fc is not None:
        ax.axvline(x=fc["ceiling"], color=C_CEIL, ls="--", lw=3.0, alpha=0.97, zorder=28,
                   label=f"CEILING Rs{fc['ceiling']:.0f}")
        ax.axvspan(fc["ceiling"] - 25, fc["ceiling"] + 25, alpha=0.10, color=C_CEIL, zorder=4)
        ax.text(fc["ceiling"], ymax * 0.75,
                f"CEILING\nRs{fc['ceiling']:.0f}\n{fc['ceiling_strength']}\n{fc['ceiling_distance_pct']:+.2f}%",
                va="top", ha="center", color="white", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.45", fc=C_CEIL, ec="white", lw=1.8, alpha=0.95), zorder=29)
        ax.axvline(x=fc["floor"], color=C_FLOOR, ls="--", lw=3.0, alpha=0.97, zorder=28,
                   label=f"FLOOR Rs{fc['floor']:.0f}")
        ax.axvspan(fc["floor"] - 25, fc["floor"] + 25, alpha=0.10, color=C_FLOOR, zorder=4)
        ax.text(fc["floor"], ymax * 0.55,
                f"FLOOR\nRs{fc['floor']:.0f}\n{fc['floor_strength']}\n{fc['floor_distance_pct']:+.2f}%",
                va="top", ha="center", color="white", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.45", fc=C_FLOOR, ec="white", lw=1.8, alpha=0.95), zorder=29)

    if hw is not None:
        ax.axvline(x=hw["hedge_wall"], color="#8B008B", ls="--", lw=2.6, alpha=0.95, zorder=27,
                   label=f"Upside HW Rs{hw['hedge_wall']:.0f}")
        ax.text(hw["hedge_wall"], ymax * 0.35,
                f"UPSIDE HW\nRs{hw['hedge_wall']:.0f}\n[{hw['gap_label']}]",
                va="top", ha="center", color="white", fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", fc="#8B008B", ec="white", lw=1.6, alpha=0.95), zorder=28)
    if dhw is not None:
        ax.axvline(x=dhw["downside_hedge_wall"], color="#006400", ls="--", lw=2.6, alpha=0.95, zorder=27,
                   label=f"Downside HW Rs{dhw['downside_hedge_wall']:.0f}")
        ax.text(dhw["downside_hedge_wall"], ymax * 0.35,
                f"DOWNSIDE HW\nRs{dhw['downside_hedge_wall']:.0f}\n[{dhw['gap_label']}]",
                va="top", ha="center", color="white", fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", fc="#006400", ec="white", lw=1.6, alpha=0.95), zorder=28)

    ax.fill_between(S, 0, net_s, where=(net_s < 0), facecolor="#ff4444", alpha=0.12,
                    interpolate=True, label="Negative GEX")
    ax.fill_between(S, 0, net_s, where=(net_s > 0), facecolor="#22aa22", alpha=0.12,
                    interpolate=True, label="Positive GEX")

    if np.any(net_s > 0):
        pos_idx = int(np.argmax(net_s))
        ax.plot(S[pos_idx], net_s[pos_idx], marker="D", markersize=11, color="#00AA44",
                markeredgecolor="white", markeredgewidth=1.5, zorder=40, label="Peak +ve Net GEX")
    if np.any(net_s < 0):
        neg_idx = int(np.argmin(net_s))
        ax.plot(S[neg_idx], net_s[neg_idx], marker="D", markersize=11, color="#DD0033",
                markeredgecolor="white", markeredgewidth=1.5, zorder=40, label="Peak -ve Net GEX")

    tks = np.arange(min_strike, max_strike + 1, 50)
    ax.set_xticks(tks)
    ax.set_xticklabels([f"{s:.0f}" for s in tks], rotation=90, fontsize=7, color=FG)
    ax.tick_params(axis="both", colors=FG, labelcolor=FG)
    ax.axhline(y=0, color="#555555", ls="-", alpha=0.55, lw=1.2)
    ax.set_xlabel("Strike Price (K)", fontsize=12, fontweight="bold", color=FG)
    ax.set_ylabel("Gamma Density", fontsize=12, fontweight="bold", color=FG)
    ax.set_title(
        f"Gamma Density | Gravity (Fixed/Centroid/Median) | Pin | Floor/Ceil | "
        f"Upside HW | Downside HW | Color\n{ticker} | Expiry: {exp_str} ({exp_label}) | {time_str}",
        fontsize=12, fontweight="bold", color=FG)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()
    ax.legend(h1 + h2 + h3, l1 + l2 + l3, loc="upper right", framealpha=0.92,
              fontsize=7, facecolor="#f8f8ff", edgecolor="#aaaaaa", labelcolor=FG)

    ax.grid(True, alpha=0.40, ls="--", lw=0.5, color=GRID)
    ym = max(np.max(sell_s), np.max(buy_s), np.max(tot_s))
    ax.set_ylim(np.min(net_s) * 1.2, ym * 1.2)
    for sp in list(ax.spines.values()) + list(ax2.spines.values()):
        sp.set_edgecolor(SPINE)
    fig.tight_layout()
    return fig


# =============================================================================
# PRICE VIEW  — VS3D forward-simulation net-GEX gradient (decays to expiry)
# =============================================================================
# The real vs3d Gradient Chart: each time column is genuinely recomputed with
# Black-Scholes at T = time from that column to expiry, so the net-GEX seam
# SHARPENS as it approaches expiry (T -> 0). Fields:
#   grid  : price (y) × time (x, first candle -> expiry)
#   value : Σ_legs  sign · OI · bs_gamma(P, K, T, iv) · 100 · P
#   IV    : each strike's own IV from the NSE chain (fallback IV otherwise)
#   weight: OI (NSE updates OI intraday)
# Normalized by 92nd percentile (charm/gamma blow up as T->0; percentile clip
# keeps it readable), diverging GEX colormap with a black seam at zero.

from matplotlib.colors import LinearSegmentedColormap

# GEX colormap — near-zero is DARK-COLORED (not black); only a thin seam stays
# near-black. Strong +GEX -> bright green, strong -GEX -> bright red. This kills
# the dead black space while keeping the flip readable as a thin seam.
_GEX_CMAP = LinearSegmentedColormap.from_list("vs3d_gex", [
    (0.00, (0.95, 0.15, 0.12)),   # strong -GEX  -> bright red
    (0.30, (0.55, 0.05, 0.04)),   # mid red
    (0.46, (0.18, 0.02, 0.02)),   # faint -GEX   -> dark red (NOT black)
    (0.50, (0.02, 0.02, 0.02)),   # thin seam    -> near-black flip line
    (0.54, (0.02, 0.16, 0.04)),   # faint +GEX   -> dark green (NOT black)
    (0.70, (0.06, 0.50, 0.14)),   # mid green
    (1.00, (0.15, 0.95, 0.30)),   # strong +GEX  -> bright green
])


def build_forward_gradient_figure(candles, gex_df, spot, vol_trigger, levels,
                                  title, band_pct=4.0, n_price=400, smooth=0.6,
                                  cone_mode=True, cone_gain=4.5):
    """
    Net-GEX field behind spot candles. y = price, green = +ve net GEX (dampening),
    red = -ve (amplifying), black seam = the flip / vol trigger.
      cone_mode=False : flat horizontal color-bands (net GEX painted at each price)
      cone_mode=True  : vs3d tanh "cone" — each price level reaches horizontally in
                        proportion to |net GEX|, tails coming in from the right edge,
                        giving the tapering cone from the strike-view profile.
    Per-strike structure (walls, pin) preserved (light smoothing only).
    """
    import matplotlib.dates as mdates

    # ---- Net GEX per strike -> price grid (minimal smoothing, keep the bumps) ----
    ks = gex_df["strike"].to_numpy(dtype=float)
    net = gex_df["net_gex"].to_numpy(dtype=float)
    order = np.argsort(ks)
    ks, net = ks[order], net[order]

    lo = spot * (1 - band_pct / 100.0)
    hi = spot * (1 + band_pct / 100.0)
    lvl = [p for _, p, _, _ in levels if p is not None and np.isfinite(p)]
    if lvl:
        lo = min(lo, min(lvl)); hi = max(hi, max(lvl))
    # clamp to the strikes we actually have so we don't paint flat tails forever
    lo = max(lo, ks.min()); hi = min(hi, ks.max())
    pg = np.linspace(lo, hi, n_price)

    prof = np.interp(pg, ks, net, left=0.0, right=0.0)
    if smooth and smooth > 0:
        prof = gaussian_filter1d(prof, max(0.3, n_price * (smooth / 100.0)))

    # ---- normalize about ZERO so the seam is exactly at net GEX = 0 (the flip) ----
    scale = np.percentile(np.abs(prof), 98) or 1.0
    col = np.clip(prof / scale, -1.0, 1.0)           # -1..1, 0 -> black seam

    if cone_mode:
        # vs3d field_from_profile: each price row "reaches" across x in proportion
        # to |net GEX| -> tapering cone. A baseline tint floor keeps every row
        # colored on its sign side (green above flip / red below) so there is no
        # dead black space — the cone just brightens into the walls.
        n_x = 360
        mag = np.abs(col)                            # 0..1 reach per price row
        xs = np.linspace(0.0, 1.0, n_x)
        reach = np.clip(np.tanh(cone_gain * (mag[:, None] - xs[None, :])), 0.0, 1.0)
        base = 0.20                                  # regime tint floor
        intensity = base + (1.0 - base) * reach      # base..1
        field = np.sign(col)[:, None] * intensity    # ±(base..1); sign=0 at seam
        field = field[:, ::-1]                        # tails reach in from the RIGHT
    else:
        # flat color-bands; near-zero now maps to dark colour (not black) via cmap
        field = np.tile(col[:, None], (1, 8))

    # ---- time extent from the candles ----
    idx = candles.index
    xc = mdates.date2num(idx.to_pydatetime())
    x0, x1 = float(xc[0]), float(xc[-1])
    if x1 <= x0:
        x1 = x0 + 1.0

    fig, ax = plt.subplots(figsize=(20, 10))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.imshow(field, origin="lower", extent=[x0, x1, pg[0], pg[-1]], aspect="auto",
              cmap=_GEX_CMAP, vmin=-1, vmax=1, interpolation="bilinear", zorder=0)

    # ---- candles (width from median bar spacing -> no overlap) ----
    o = candles["open"].to_numpy(dtype=float)
    h = candles["high"].to_numpy(dtype=float)
    l = candles["low"].to_numpy(dtype=float)
    c = candles["close"].to_numpy(dtype=float)
    bw = float(np.median(np.diff(xc))) * 0.7 if len(xc) > 1 else (x1 - x0) / 100.0
    wick_min = float(np.nanmean(h - l)) * 0.02 if len(h) else 0.01
    for i in range(len(xc)):
        up = c[i] >= o[i]
        body = "#f0f0f0" if up else "#151515"
        edge = "#ffffff" if up else "#9a9a9a"
        ax.plot([xc[i], xc[i]], [l[i], h[i]], color=edge, lw=0.7, zorder=5)
        lo_b, hi_b = (o[i], c[i]) if up else (c[i], o[i])
        ax.add_patch(plt.Rectangle((xc[i] - bw / 2, lo_b), bw,
                                   max(hi_b - lo_b, wick_min),
                                   facecolor=body, edgecolor=edge, lw=0.4, zorder=6))

    # ---- spot + flip + level lines, de-collided labels ----
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.axhline(spot, color="white", ls="--", lw=1.3, alpha=0.9, zorder=7)
    if vol_trigger is not None and pg[0] <= vol_trigger <= pg[-1]:
        ax.axhline(vol_trigger, color="#33aaff", ls="-", lw=1.6, alpha=0.95, zorder=7)

    placed = []
    for lbl, price, color, ls in sorted(
            [(a, p, cl, s) for (a, p, cl, s) in levels
             if p is not None and np.isfinite(p) and pg[0] <= p <= pg[-1]],
            key=lambda t: t[1]):
        ax.axhline(price, color=color, ls=ls, lw=1.1, alpha=0.85, zorder=7)
        if all(abs(price - q) > 0.0015 * spot for q in placed):
            ax.text(0.004, price, f"{lbl} {price:,.0f}", transform=trans, va="center",
                    ha="left", fontsize=8, color="white", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc=color, ec="none", alpha=0.85),
                    zorder=8)
            placed.append(price)

    ax.set_ylim(pg[0], pg[-1])
    ax.set_xlim(x0, x1)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
    for lab in ax.get_xticklabels():
        lab.set_rotation(45); lab.set_ha("right"); lab.set_color("white"); lab.set_fontsize(8)
    ax.tick_params(colors="white")
    ax.set_ylabel("Price", color="white", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time", color="white", fontsize=12, fontweight="bold")
    ax.set_title(title, color="white", fontsize=12, fontweight="bold", loc="left")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444444")
    fig.tight_layout()
    return fig


# =============================================================================
# POSITIONING LADDER  — pre-trade support/resistance read (net GEX by strike)
# =============================================================================

def build_positioning_ladder(gex_df, spot, vol_trigger, pin, levels,
                             n_strikes=16):
    """
    Trader's glance view: net GEX as horizontal bars per strike near spot.
      +ve net GEX (green) points RIGHT  = resistance / dampening above
      -ve net GEX (red)   points LEFT   = support / amplifying below
    Spot, flip (vol trigger) and pin marked. Answers "nearest wall each way and
    how strong" without reading the full strike chart.
    """
    d = gex_df.copy()
    d["dist"] = (d["strike"] - spot).abs()
    d = d.nsmallest(n_strikes, "dist").sort_values("strike")
    ks = d["strike"].to_numpy(dtype=float)
    net = d["net_gex"].to_numpy(dtype=float)
    scale = np.percentile(np.abs(net), 95) or 1.0
    xn = net / scale                                   # normalized bar length

    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    colors = ["#1a9e4b" if v >= 0 else "#d1332e" for v in xn]
    ax.barh(ks, xn, height=38, color=colors, edgecolor="white", linewidth=0.5, zorder=3)
    ax.axvline(0, color="#333333", lw=1.2, zorder=2)

    # spot / flip / pin markers
    ax.axhline(spot, color="#c400c4", ls="--", lw=2.0, zorder=4)
    ax.text(1.02, spot, f"Spot {spot:,.0f}", transform=ax.get_yaxis_transform(),
            va="center", ha="left", color="white", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="#c400c4", ec="none"))
    if vol_trigger is not None:
        ax.axhline(vol_trigger, color="#0066cc", ls="-", lw=2.0, zorder=4)
        ax.text(1.02, vol_trigger, f"Flip {vol_trigger:,.0f}",
                transform=ax.get_yaxis_transform(), va="center", ha="left",
                color="white", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="#0066cc", ec="none"))
    if pin is not None:
        ax.axhline(pin["pin_strike"], color="#ff8800", ls=":", lw=2.0, zorder=4)
        ax.text(1.02, pin["pin_strike"], f"Pin {pin['pin_strike']:,.0f}",
                transform=ax.get_yaxis_transform(), va="center", ha="left",
                color="white", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="#ff8800", ec="none"))

    # value labels on the bars
    for k, v in zip(ks, xn):
        ax.text(v + (0.03 if v >= 0 else -0.03), k, f"{k:,.0f}",
                va="center", ha="left" if v >= 0 else "right",
                fontsize=7, color="#333333")

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(ks.min() - 40, ks.max() + 40)
    ax.set_xlabel("← support (−net GEX)      resistance (+net GEX) →",
                  fontsize=11, fontweight="bold")
    ax.set_ylabel("Strike", fontsize=11, fontweight="bold")
    ax.set_title("Positioning ladder — net GEX by strike (near spot)",
                 fontsize=12, fontweight="bold", loc="left")
    ax.set_xticks([])
    ax.grid(True, axis="y", alpha=0.25, ls="--", lw=0.5)
    for sp in ax.spines.values():
        sp.set_edgecolor("#bbbbbb")
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def run_analysis(_raw_json, expiry, spot, fallback_iv, rv, risk_free, strike_lo,
                 strike_hi, token):
    # v3 response is already scoped to the requested expiry — no client-side filter
    calls_list, puts_list = parse_options(_raw_json, FIELD_MAPPINGS, expiry_filter=None)
    calls = _clean_df(calls_list)
    puts = _clean_df(puts_list)
    if calls.empty or puts.empty:
        raise RuntimeError("No option rows for the selected expiry.")

    t_expiry, expiry_dt, dte_days = parse_expiry_to_years(expiry)
    exp_label = "0DTE" if dte_days == 0 else f"{dte_days} days"
    iv_market = fallback_iv

    gex_df = compute_gex_density(calls, puts, spot, t_expiry, risk_free, fallback_iv)
    gex_df["total_gamma"] = gex_df["call_gex"] + gex_df["put_gex"]

    dte = max(dte_days, 1)
    gex_df["gamma_detonation"] = gex_df.apply(
        lambda r: compute_gamma_detonation(r["total_oi"], r["avg_premium"], spot,
                                           rv, iv_market, 0.5, dte), axis=1)

    sig_up, sig_dn = compute_sigma_levels(spot, iv_market, t_expiry, n=1)

    gex_df = gex_df[(gex_df["strike"] >= strike_lo) & (gex_df["strike"] <= strike_hi)]
    gex_df = gex_df[(gex_df["call_gex"] != 0) | (gex_df["put_gex"] != 0)]
    gex_df = gex_df.reset_index(drop=True)
    if gex_df.empty:
        raise RuntimeError("No strikes in the selected range have GEX. Widen the range.")

    vol_trigger = compute_volatility_trigger(gex_df, spot)
    gravity = compute_gravity_centers(gex_df, spot, vol_trigger)
    pin = compute_gamma_pin_level(gex_df, spot, vol_trigger, gravity)
    fc = compute_floor_ceiling(gex_df, spot, sig_up, sig_dn)
    hw = compute_hedge_wall(gex_df, spot, gravity["call_wall"])
    dhw = compute_downside_hedge_wall(gex_df, spot, gravity["put_wall"])
    K_star, F_at_Kstar, _ = find_optimal_strike_K_star(gex_df)

    # --- per-strike legs for the forward-simulation gradient ---
    # OI-weighted (NSE updates OI intraday), per-strike IV (NSE IV is in %, -> decimal),
    # sign = +1 for calls, -1 for puts. Fallback IV when a strike has none.
    def _legs(df, sign):
        d = df[(df["strike"] >= strike_lo) & (df["strike"] <= strike_hi)]
        out = []
        for _, rr in d.iterrows():
            iv = rr["impliedVolatility"]
            iv = iv / 100.0 if iv > 3 else iv          # 12.34% -> 0.1234
            if iv <= 0:
                iv = fallback_iv
            oi = rr["openInterest"]
            if oi > 0 and iv > 0:
                out.append((float(rr["strike"]), float(iv), float(oi), sign))
        return out
    sim_legs = _legs(calls, +1) + _legs(puts, -1)

    return {
        "gex_df": gex_df, "t_expiry": t_expiry, "exp_label": exp_label,
        "iv_market": iv_market, "sig_up": sig_up, "sig_dn": sig_dn,
        "vol_trigger": vol_trigger, "gravity": gravity, "pin": pin,
        "fc": fc, "hw": hw, "dhw": dhw, "K_star": K_star, "F_at_Kstar": F_at_Kstar,
        "sim_legs": sim_legs, "expiry_dt": expiry_dt,
    }


# =============================================================================
# STREAMLIT UI
# =============================================================================

st.set_page_config(page_title="NIFTY GEX Dashboard", page_icon="📊", layout="wide")

st.title("NIFTY Gamma Exposure (GEX) Dashboard")
st.caption("Spot via tvdatafeed (NSE:NIFTY) · Option chain via NSE · "
           "Gravity centres · Pin · Floor/Ceiling · Hedge walls · DgammaDtime")

# ── Sidebar ──────────────────────────────────────────────────────────────────
if "token" not in st.session_state:
    st.session_state["token"] = 0.0

with st.sidebar:
    st.header("Settings")

    st.subheader("Expiry")
    manual_opts = next_n_days_expiries(20)
    try_auto = st.checkbox(
        "Try NSE auto-detect", value=False,
        help="Attempt to fetch NSE's valid expiry list. If it fails (cloud IP "
             "block), the manual next-20-days list below is used instead.")
    expiry_options = manual_opts
    if try_auto:
        try:
            with st.spinner("Auto-detecting expiries…"):
                expiry_options = fetch_expiry_list(st.session_state["token"])
        except Exception as e:
            st.warning(f"Auto-detect failed — using manual list.\n\n{e}")
            expiry_options = manual_opts
    expiry = st.selectbox("Expiry date", expiry_options,
                          index=default_expiry_index(expiry_options))
    st.caption("NIFTY weekly expiry = Tuesday · monthly = last Tuesday. "
               "Pick a valid expiry date.")

    st.subheader("tvdatafeed (spot source)")
    tv_symbol = st.text_input("Symbol", value="NIFTY")
    tv_exchange = st.text_input("Exchange", value="NSE")

    st.subheader("Model")
    risk_free = st.number_input("Risk-free rate", value=RISK_FREE_RATE_DEFAULT,
                                min_value=0.0, max_value=0.20, step=0.005, format="%.3f")

    st.subheader("Price view (candles)")
    candle_interval = st.selectbox("Candle interval",
                                   list(_TV_INTERVALS.keys()), index=2)  # 15 min
    candle_bars = st.slider("Candles (bars)", 30, 500, 150, step=10)

    st.subheader("Refresh")
    if st.button("🔄 Refresh data now", use_container_width=True):
        st.session_state["token"] = time.time()
        st.cache_data.clear()

    auto = st.checkbox("Auto-refresh", value=False)
    interval = st.slider("Auto-refresh every (sec)", 15, 300, 60, disabled=not auto)

token = st.session_state["token"]

if auto:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=interval * 1000, key="auto_refresh")
        token = time.time() // interval
    except Exception:
        st.sidebar.info("Install `streamlit-autorefresh` for auto-refresh; using manual refresh.")

# ── Data acquisition ─────────────────────────────────────────────────────────
try:
    with st.spinner("Connecting to tvdatafeed…"):
        tv = get_tv()
except Exception as e:
    st.error(f"Could not initialise tvdatafeed: {e}")
    st.stop()

col_err = st.container()

try:
    with st.spinner("Fetching spot (NSE:NIFTY) via tvdatafeed…"):
        spot, spot_ts = fetch_spot(tv_symbol, tv_exchange, tv, token)
except Exception as e:
    col_err.error(f"Spot fetch failed: {e}")
    st.stop()

try:
    fallback_iv, rv = fetch_vol_from_tv(tv_symbol, tv_exchange, tv, token)
except Exception:
    fallback_iv, rv = 0.20, 0.20

_nse_note = (
    "\n\nNSE frequently blocks cloud/datacenter IPs (Streamlit Cloud, AWS, GCP). "
    "If you see this on a hosted deploy, run the app locally or route NSE "
    "traffic through a residential/India proxy. The expiry is selectable in the "
    "left sidebar regardless."
)

# ── Strike-range control + spot (expiry is chosen in the sidebar) ────────────
atm = round(spot / 50) * 50
default_lo = int(atm - 2000)
default_hi = int(atm + 2000)
top = st.columns([3, 1])
with top[0]:
    strike_lo, strike_hi = st.slider(
        "Strike range", min_value=int(atm - 6000), max_value=int(atm + 6000),
        value=(default_lo, default_hi), step=50)
with top[1]:
    st.metric("Spot (NSE:NIFTY)", f"{spot:,.2f}",
              help=f"tvdatafeed bar time: {spot_ts}")

# ── Fetch the selected expiry's chain from the v3 endpoint ───────────────────
try:
    with st.spinner(f"Fetching option chain (v3) for {expiry}…"):
        raw = fetch_option_chain_v3(expiry, token)
except Exception as e:
    col_err.error(f"NSE v3 option-chain fetch failed: {e}{_nse_note}")
    st.stop()

# If NSE told us the real valid expiries, surface them and flag a bad pick.
valid_exps = raw.get("records", {}).get("expiryDates", [])
if valid_exps:
    st.caption("NSE valid expiries: " + " · ".join(valid_exps[:8])
               + (" …" if len(valid_exps) > 8 else ""))
    if expiry not in valid_exps:
        st.warning(
            f"'{expiry}' is not an NSE expiry — data may be empty or for a "
            f"different expiry. Pick one of the valid dates above in the sidebar "
            f"(nearest: {valid_exps[0]})."
        )

# ── Run analytics ────────────────────────────────────────────────────────────
try:
    with st.spinner("Computing GEX analytics…"):
        res = run_analysis(raw, expiry, spot, fallback_iv, rv, risk_free,
                           strike_lo, strike_hi, token)
except Exception as e:
    st.error(f"Analysis error: {e}")
    st.stop()

gex_df = res["gex_df"]
gravity, pin, fc, hw, dhw = res["gravity"], res["pin"], res["fc"], res["hw"], res["dhw"]
vol_trigger = res["vol_trigger"]
net_spot = float(np.interp(spot, gex_df["strike"].values, gex_df["net_gex"].values))
zone = ("POSITIVE γ (vol-dampening / pinning)" if net_spot > 1e-6 else
        "NEGATIVE γ (vol-amplifying / trending)" if net_spot < -1e-6 else "NEUTRAL γ")

# ── KPI cards ────────────────────────────────────────────────────────────────
k = st.columns(5)
k[0].metric("Gamma zone @ spot", zone.split(" ")[0], help=zone)
k[1].metric("Vol Trigger (flip)", f"{vol_trigger:,.0f}" if vol_trigger else "N/A",
            delta=f"{'ABOVE' if vol_trigger and spot > vol_trigger else 'BELOW'}" if vol_trigger else None)
k[2].metric("Pin", f"{pin['pin_strike']:,.0f}",
            delta=f"{pin['strength_score']}/100 · {pin['strength_label'].split('(')[0].strip()}")
k[3].metric("K* (optimal)", f"{res['K_star']:,.0f}")
k[4].metric("Net GEX @ spot", f"{net_spot:,.2e}")

k2 = st.columns(6)
k2[0].metric("Call Wall", f"{gravity['call_wall']:,.0f}")
k2[1].metric("Put Wall", f"{gravity['put_wall']:,.0f}")
k2[2].metric("Ceiling", f"{fc['ceiling']:,.0f}", delta=f"{fc['ceiling_distance_pct']:+.2f}%")
k2[3].metric("Floor", f"{fc['floor']:,.0f}", delta=f"{fc['floor_distance_pct']:+.2f}%")
k2[4].metric("Upside HW", f"{hw['hedge_wall']:,.0f}" if hw else "N/A",
             delta=hw["gap_label"] if hw else None)
k2[5].metric("Downside HW", f"{dhw['downside_hedge_wall']:,.0f}" if dhw else "N/A",
             delta=dhw["gap_label"] if dhw else None)

# ── Chart ────────────────────────────────────────────────────────────────────
ist_str = (datetime.now(pytz.utc).astimezone(pytz.timezone("Asia/Kolkata"))
           .strftime("%d-%b-%Y %H:%M IST"))

view = st.radio(
    "View",
    ["Gamma density (strike axis)", "Net-GEX gradient (price × candles)",
     "Positioning ladder"],
    horizontal=True,
)

if view == "Gamma density (strike axis)":
    fig = build_gamma_figure(
        gex_df, spot, res["K_star"], res["F_at_Kstar"],
        f"{tv_exchange}:{tv_symbol}", expiry, res["exp_label"], ist_str,
        strike_lo, strike_hi, sigma_upper=res["sig_up"], sigma_lower=res["sig_dn"],
        vol_trigger=vol_trigger, gravity=gravity, pin=pin, fc=fc, hw=hw, dhw=dhw)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
elif view == "Positioning ladder":
    figl = build_positioning_ladder(gex_df, spot, vol_trigger, pin, None)
    lc, _ = st.columns([2, 1])
    with lc:
        st.pyplot(figl, use_container_width=True)
    plt.close(figl)
    st.caption("Nearest support/resistance at a glance: green bars (right) = +net "
               "GEX resistance above, red bars (left) = −net GEX support below. "
               "Longer bar = stronger wall. Spot, flip and pin marked.")
else:
    levels = [
        ("Spot",        spot,                                         "#c400c4", "--"),
        ("Vol Trigger", vol_trigger,                                  "#0066cc", "-"),
        ("Call Wall",   gravity["call_wall"],                         "#ff5544", "--"),
        ("Put Wall",    gravity["put_wall"],                          "#4488ff", "--"),
        ("Pin",         pin["pin_strike"],                            "#ffaa33", "-"),
        ("Ceiling",     fc["ceiling"],                                "#ff8844", "--"),
        ("Floor",       fc["floor"],                                  "#33cc99", "--"),
        ("Upside HW",   hw["hedge_wall"] if hw else None,             "#cc66ff", "--"),
        ("Downside HW", dhw["downside_hedge_wall"] if dhw else None,  "#66dd66", "--"),
        ("K*",          res["K_star"],                                "#66ccff", ":"),
    ]
    cc1, cc2, cc3 = st.columns(3)
    band = cc1.slider("Price band ±%", 1.0, 6.0, 4.0, step=0.5)
    smooth = cc2.slider("Smoothing (0 = raw per-strike bumps)", 0.0, 3.0, 0.6, step=0.1)
    cone_on = cc3.checkbox("Cone mode (tapering glow)", value=True)
    cone_gain = cc3.slider("Cone gain", 2.0, 8.0, 4.5, step=0.5, disabled=not cone_on)
    try:
        with st.spinner(f"Fetching {candle_bars} × {candle_interval} candles…"):
            candles = fetch_candles(tv_symbol, tv_exchange, tv,
                                    candle_interval, candle_bars, token)
        title = (f"{tv_exchange}:{tv_symbol}  ({candle_interval})  net-GEX gradient  "
                 f"| expiry {expiry}  |  {ist_str}")
        figp = build_forward_gradient_figure(
            candles, gex_df, spot, vol_trigger, levels, title,
            band_pct=band, smooth=smooth, cone_mode=cone_on, cone_gain=cone_gain)
        st.pyplot(figp, use_container_width=True)
        plt.close(figp)
        st.caption("Net-GEX field: green = +ve (dampening), red = −ve (amplifying), "
                   "black seam = gamma flip / vol trigger. Cone mode = each price "
                   "reaches horizontally in proportion to |net GEX| (tails from the "
                   "right). Walls/pin preserved; candles overlaid. Calls-plus / "
                   "puts-minus convention, not participant-signed.")
    except Exception as e:
        st.error(f"Could not build gradient view: {e}")

# ── Details / table ──────────────────────────────────────────────────────────
with st.expander("Gravity centres · Pin · Floor/Ceiling · Hedge walls (detail)"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Gravity centres**")
        st.table(pd.DataFrame({
            "Level": ["Wall", "Fixed (30/35%)", "Centroid", "Cum.Median (50%)"],
            "Call": [gravity["call_wall"], gravity["call_gravity_fixed"],
                     gravity["call_centroid"], gravity["call_median"]],
            "Put": [gravity["put_wall"], gravity["put_gravity_fixed"],
                    gravity["put_centroid"], gravity["put_median"]],
        }).round(0).astype({"Call": int, "Put": int}))
        st.markdown("**Pin**")
        st.write(f"Pin strike **{pin['pin_strike']:.0f}** · score **{pin['strength_score']}/100** "
                 f"· {pin['strength_label']}")
        st.write(f"Max GEX strike {pin['max_gex_strike']:.0f} · Max OI strike {pin['max_oi_strike']:.0f} "
                 f"· gap {pin['gex_oi_gap']:.0f}")
    with c2:
        st.markdown("**Floor & Ceiling**")
        st.write(f"Ceiling **{fc['ceiling']:.0f}** ({fc['ceiling_distance_pct']:+.2f}%) · {fc['ceiling_strength']}")
        st.write(f"Floor **{fc['floor']:.0f}** ({fc['floor_distance_pct']:+.2f}%) · {fc['floor_strength']}")
        st.write(f"Trading range **{fc['trading_range']:.0f}** ({fc['range_pct']:.2f}%)")
        st.markdown("**Hedge walls**")
        if hw:
            st.write(f"Upside HW **{hw['hedge_wall']:.0f}** · gap from Call Wall "
                     f"{hw['gap_from_call_wall']:+.0f} [{hw['gap_label']}] · conc {hw['pressure_concentration']:.1f}%")
        if dhw:
            st.write(f"Downside HW **{dhw['downside_hedge_wall']:.0f}** · gap from Put Wall "
                     f"{dhw['gap_from_put_wall']:+.0f} [{dhw['gap_label']}] · conc {dhw['pressure_concentration']:.1f}%")
        st.markdown("**Volatility**")
        st.write(f"Fallback IV (60d) **{fallback_iv:.2%}** · Realized vol (30d) **{rv:.2%}**")

with st.expander("Per-strike GEX table"):
    show = gex_df[["strike", "call_gex", "put_gex", "net_gex", "total_gamma",
                   "call_oi", "put_oi", "total_oi", "total_color",
                   "gamma_detonation", "ce_price", "pe_price"]].copy()
    st.dataframe(show, use_container_width=True, height=380)
    st.download_button("Download CSV", show.to_csv(index=False).encode(),
                       file_name=f"nifty_gex_{expiry}.csv", mime="text/csv")

st.caption("Educational tool. Not investment advice. Dealer-positioning is a "
           "modeling assumption, not observed positioning.")
