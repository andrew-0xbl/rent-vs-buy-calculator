from math import ceil


def sdlt_england_main_residence(price_gbp: float, surcharge: float = 0.0) -> float:
    """
    SDLT for a MAIN residence in England & NI – base residential bands as of Apr 1, 2025.
    Bands (single property):
      - 0%    on up to £125,000
      - 2%    on £125,001–£250,000
      - 5%    on £250,001–£925,000
      - 10%   on £925,001–£1,500,000
      - 12%   above £1,500,000

    'surcharge' is an optional flat uplift to approximate higher-rate regimes:
      - Additional dwellings surcharge = +5% (from 31 Oct 2024)
      - Non-resident surcharge = +2%
      - If both apply: pass 0.07 (i.e., 7%)
    NOTE: HMRC implements higher rates by adding +X percentage points to each band.
    Algebraically, that extra equals X% of the *full* price, so adding price*surcharge
    reproduces the higher-rate uplift exactly.

    Returns: total SDLT rounded to the nearest pound (float for compatibility).
    Caveats:
      - Does NOT model first-time buyer relief or special reliefs.
      - Does NOT handle linked transactions, MDR, corporate/6+ dwellings cases, or lease rent SDLT.
    """
    p = max(0.0, float(price_gbp))
    bands = [
        (125_000, 0.00),
        (250_000, 0.02),
        (925_000, 0.05),
        (1_500_000, 0.10),
        (float("inf"), 0.12),
    ]
    tax = 0.0
    prev = 0.0
    for cap, rate in bands:
        portion = max(0.0, min(p, cap) - prev)
        if portion <= 0:
            break
        tax += portion * rate
        prev = cap

    # Higher-rate approximation (see docstring):
    if surcharge > 0:
        tax += p * surcharge

    return float(round(tax))


def hk_avd_scale2(x: float) -> float:
    """
    Hong Kong Ad Valorem Duty – SCALE 2 (effective 26 Feb 2025).
    Apply stepped schedule on the higher of consideration or value.
    Duty must be **rounded UP** to the nearest HK$1 (ceil).

    Tiers:
      ≤ 4,000,000                         : $100
      4,000,000 – 4,323,780               : $100 + 20% of excess over 4,000,000
      4,323,780 – 4,500,000               : 1.5% of price
      4,500,000 – 4,935,480               : $67,500 + 10% of excess over 4,500,000
      4,935,480 – 6,000,000               : 2.25% of price
      6,000,000 – 6,642,860               : $135,000 + 10% of excess over 6,000,000
      6,642,860 – 9,000,000               : 3.00% of price
      9,000,000 – 10,080,000              : $270,000 + 10% of excess over 9,000,000
      10,080,000 – 20,000,000             : 3.75% of price
      20,000,000 – 21,739,120             : $750,000 + 10% of excess over 20,000,000
      > 21,739,120                        : 4.25% of price

    Notes:
      - BSD/SSD measures no longer apply to *new* purchases after 28 Feb 2024.
      - Agreements are charged the same rates; the conveyance that follows is $100.

    Returns: duty as float; computed with ceil() per IRD rule.
    """
    x = float(x)
    if x <= 4_000_000:
        duty = 100.0
    elif x <= 4_323_780:
        duty = 100.0 + 0.20 * (x - 4_000_000)
    elif x <= 4_500_000:
        duty = 0.015 * x
    elif x <= 4_935_480:
        duty = 67_500.0 + 0.10 * (x - 4_500_000)
    elif x <= 6_000_000:
        duty = 0.0225 * x
    elif x <= 6_642_860:
        duty = 135_000.0 + 0.10 * (x - 6_000_000)
    elif x <= 9_000_000:
        duty = 0.03 * x
    elif x <= 10_080_000:
        duty = 270_000.0 + 0.10 * (x - 9_000_000)
    elif x <= 20_000_000:
        duty = 0.0375 * x
    elif x <= 21_739_120:
        duty = 750_000.0 + 0.10 * (x - 20_000_000)
    else:
        duty = 0.0425 * x

    return float(ceil(duty))  # IRD: round UP to nearest $1


def hk_rates_annual_from_rent(annual_rent_hkd: float) -> float:
    """
    Hong Kong 'Rates' (ANNUAL) using the **progressive domestic** schedule from 1 Jan 2025:
      - 5% on first HK$550,000 of rateable value (RV)
      - 8% on next HK$250,000
      - 12% on the remainder

    IMPORTANT:
      - This function historically took 'annual_rent_hkd'. In practice, Rates are assessed on RV,
        which may differ from actual rent. We treat the input as a proxy for RV for decisioning.
      - Budget waivers (e.g., quarterly concessions) are NOT applied here—use your manual override
        for precise cash planning.

    Returns: annual Rates rounded to nearest HK$.
    """
    rv = max(0.0, float(annual_rent_hkd))
    tier1 = min(rv, 550_000.0)
    tier2 = min(max(rv - 550_000.0, 0.0), 250_000.0)
    tier3 = max(rv - 800_000.0, 0.0)
    tax = 0.05 * tier1 + 0.08 * tier2 + 0.12 * tier3
    return float(round(tax))


def hk_government_rent_annual(annual_rent_hkd: float) -> float:
    """
    Hong Kong Government Rent (ANNUAL) = 3% of RV (proxied here by input).
    Notes:
      - Most leases are liable; a few legacy cases vary. We assume standard terms.
      - No waivers applied here; use overrides to reflect current concessions (if any).

    Returns: annual Gov Rent rounded to nearest HK$.
    """
    return float(round(0.03 * max(0.0, float(annual_rent_hkd))))


def hk_rates_and_govrent_from_rent_monthly(rent_monthly: float):
    """
    Convenience helper to infer annual 'Rates' and Government Rent from a monthly rent proxy.
    Assumes RV ≈ 12 × rent_monthly. Suitable for modelling; not an official assessment.

    Returns: (rates_monthly, govrent_monthly) as floats.
    """
    rv_annual = max(0.0, float(rent_monthly)) * 12.0
    rates_annual = hk_rates_annual_from_rent(rv_annual)
    govrent_annual = hk_government_rent_annual(rv_annual)
    return rates_annual / 12.0, govrent_annual / 12.0
