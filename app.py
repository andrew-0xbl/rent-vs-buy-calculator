from dataclasses import dataclass
from typing import Optional
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from copy import deepcopy

st.set_page_config(page_title="Rent vs Buy (UK & Hong Kong)", page_icon="🏠", layout="wide")

# ------------------------- Default Values Configuration -------------------------
DEFAULT_VALUES = {
    "UK": {
        "price": 425000.0,
        "rent": 1800.0,
        "ltv": 0.75,
        "rate": 4.25,  # percentage
        "term_years": 30,
        "hold_years": 5,
        "rent_growth": 3.0,  # percentage
        "price_growth": 1.0,  # percentage
        "opportunity_rate": 3.0,  # percentage
        "maintenance_rate": 1.0,  # percentage
        "service_charge": 150.0,
        "sdlt_surcharge": 0.0,  # percentage
        "buy_legal": 2500.0,
        "sell_fee_rate": 1.5,  # percentage
    },
    "HK": {
        "price": 9000000.0,
        "rent": 18000.0,
        "ltv": 0.70,
        "rate": 2.3,  # percentage
        "term_years": 30,
        "hold_years": 5,
        "rent_growth": 1.0,  # percentage
        "price_growth": 1.0,  # percentage
        "opportunity_rate": 2.5,  # percentage
        "maintenance_rate": 0.50,  # percentage
        "mgmt_fee_psf": 5.5,
        "net_area_sqft": 350.0,
        "buy_legal": 15000.0,
        "agent_buy_rate": 1.0,  # percentage
        "sell_fee_rate": 1.0,  # percentage
    }
}

# ------------------------- Mortgage math -------------------------

def monthly_payment(principal: float, annual_rate: float, years: int) -> float:
    r = annual_rate / 12.0
    n = years * 12
    if r == 0:
        return principal / n
    return principal * (r * (1 + r)**n) / ((1 + r)**n - 1)

def remaining_balance(principal: float, annual_rate: float, years: int, months_paid: int) -> float:
    r = annual_rate / 12.0
    if r == 0:
        n = years * 12
        return principal * (1 - months_paid / n)
    A = monthly_payment(principal, annual_rate, years)
    return principal * (1 + r)**months_paid - A * (((1 + r)**months_paid - 1) / r)

# ------------------------- Taxes & fees -------------------------

def sdlt_england_main_residence(price_gbp: float, surcharge: float = 0.0) -> float:
    bands = [
        (125000, 0.00),
        (250000, 0.02),
        (925000, 0.05),
        (1500000, 0.10),
        (float("inf"), 0.12),
    ]
    tax = 0.0
    prev = 0.0
    for cap, rate in bands:
        portion = max(0.0, min(price_gbp, cap) - prev)
        if portion > 0:
            tax += portion * rate
        prev = cap
        if price_gbp <= cap:
            break
    if surcharge > 0:
        tax += price_gbp * surcharge
    return round(tax)

def hk_avd_scale2(x: float) -> float:
    # 2025 Scale 2 (simplified piecewise for decisioning)
    if x <= 4000000:
        return 100.0
    if x <= 4323780:
        return 100.0 + 0.20 * (x - 4000000)
    if x <= 4500000:
        return 0.015 * x
    if x <= 4935480:
        return 67500.0 + 0.10 * (x - 4500000)
    if x <= 6000000:
        return 0.0225 * x
    if x <= 6642860:
        return 135000.0 + 0.10 * (x - 6000000)
    if x <= 6720000:
        return 0.03 * x
    if x <= 20000000:
        return 0.0375 * x
    return 0.0425 * x

def hk_rates_annual_from_rent(annual_rent_hkd: float) -> float:
    rv = annual_rent_hkd
    tax = 0.0
    tier1 = min(rv, 550000)
    tax += 0.05 * tier1
    if rv > 550000:
        tier2 = min(rv - 550000, 250000)
        tax += 0.08 * tier2
    if rv > 800000:
        tax += 0.12 * (rv - 800000)
    return tax

def hk_government_rent_annual(annual_rent_hkd: float) -> float:
    return 0.03 * annual_rent_hkd

# ------------------------- Core simulation -------------------------

@dataclass
class Inputs:
    jurisdiction: str  # 'UK' or 'HK'
    price: float
    rent: float
    ltv: float
    rate: float
    term_years: int
    hold_years: int
    rent_growth: float
    price_growth: float
    maintenance_rate: float
    service_charge: float = 0.0
    mgmt_fee_psf: float = 0.0
    net_area_sqft: float = 0.0
    buy_legal: float = 0.0
    agent_buy_rate: float = 0.0
    sell_fee_rate: float = 0.015
    sdlt_surcharge: float = 0.0
    sdlt_override: Optional[float] = None
    hk_rates_override_annual: Optional[float] = None
    hk_govrent_override_annual: Optional[float] = None
    opportunity_rate: float = 0.0

@dataclass
class Results:
    total_rent_paid: float
    total_owner_cash_outflow: float
    final_sale_price: float
    outstanding_balance: float
    sale_costs: float
    final_equity_returned: float
    buy_one_offs: float
    owner_running: float
    interest_paid: float
    principal_paid: float
    net_cost_owning_simple: float
    net_cost_renting_simple: float
    net_cost_owning_opp: Optional[float] = None
    net_cost_renting_opp: Optional[float] = None
    breakeven_growth_simple: Optional[float] = None
    
    @property
    def net_sale_price(self) -> float:
        """Sale price minus sale costs"""
        return self.final_sale_price - self.sale_costs
    
    def net_gain_after_sale(self, initial_price: float, ltv: float) -> float:
        """Net gain = net sale price - outstanding mortgage - total cash invested"""
        deposit = initial_price * (1 - ltv)
        total_cash_invested = deposit + self.buy_one_offs + self.owner_running + self.interest_paid
        return self.net_sale_price - self.outstanding_balance - total_cash_invested


def _compute_core(inputs: Inputs) -> Results:
    """Pure computation for a single scenario WITHOUT breakeven search (no recursion)."""
    months = inputs.hold_years * 12
    price_T = inputs.price * ((1 + inputs.price_growth) ** inputs.hold_years)
    deposit = inputs.price * (1 - inputs.ltv)
    loan = inputs.price - deposit

    if inputs.jurisdiction.upper() == "UK":
        tax_buy = sdlt_england_main_residence(inputs.price, surcharge=inputs.sdlt_surcharge) \
                  if inputs.sdlt_override is None else inputs.sdlt_override
        owner_extra_monthly = inputs.service_charge
        rates_like_annual = 0.0
        govrent_annual = 0.0
    else:
        tax_buy = hk_avd_scale2(inputs.price) if inputs.sdlt_override is None else inputs.sdlt_override
        owner_extra_monthly = inputs.mgmt_fee_psf * inputs.net_area_sqft + inputs.service_charge
        rv_annual = inputs.rent * 12
        rates_like_annual = inputs.hk_rates_override_annual if inputs.hk_rates_override_annual is not None else hk_rates_annual_from_rent(rv_annual)
        govrent_annual = inputs.hk_govrent_override_annual if inputs.hk_govrent_override_annual is not None else hk_government_rent_annual(rv_annual)

    agent_buy = inputs.agent_buy_rate * inputs.price
    buy_one_offs = tax_buy + inputs.buy_legal + agent_buy

    A = monthly_payment(loan, inputs.rate, inputs.term_years)
    bal_T = remaining_balance(loan, inputs.rate, inputs.term_years, months)
    principal_paid = loan - bal_T
    total_interest = A * months - principal_paid

    maintenance_monthly = (inputs.maintenance_rate * inputs.price) / 12.0
    hk_rates_monthly = rates_like_annual / 12.0
    hk_govrent_monthly = govrent_annual / 12.0
    owner_running_monthly = maintenance_monthly + owner_extra_monthly + hk_rates_monthly + hk_govrent_monthly
    owner_running_total = owner_running_monthly * months

    # Renting payments (with growth)
    rent_total = 0.0
    current_rent = inputs.rent
    for _ in range(inputs.hold_years):
        rent_total += current_rent * 12
        current_rent *= (1 + inputs.rent_growth)

    # Selling costs and final equity
    sale_costs = inputs.sell_fee_rate * price_T
    equity_returned = max(0.0, price_T - sale_costs - bal_T)

    # SIMPLE mode: net out-of-pocket
    owner_outflow = deposit + buy_one_offs + (A * months) + owner_running_total - equity_returned
    renter_outflow = rent_total

    # OPPORTUNITY COST CALCULATION - PROPER IMPLEMENTATION
    def future_value_stream(monthly_amounts, r_annual):
        """Calculate future value of a stream of monthly payments at the end of the period"""
        r = r_annual / 12.0
        FV = 0.0
        for i, amt in enumerate(monthly_amounts, start=1):
            months_remaining = len(monthly_amounts) - i
            FV += amt * ((1 + r) ** months_remaining)
        return FV

    # BUYING SCENARIO OPPORTUNITY COSTS:
    # 1. Down payment could have been invested at opportunity rate
    down_payment_opportunity_cost = deposit * ((1 + inputs.opportunity_rate) ** inputs.hold_years) - deposit
    
    # 2. Buy-related one-off costs could have been invested
    buy_costs_opportunity_cost = buy_one_offs * ((1 + inputs.opportunity_rate) ** inputs.hold_years) - buy_one_offs
    
    # 3. Monthly mortgage + running costs opportunity cost
    owner_monthly_cost = A + owner_running_monthly
    owner_monthly_payments = [owner_monthly_cost] * months
    owner_monthly_opportunity_cost = future_value_stream(owner_monthly_payments, inputs.opportunity_rate) - (owner_monthly_cost * months)
    
    # RENTING SCENARIO OPPORTUNITY COSTS:
    # 1. Monthly rent payments opportunity cost
    rent_payments = []
    current_rent = inputs.rent
    for month in range(1, months + 1):
        rent_payments.append(current_rent)
        if month % 12 == 0:
            current_rent *= (1 + inputs.rent_growth)
    
    rent_opportunity_cost = future_value_stream(rent_payments, inputs.opportunity_rate) - sum(rent_payments)
    
    # ALTERNATIVE INVESTMENT SCENARIO FOR RENTER:
    # Renter invests down payment + buy costs at opportunity rate
    renter_alternative_investment = (deposit + buy_one_offs) * ((1 + inputs.opportunity_rate) ** inputs.hold_years)
    
    # NET OPPORTUNITY-ADJUSTED COSTS:
    # Buying: base cost + opportunity costs - equity returned
    net_cost_owning_opp = (owner_outflow + 
                          down_payment_opportunity_cost + 
                          buy_costs_opportunity_cost + 
                          owner_monthly_opportunity_cost)
    
    # Renting: base cost + opportunity cost - alternative investment gains
    net_cost_renting_opp = (renter_outflow + 
                           rent_opportunity_cost - 
                           (renter_alternative_investment - deposit - buy_one_offs))

    return Results(
        total_rent_paid=renter_outflow,
        total_owner_cash_outflow=owner_outflow,
        final_sale_price=price_T,
        outstanding_balance=bal_T,
        sale_costs=sale_costs,
        final_equity_returned=equity_returned,
        buy_one_offs=buy_one_offs,
        owner_running=owner_running_total,
        interest_paid=total_interest,
        principal_paid=principal_paid,
        net_cost_owning_simple=owner_outflow,
        net_cost_renting_simple=renter_outflow,
        net_cost_owning_opp=net_cost_owning_opp,
        net_cost_renting_opp=net_cost_renting_opp,
        breakeven_growth_simple=None,
    )

def build_owner_cashflows_for_irr(inputs: Inputs, res: Results):
    """Return monthly equity cash flows for IRR:
    t=0: -(deposit + buy_one_offs)
    months 1..T-1: -(interest + principal + owner_running_monthly)
    month T: same as above but PLUS terminal equity (net sale price - outstanding balance)
    """
    months = inputs.hold_years * 12
    # Recompute pieces exactly as in _compute_core to ensure consistency
    deposit = inputs.price * (1 - inputs.ltv)
    loan = inputs.price - deposit
    A = monthly_payment(loan, inputs.rate, inputs.term_years)

    # Owner running monthly (mirror _compute_core)
    if inputs.jurisdiction.upper() == "UK":
        owner_extra_monthly = inputs.service_charge
        rv_annual = 0.0
        rates_like_annual = 0.0
        govrent_annual = 0.0
    else:
        owner_extra_monthly = inputs.mgmt_fee_psf * inputs.net_area_sqft + inputs.service_charge
        rv_annual = inputs.rent * 12
        rates_like_annual = inputs.hk_rates_override_annual if inputs.hk_rates_override_annual is not None else hk_rates_annual_from_rent(rv_annual)
        govrent_annual = inputs.hk_govrent_override_annual if inputs.hk_govrent_override_annual is not None else hk_government_rent_annual(rv_annual)

    maintenance_monthly = (inputs.maintenance_rate * inputs.price) / 12.0
    owner_running_monthly = maintenance_monthly + owner_extra_monthly + (rates_like_annual / 12.0) + (govrent_annual / 12.0)

    # Buy one-offs
    if inputs.jurisdiction.upper() == "UK":
        buy_one_offs = (sdlt_england_main_residence(inputs.price, surcharge=inputs.sdlt_surcharge)
                        if inputs.sdlt_override is None else inputs.sdlt_override) + inputs.buy_legal
    else:
        buy_one_offs = (hk_avd_scale2(inputs.price)
                        if inputs.sdlt_override is None else inputs.sdlt_override) + inputs.buy_legal + (inputs.agent_buy_rate * inputs.price)

    # Monthly split of payment into interest/principal
    r_m = inputs.rate / 12.0
    bal = loan
    cashflows = []
    # t=0 outflow
    cashflows.append(-(deposit + buy_one_offs))

    for m in range(1, months + 1):
        interest = bal * r_m
        principal = A - interest
        bal = max(0.0, bal - principal)
        outflow = -(interest + principal + owner_running_monthly)
        cashflows.append(outflow)

    # Add terminal equity in final month
    ending_equity = res.net_sale_price - res.outstanding_balance
    cashflows[-1] += ending_equity  # add terminal inflow to last month
    return cashflows

def irr_annual_from_monthly_cfs(cashflows, lower=-0.99, upper=10.0, iters=80):
    """Bisection IRR on monthly cashflows; return ANNUALISED IRR. Robust for typical cases."""
    def npv(rate):
        return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows))
    lo, hi = lower, upper
    for _ in range(iters):
        mid = (lo + hi) / 2
        v = npv(mid)
        if v > 0:
            lo = mid
        else:
            hi = mid
    monthly = (lo + hi) / 2
    return (1 + monthly) ** 12 - 1

def simulate(inputs: Inputs) -> Results:
    # Compute once
    base = _compute_core(inputs)

    # Breakeven search using the core (no recursion)
    low, high = -0.20, 0.20
    for _ in range(40):
        mid = (low + high) / 2
        tmp_inputs = Inputs(**{**inputs.__dict__, "price_growth": mid})
        tmp = _compute_core(tmp_inputs)
        diff = tmp.net_cost_owning_simple - tmp.net_cost_renting_simple
        if diff > 0:
            low = mid  # owning more expensive; need higher growth
        else:
            high = mid
    breakeven = (low + high) / 2

    return Results(
        **{**base.__dict__, "breakeven_growth_simple": breakeven}
    )

def horizon_profile_dataframe(inputs: Inputs, max_years: int = 30) -> pd.DataFrame:
    """
    Build a DataFrame showing how Equity IRR (annualised) and Net Gain After Sale (% of price)
    evolve as the holding horizon varies from 1..max_years, using the current inputs for everything else.
    """
    horizons = []
    irr_vals = []
    gain_vals_pct = []

    for h in range(1, max_years + 1):
        # Create a shallow copy of inputs with modified hold_years
        tmp_inputs = Inputs(**{**inputs.__dict__, "hold_years": h})

        # Run the simulation for this horizon
        res_h = simulate(tmp_inputs)

        # Equity IRR (annualised) for this horizon
        owner_cfs = build_owner_cashflows_for_irr(tmp_inputs, res_h)
        irr_annual = irr_annual_from_monthly_cfs(owner_cfs)

        # Economic profit (simple) as % of purchase price
        gain_simple = res_h.net_gain_after_sale(tmp_inputs.price, tmp_inputs.ltv)
        gain_pct = (gain_simple / tmp_inputs.price) * 100.0

        horizons.append(h)
        irr_vals.append(irr_annual * 100.0)  # convert to %
        gain_vals_pct.append(gain_pct)

    return pd.DataFrame({
        "Horizon (yrs)": horizons,
        "Equity IRR (annual %)": irr_vals,
        "Net Gain After Sale (%)": gain_vals_pct,
    })

# --- A1. Breakeven appreciation (two flavors) ---

def breakeven_price_growth_simple(inputs: Inputs, tol=1e-6, lo=-0.20, hi=0.20, iters=60):
    """Solve for annual price growth where SIMPLE net costs are equal: own == rent."""
    def diff_at(g):
        tmp = Inputs(**{**inputs.__dict__, "price_growth": g})
        r = _compute_core(tmp)
        return r.net_cost_owning_simple - r.net_cost_renting_simple
    a, b = lo, hi
    fa, fb = diff_at(a), diff_at(b)
    # Expand if needed
    k = 0
    while fa*fb > 0 and k < 8:
        a -= 0.05; b += 0.05
        fa, fb = diff_at(a), diff_at(b); k += 1
    for _ in range(iters):
        m = 0.5*(a+b)
        fm = diff_at(m)
        if abs(fm) < tol: return m
        if fa*fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5*(a+b)

def breakeven_price_growth_irr_nonnegative(inputs: Inputs, tol=1e-6, lo=-0.20, hi=0.20, iters=60):
    """Solve for annual price growth where Equity IRR crosses 0%."""
    def irr_at(g):
        tmp = Inputs(**{**inputs.__dict__, "price_growth": g})
        r = simulate(tmp)
        cfs = build_owner_cashflows_for_irr(tmp, r)
        return irr_annual_from_monthly_cfs(cfs)
    a, b = lo, hi
    fa, fb = irr_at(a), irr_at(b)
    k = 0
    while (fa < 0 and fb < 0) or (fa > 0 and fb > 0):
        a -= 0.05; b += 0.05
        fa, fb = irr_at(a), irr_at(b); k += 1
        if k > 8: break
    for _ in range(iters):
        m = 0.5*(a+b)
        fm = irr_at(m)
        if abs(fm) < 1e-5: return m
        if (fa <= 0 and fm >= 0) or (fa >= 0 and fm <= 0):
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5*(a+b)

# --- A2. HK Rates / Gov Rent toggle calculators (used if auto) ---

def hk_rates_and_govrent_from_rent_monthly(rent_monthly: float):
    rv_annual = rent_monthly * 12.0
    rates_annual = hk_rates_annual_from_rent(rv_annual)   # your existing stepped function
    govrent_annual = hk_government_rent_annual(rv_annual) # 3% of RV
    return rates_annual/12.0, govrent_annual/12.0

# --- A3. Wealth trajectories (month-by-month) ---

def wealth_trajectories(inputs: Inputs):
    """Return DataFrame with years, Buy_NetWorth, Rent_Investor_Wealth (FV basis and level cash).
       Buy_NetWorth(t): property_value(t) - mortgage_balance(t) - (optional sale cost fraction at t -> 0 mid-hold).
       Rent_Investor_Wealth(t): FV of initial (deposit+fees) plus FV of monthly deltas (rent cheaper than own).
       We keep sale costs out during hold (only at terminal) to avoid artificial dips mid-hold.
    """
    months = inputs.hold_years * 12
    dep = inputs.price * (1 - inputs.ltv)
    loan = inputs.price - dep
    r_m = inputs.rate/12.0
    opp_m = inputs.opportunity_rate/12.0

    # Property path
    price_path = [inputs.price * ((1 + inputs.price_growth) ** (t/12.0)) for t in range(months+1)]

    # Mortgage balance path
    A = monthly_payment(loan, inputs.rate, inputs.term_years)
    bal = loan
    balances = [bal]
    for _ in range(months):
        interest = bal * r_m
        principal = A - interest
        bal = max(0.0, bal - principal)
        balances.append(bal)

    # Owner running monthly (mirror _compute_core)
    if inputs.jurisdiction.upper() == "UK":
        owner_extra_monthly = inputs.service_charge
        rates_m = 0.0; govrent_m = 0.0
    else:
        rates_m, govrent_m = hk_rates_and_govrent_from_rent_monthly(inputs.rent) \
            if (inputs.hk_rates_override_annual is None and inputs.hk_govrent_override_annual is None) \
            else ((inputs.hk_rates_override_annual or 0.0)/12.0, (inputs.hk_govrent_override_annual or 0.0)/12.0)
        owner_extra_monthly = inputs.mgmt_fee_psf * inputs.net_area_sqft + inputs.service_charge

    maint_m = (inputs.maintenance_rate * inputs.price) / 12.0
    owner_running_m = maint_m + owner_extra_monthly + rates_m + govrent_m

    # Buy net worth path (in level dollars, not FV) during hold:
    buy_networth = []
    for t in range(months+1):
        eq = price_path[t] - balances[t]
        buy_networth.append(eq)

    # Rent investor wealth path (FV basis):
    # FV of initial (dep + buy_one_offs)
    if inputs.jurisdiction.upper() == "UK":
        tax_buy = sdlt_england_main_residence(inputs.price, surcharge=inputs.sdlt_surcharge) \
                  if inputs.sdlt_override is None else inputs.sdlt_override
        buy_one_offs = tax_buy + inputs.buy_legal
    else:
        tax_buy = hk_avd_scale2(inputs.price) if inputs.sdlt_override is None else inputs.sdlt_override
        buy_one_offs = tax_buy + inputs.buy_legal + inputs.agent_buy_rate*inputs.price

    lump = dep + buy_one_offs
    wealth_rent = []
    FV_lump = lump  # will compound forward
    rent_level = inputs.rent
    for t in range(months+1):
        # compound lump
        if t > 0:
            FV_lump *= (1 + opp_m)
        # monthly delta (if renting cheaper than owning)
        owner_monthly = A + owner_running_m
        monthly_saving = max(0.0, owner_monthly - rent_level)  # money renter can invest
        # invest saving at end of each month
        if t > 0:
            FV_lump += monthly_saving
        wealth_rent.append(FV_lump)
        # step rent annually
        if t > 0 and t % 12 == 0:
            rent_level *= (1 + inputs.rent_growth)

    # Sample data points at year boundaries only (whole numbers)
    year_indices = [t for t in range(months+1) if t % 12 == 0]  # 0, 12, 24, 36, etc.
    year_numbers = [t // 12 for t in year_indices]  # 0, 1, 2, 3, etc.
    year_buy_networth = [buy_networth[i] for i in year_indices]
    year_wealth_rent = [wealth_rent[i] for i in year_indices]
    
    df = pd.DataFrame({
        "Years": year_numbers,  # Whole number years
        "Buy_NetWorth": year_buy_networth,
        "Rent_Investor_Wealth_FV": year_wealth_rent
    })
    return df

# --- A4. Equivalent Monthly Cost (EAC) on FV basis ---

def equivalent_monthly_cost_from_fv(total_fv_cost: float, months: int, opp_annual: float):
    """Convert end-of-horizon FV cost into a level monthly 'equivalent' using the FV annuity factor."""
    r = opp_annual / 12.0
    if months == 0:
        return 0.0
    if r == 0:
        return total_fv_cost / months
    fv_annuity_factor = ((1 + r)**months - 1) / r
    return total_fv_cost / fv_annuity_factor

# --- A5. Ownership cost breakdown for stacked bar (small chart) ---

def ownership_cost_breakdown(inputs: Inputs, res: Results):
    """Return DataFrame with categories and amounts for pie chart."""
    # Create data with categories and amounts
    data = {
        "Category": ["Deposit", "Buy taxes & legal", "Interest", "Maintenance", "Sale costs"],
        "Amount": [
            inputs.price*(1-inputs.ltv),
            res.buy_one_offs,
            res.interest_paid,
            res.owner_running,
            res.sale_costs
        ]
    }
    return pd.DataFrame(data)

# ------------------------- UI LAYOUT -------------------------

st.title("🏠 Rent vs Buy — UK & Hong Kong")

left, right = st.columns([1, 3], gap="large")

with left:
    st.markdown("### Inputs")
    jurisdiction = st.radio("Jurisdiction", ["UK", "HK"], horizontal=True)
    
    # Get defaults for selected jurisdiction
    defaults = DEFAULT_VALUES[jurisdiction]

    price = st.number_input("Purchase price", min_value=0.0, value=defaults["price"], step=1000.0, format="%.0f", help="Total price of the property you're considering buying")
    rent = st.number_input("Current monthly rent", min_value=1.0, value=defaults["rent"], step=50.0, format="%.0f", help="Monthly rent for a comparable property to the one you're buying")
    ltv = st.slider("Loan-to-value (LTV)", 0.0, 0.95, defaults["ltv"], 0.01, help="Percentage of purchase price financed by mortgage (e.g., 0.80 = 80% mortgage, 20% down payment)")
    rate = st.slider("Mortgage rate (annual %)", 0.0, 10.0, defaults["rate"], 0.05, help="Annual interest rate for your mortgage") / 100.0
    term_years = st.slider("Mortgage term (years)", 5, 40, defaults["term_years"], 1, help="Length of mortgage repayment period")
    hold_years = st.slider("Holding horizon (years)", 1, 30, defaults["hold_years"], 1, help="How long you plan to own the property before selling")

    rent_growth = st.slider("Annual rent growth (%)", 0.0, 10.0, defaults["rent_growth"], 0.25, help="Expected annual increase in rental prices") / 100.0
    price_growth = st.slider("Annual price growth (%)", -10.0, 10.0, defaults["price_growth"], 0.25, help="Expected annual increase in property values") / 100.0
    opportunity_rate = st.slider("Opportunity rate (annual %, investable alt.)", 0.0, 10.0, defaults["opportunity_rate"], 0.25, help="Rate of return you could earn by investing money elsewhere (e.g., stocks, bonds)") / 100.0

    st.markdown("#### Owner running & one-offs")
    if jurisdiction == "UK":
        maintenance_rate = st.slider("Maintenance (annual % of price)", 0.0, 3.0, defaults["maintenance_rate"], 0.1, help="Annual maintenance costs as percentage of property value") / 100.0
        service_charge = st.number_input("Service/ground/estate charges (monthly)", min_value=0.0, value=defaults["service_charge"], step=10.0, help="Monthly service charges, ground rent, or estate management fees")
        sdlt_surcharge = st.slider("SDLT surcharge (additional property)", 0.0, 5.0, defaults["sdlt_surcharge"], 0.5, help="Additional SDLT rate if this is not your main residence (usually 3%)") / 100.0
        buy_legal = st.number_input("Buyer legal/surveys", min_value=0.0, value=defaults["buy_legal"], step=100.0, help="Legal fees, surveys, and other costs when buying")
        sell_fee_rate = st.slider("Selling costs (% of sale price)", 0.0, 3.0, defaults["sell_fee_rate"], 0.1, help="Estate agent fees and legal costs when selling (typically 1-2%)") / 100.0
        sdlt_override = st.number_input("Override SDLT (0 = auto)", min_value=0.0, value=0.0, step=100.0, help="Manual SDLT amount (leave 0 for automatic calculation based on price)")
        sdlt_override_val = None if sdlt_override == 0 else sdlt_override
        agent_buy_rate = 0.0
        mgmt_fee_psf = 0.0
        net_area_sqft = 0.0
        hk_rates_override_val = None
        hk_govrent_override_val = None
    else:
        maintenance_rate = st.slider("Maintenance (annual % of price)", 0.0, 3.0, defaults["maintenance_rate"], 0.1, help="Annual maintenance costs as percentage of property value") / 100.0
        mgmt_fee_psf = st.number_input("Management fee (HKD/ft²/mo)", min_value=0.0, value=defaults["mgmt_fee_psf"], step=0.1, help="Monthly management fees charged per square foot")
        net_area_sqft = st.number_input("Net area (ft²)", min_value=0.0, value=defaults["net_area_sqft"], step=10.0, help="Net floor area of the property in square feet")
        buy_legal = st.number_input("Buyer legal/misc (HKD)", min_value=0.0, value=defaults["buy_legal"], step=500.0, help="Legal fees and miscellaneous costs when buying")
        agent_buy_rate = st.slider("Buyer agent fee (% of price)", 0.0, 2.0, defaults["agent_buy_rate"], 0.1, help="Real estate agent commission when buying (typically 1%)") / 100.0
        sell_fee_rate = st.slider("Selling costs (% of sale price)", 0.0, 3.0, defaults["sell_fee_rate"], 0.1, help="Agent fees and legal costs when selling (typically 2%)") / 100.0
        sdlt_override = st.number_input("Override AVD (Simplified) (0 = auto)", min_value=0.0, value=0.0, step=1000.0, help="Manual Ad Valorem Duty amount (leave 0 for automatic calculation)")
        
        # HK Rates & Government Rent toggle
        auto_rates = st.checkbox("Auto-calc Rates & Gov't Rent from rent", value=True,
                                help="If checked, uses current rent to estimate rateable value. Otherwise, enter annual amounts manually.")
        
        if auto_rates:
            hk_rates_override_val = None
            hk_govrent_override_val = None
            st.caption("Using auto: Rates ≈ stepped % of RV; Gov't Rent = 3% of RV. RV derived from current rent.")
        else:
            hk_rates_override = st.number_input("HK Rates (annual)", min_value=0.0, value=0.0, step=100.0)
            hk_govrent_override = st.number_input("Gov't Rent (annual)", min_value=0.0, value=0.0, step=100.0)
            hk_rates_override_val = hk_rates_override
            hk_govrent_override_val = hk_govrent_override
        
        sdlt_override_val = None if sdlt_override == 0 else sdlt_override
        service_charge = 0.0
        sdlt_surcharge = 0.0

    inputs = Inputs(
        jurisdiction=jurisdiction, price=price, rent=rent, ltv=ltv, rate=rate,
        term_years=term_years, hold_years=hold_years, rent_growth=rent_growth,
        price_growth=price_growth, maintenance_rate=maintenance_rate, service_charge=service_charge,
        mgmt_fee_psf=mgmt_fee_psf, net_area_sqft=net_area_sqft, buy_legal=buy_legal, agent_buy_rate=agent_buy_rate,
        sell_fee_rate=sell_fee_rate, sdlt_surcharge=sdlt_surcharge, sdlt_override=sdlt_override_val,
        hk_rates_override_annual=hk_rates_override_val, hk_govrent_override_annual=hk_govrent_override_val,
        opportunity_rate=opportunity_rate
    )

with right:
    res = simulate(inputs)
    cur = "£" if jurisdiction == "UK" else "$"
    # ---- Summary verdict box ----
    st.markdown("### Summary verdict")
    basis = st.radio("Assess using", ["Simple (cash only)", "Opportunity-adjusted"], horizontal=True, key="basis_radio")
    
    if basis.startswith("Simple"):
        st.caption("Simple Mode: Economic profit (excludes principal as expense; includes interest, running, buy/sell costs; no discounting; pre-tax).")
    else:
        st.caption("Opportunity-Adjusted Mode: Economic profit with opportunity costs on tied-up capital and rent payments. All opportunity adjustments are expressed as end-of-horizon future values (no discounting).")
    
    if basis.startswith("Simple"):
        diff = res.net_cost_owning_simple - res.net_cost_renting_simple
        owning = res.net_cost_owning_simple
        renting = res.net_cost_renting_simple
    else:
        if res.net_cost_owning_opp is None or res.net_cost_renting_opp is None:
            raise ValueError("Opportunity costs not calculated")
        diff = res.net_cost_owning_opp - res.net_cost_renting_opp
        owning = res.net_cost_owning_opp
        renting = res.net_cost_renting_opp

    eps = 1.0  # treat within ±£/HK$1 as tie

    # Simplified summary text showing only which is cheaper
    if abs(diff) <= eps:
        st.warning("**Tie**")
    elif diff > 0:
        st.success("**Rent is cheaper**")
    else:
        st.error("**Buy is cheaper**")
    
    # Three metric boxes in a single row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Owning", f"{cur}{owning:,.0f}", help=f"Total net cost of owning over {inputs.hold_years} years")
    
    with col2:
        st.metric("Renting", f"{cur}{renting:,.0f}", help=f"Total cost of renting over {inputs.hold_years} years")
    
    with col3:
        if abs(diff) <= eps:
            color = "#262730"
        elif diff > 0:
            color = "#00C851"
        else:
            color = "#FF4444"
        
        st.markdown(f'<p style="font-size:14px; color:#8892b0; margin:0;">Difference</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:36px; font-weight:600; color:{color}; margin:0; line-height:1;">{cur}{diff:,.0f}</p>', unsafe_allow_html=True)
    
    # ---- Opportunity Cost Breakdown (when in opportunity mode) ----
    if not basis.startswith("Simple"):
        st.markdown("### 🔍 Opportunity Cost Analysis")
        st.markdown("**Understanding the Opportunity-Adjusted Calculation:**")
        
        # Calculate the individual components for display
        deposit = inputs.price * (1 - inputs.ltv)
        
        # Recalculate components for display (matching the calculation logic)
        if inputs.jurisdiction.upper() == "UK":
            buy_one_offs = (sdlt_england_main_residence(inputs.price, surcharge=inputs.sdlt_surcharge) 
                           if inputs.sdlt_override is None else inputs.sdlt_override) + inputs.buy_legal
        else:
            buy_one_offs = (hk_avd_scale2(inputs.price) 
                           if inputs.sdlt_override is None else inputs.sdlt_override) + inputs.buy_legal + (inputs.agent_buy_rate * inputs.price)
        
        down_payment_opp_cost = deposit * ((1 + inputs.opportunity_rate) ** inputs.hold_years) - deposit
        buy_costs_opp_cost = buy_one_offs * ((1 + inputs.opportunity_rate) ** inputs.hold_years) - buy_one_offs
        
        renter_alt_investment_total = (deposit + buy_one_offs) * ((1 + inputs.opportunity_rate) ** inputs.hold_years)
        renter_alt_investment_gains = renter_alt_investment_total - (deposit + buy_one_offs)
        
        # Recompute rent opportunity cost for display (mirror _compute_core logic)
        months = inputs.hold_years * 12
        r_m = inputs.opportunity_rate / 12.0
        current_rent = inputs.rent
        fv_rent_stream = 0.0
        sum_rent = 0.0
        for m in range(1, months + 1):
            sum_rent += current_rent
            months_remaining = months - m
            fv_rent_stream += current_rent * ((1 + r_m) ** months_remaining)
            if m % 12 == 0:
                current_rent *= (1 + inputs.rent_growth)

        rent_opportunity_cost_display = fv_rent_stream - sum_rent
        net_benefit_to_renter = renter_alt_investment_gains - rent_opportunity_cost_display
        
        st.markdown("**Buying Scenario Adjustments:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Down Payment Opportunity Cost", f"{cur}{down_payment_opp_cost:,.0f}")
            st.caption(f"What {cur}{deposit:,.0f} down payment would earn at {inputs.opportunity_rate:.1%}/year")
        with col2:
            st.metric("Buy Costs Opportunity Cost", f"{cur}{buy_costs_opp_cost:,.0f}")
            st.caption(f"What {cur}{buy_one_offs:,.0f} in buying fees would earn at {inputs.opportunity_rate:.1%}/year")
        
        st.markdown("**Renting Scenario Adjustments:**")
        # Show both components + net
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Investment gains on deposit + fees", f"{cur}{renter_alt_investment_gains:,.0f}")
        with col2:
            st.metric("Opportunity cost on rent stream", f"{cur}{rent_opportunity_cost_display:,.0f}")
        with col3:
            st.metric("Net Benefit to Renter (opportunity)", f"{cur}{net_benefit_to_renter:,.0f}")
        st.caption("Net = gains from investing the initial lump sum minus the opportunity cost applied to the rent stream.")
        
        st.markdown("---")
        st.markdown("**Interpretation:**")
        if inputs.opportunity_rate > 0:
            st.markdown(f"""
            - **Buying** becomes more expensive when you consider that your down payment and fees could earn {inputs.opportunity_rate:.1%}/year elsewhere
            - **Renting** becomes relatively cheaper because you can invest the down payment money instead
            - The opportunity rate of {inputs.opportunity_rate:.1%}/year represents your best alternative investment (e.g., index funds, bonds)
            """)
        else:
            st.markdown("- With 0% opportunity rate, both modes show the same results (no alternative investment consideration)")



    # ---- Breakdown (top area) ----
    st.markdown("### Breakdown")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Deposit", f"{cur}{inputs.price*(1-inputs.ltv):,.0f}", border=True, help="Down payment required upfront")
    m2.metric("Upfront taxes & fees", f"{cur}{res.buy_one_offs:,.0f}", border=True, help="One-time costs when purchasing: stamp duty, legal fees, surveys")
    m3.metric("Interest paid (horizon)", f"{cur}{res.interest_paid:,.0f}", border=True, help=f"Total mortgage interest payments over {inputs.hold_years} years")
    m4.metric("Principal repaid (horizon)", f"{cur}{res.principal_paid:,.0f}", border=True, help=f"Mortgage principal paid down over {inputs.hold_years} years")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Owner running (horizon)", f"{cur}{res.owner_running:,.0f}", border=True, help=f"Ongoing costs over {inputs.hold_years} years: maintenance, service charges, insurance")
    m6.metric("Estimated Sale costs", f"{cur}{res.sale_costs:,.0f}", border=True, help="Estate agent fees and legal costs when selling")
    m7.metric("Mortgage balance at sale", f"{cur}{res.outstanding_balance:,.0f}", border=True, help=f"Remaining mortgage debt after {inputs.hold_years} years")

    # Sale Analysis Section
    st.markdown("### Sale Analysis")
    st.caption("Economic profit shown above treats principal as equity, not expense (no discounting, pre-tax). Equity IRR uses cash flows (including principal timing) and provides a time-weighted return on equity.")
    
    # Calculate percentage changes for deltas
    price_change_pct = ((res.final_sale_price - inputs.price) / inputs.price) * 100
    net_sale_change_pct = ((res.net_sale_price - inputs.price) / inputs.price) * 100
    net_gain = res.net_gain_after_sale(inputs.price, inputs.ltv)
    
    # Equity contributed = deposit + cumulative principal repaid
    deposit = inputs.price * (1 - inputs.ltv)
    equity_contributed = deposit + res.principal_paid
    ending_equity = res.net_sale_price - res.outstanding_balance
    equity_multiple = (ending_equity / equity_contributed) if equity_contributed > 0 else float("nan")

    # IRR on equity cash flows
    owner_cfs = build_owner_cashflows_for_irr(inputs, res)
    equity_irr_annual = irr_annual_from_monthly_cfs(owner_cfs)
    
    s1, s2, s3 = st.columns(3)
    s1.metric("Sale price (horizon)", f"{cur}{res.final_sale_price:,.0f}", 
              delta=f"{price_change_pct:+.1f}%", border=True, 
              help=f"Expected property value after {inputs.hold_years} years of growth")
    s2.metric("Net sale price", f"{cur}{res.net_sale_price:,.0f}", 
              delta=f"{net_sale_change_pct:+.1f}%", border=True,
              help="Sale price after deducting selling costs")
    s3.metric("Net gain after sale", f"{cur}{net_gain:,.0f}", 
              delta=f"{(net_gain/inputs.price)*100:+.1f}%" if net_gain != 0 else "0.0%", border=True,
              help="Economic profit (excludes principal as expense; includes interest, running, buy/sell costs; no discounting; pre-tax)")

    t1, t2 = st.columns(2)

    t1.metric("Equity Multiple", f"{equity_multiple:,.2f}", border=True,
              help="(Ending equity) / (Deposit + Principal repaid)")
    
    # Add a new row for IRR
    t2.metric("Equity IRR (annual)", f"{equity_irr_annual:.2%}",
              help="IRR computed on owner equity cash flows (principal treated as capital, not expense). Equity IRR reflects actual cash flow timing (principal included in cash flows), annualised.", border=True)

    st.markdown(f"**Breakeven annual price growth (simple, over {inputs.hold_years}y):** {res.breakeven_growth_simple:.2%}")

    # ---- Horizon vs Return Chart ----
    st.markdown("### Return Profile vs. Holding Horizon")
    hp_df = horizon_profile_dataframe(inputs, max_years=30)
    st.line_chart(hp_df.set_index("Horizon (yrs)"), use_container_width=True)
    st.caption("Equity IRR is annualised and cashflow-based (includes principal timing). Net Gain After Sale is an economic profit (% of purchase price) that excludes principal as an expense but includes interest, running, and transaction costs.")

    # ---- Breakeven Calculators ----
    st.markdown("### Breakeven Calculators")
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        g_simple = breakeven_price_growth_simple(inputs)
        st.metric("Breakeven price growth (Simple)", f"{g_simple:.2%}",
                  help="Annual property growth where simple net costs are equal (own = rent).", border=True)
    with bcol2:
        g_irr0 = breakeven_price_growth_irr_nonnegative(inputs)
        st.metric("Breakeven price growth (IRR = 0%)", f"{g_irr0:.2%}",
                  help="Annual property growth where equity IRR crosses 0%.", border=True)

    # ---- Consumption Premium & Equivalent Monthly Cost ----
    st.markdown("### Consumption Premium & Equivalent Monthly Cost")
    months = inputs.hold_years * 12
    # Use OPPORTUNITY-ADJUSTED totals for EAC on FV basis (consistent with your app's FV opportunity framing)
    own_fv = res.net_cost_owning_opp if res.net_cost_owning_opp is not None else res.net_cost_owning_simple
    rent_fv = res.net_cost_renting_opp if res.net_cost_renting_opp is not None else res.net_cost_renting_simple
    eac_own = equivalent_monthly_cost_from_fv(own_fv, months, inputs.opportunity_rate)
    eac_rent = equivalent_monthly_cost_from_fv(rent_fv, months, inputs.opportunity_rate)
    consumption_premium = eac_own - eac_rent  # >0 means paying extra per month to own vs rent

    p1, p2, p3, p4 = st.columns(4)
    cur = "£" if inputs.jurisdiction == "UK" else "$"
    with p1: st.metric("Equivalent Monthly Cost — Own", f"{cur}{eac_own:,.0f}", border=True)
    with p2: st.metric("Equivalent Monthly Cost — Rent", f"{cur}{eac_rent:,.0f}", border=True)
    with p3:
        st.metric("Consumption Premium (per month)", f"{cur}{consumption_premium:,.0f}",
                  help="Positive = you pay this extra per month to own (for stability/lifestyle).", border=True)
    with p4:
        # Calculate percentage difference
        if eac_rent != 0:
            premium_pct = (consumption_premium / eac_rent) * 100
        else:
            premium_pct = 0
        
        # Determine color based on premium
        if abs(premium_pct) < 0.1:  # Very close to zero
            color = "#FFA500"  # Orange/Yellow
        elif premium_pct > 0:  # Own more expensive
            color = "#00C851"  # Green
        else:  # Rent more expensive
            color = "#FF4444"  # Red
        
        st.markdown(f'<p style="font-size:14px; color:#8892b0; margin:0;">Own Premium Percentage</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:28px; font-weight:600; color:{color}; margin:0; line-height:1;">{premium_pct:+.1f}%</p>', unsafe_allow_html=True)

    # ---- Charts Row: Wealth Trajectory and Ownership Cost Mix ----
    chart_left, chart_right = st.columns([2, 1], gap="medium")
    
    with chart_left:
        st.markdown("### Wealth Trajectory (Rent & Invest vs Buy)")
        wdf = wealth_trajectories(inputs)
        
        # Prepare data for stacked bar chart
        chart_data = pd.DataFrame({
            'Years': wdf['Years'],
            'Buy Net Worth': wdf['Buy_NetWorth'],
            'Rent Investor Wealth FV': wdf['Rent_Investor_Wealth_FV']
        })
        
        # Melt data for stacked bar chart
        melted_data = pd.melt(chart_data, id_vars=['Years'], 
                            value_vars=['Buy Net Worth', 'Rent Investor Wealth FV'],
                            var_name='Scenario', value_name='Wealth')
        
        # Create grouped bar chart using Altair
        bar_chart = alt.Chart(melted_data).mark_bar().encode(
            x=alt.X('Years:O', title='Years', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('Wealth:Q', title='Wealth', axis=alt.Axis(format=',.0f')),
            color=alt.Color('Scenario:N', 
                          scale=alt.Scale(range=['#FF6B6B', '#4ECDC4']),
                          legend=alt.Legend(orient='left', titleLimit=0)),
            xOffset='Scenario:N',
            tooltip=[alt.Tooltip('Years:O'), 
                    alt.Tooltip('Scenario:N'), 
                    alt.Tooltip('Wealth:Q', format=',.0f')]
        ).properties(
            height=400
        )
        
        st.altair_chart(bar_chart, use_container_width=True)
        st.caption("Buy Net Worth = property value − mortgage balance (no sale costs mid-hold).")
        st.caption("Rent Investor Wealth FV = FV of deposit+fees invested plus invested monthly savings, at the opportunity rate.")
    
    with chart_right:
        st.markdown("### Ownership Cost Mix")
        odf = ownership_cost_breakdown(inputs, res)
        
        # Create pie chart with diverse color palette
        colors = ['#ff6e61', '#708dff', '#70ffb0', '#ffd900', '#ff4d4d']
        
        pie_chart = alt.Chart(odf).mark_arc(innerRadius=50, outerRadius=120).encode(
            theta=alt.Theta('Amount:Q'),
            color=alt.Color('Category:N', 
                          scale=alt.Scale(range=colors),
                          legend=alt.Legend(orient='left', titleLimit=0, labelLimit=0)),
            tooltip=['Category:N', alt.Tooltip('Amount:Q', format=',.0f')]
        ).resolve_scale(
            color='independent'
        )
        
        st.altair_chart(pie_chart, use_container_width=False)

    # ---- Sensitivity Analysis ----
    st.markdown("### Sensitivity — Annual Price Growth")
    
    # Slider
    g_sens = st.slider("Test price growth (%)", -5.0, 8.0, float(inputs.price_growth*100), 0.25) / 100.0
    
    # Two-column layout: Chart on left, Metrics on right
    sens_left, sens_right = st.columns([2, 1], gap="medium")
    
    # Calculate sensitivity metrics
    tmp_inputs = Inputs(**{**inputs.__dict__, "price_growth": g_sens})
    tmp_res = simulate(tmp_inputs)
    tmp_cfs = build_owner_cashflows_for_irr(tmp_inputs, tmp_res)
    tmp_irr = irr_annual_from_monthly_cfs(tmp_cfs)
    
    with sens_left:
        # Chart - Bar chart with whole number years
        wdf_s = wealth_trajectories(tmp_inputs)
        # Create bar chart data
        chart_data = pd.DataFrame({
            "Year": wdf_s["Years"],
            "Buy Net Worth": wdf_s["Buy_NetWorth"]
        })
        
        bar_chart = alt.Chart(chart_data).mark_bar(color='#45B7D1').encode(
            x=alt.X('Year:O', title='Year', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('Buy Net Worth:Q', title='Buy Net Worth', axis=alt.Axis(format=',.0f')),
            tooltip=[alt.Tooltip('Year:O'), alt.Tooltip('Buy Net Worth:Q', format=',.0f')]
        ).properties(
            height=300
        )
        
        st.altair_chart(bar_chart, use_container_width=True)
    
    with sens_right:
        # Two metrics in separate rows
        st.metric("Equity IRR (sens)", f"{tmp_irr:.2%}", border=True)
        st.metric("Net gain after sale (sens)", f"{tmp_res.net_gain_after_sale(tmp_inputs.price, tmp_inputs.ltv)/tmp_inputs.price*100:.2f}%", border=True)

# ---- FAQ Section ----
st.markdown("---")
st.markdown("## ❓ FAQ")

with st.expander("📐 How are the calculations performed?"):
    # Simple mode formulas
    st.markdown("**Simple Mode (Cash-Only Analysis):**")
    st.markdown("This mode only considers actual cash flows without opportunity costs.")
    st.latex(r"""
    \text{Net Cost Owning} = \text{Deposit} + \text{Buy Fees} + \text{Mortgage Payments} + \text{Running Costs} - \text{Equity Returned}
    """)
    st.latex(r"""
    \text{Net Cost Renting} = \sum_{t=1}^{T} \text{Rent}_t \times (1 + g_r)^{t-1}
    """)
    
    st.markdown("---")
    
    # Opportunity cost methodology
    st.markdown("**Opportunity-Adjusted Mode (Investment Alternative Analysis):**")
    st.markdown("""
    This mode considers what you could earn by investing your money elsewhere at the opportunity rate.
    
    **Key Concept:** When you buy a property, you tie up capital that could be invested. When you rent, 
    you can invest the down payment and buying costs in alternative investments.
    """)
    
    st.markdown("**Buying Scenario Opportunity Costs:**")
    st.markdown("1. **Down Payment Opportunity Cost:** What the down payment would earn if invested")
    st.latex(r"""
    \text{Down Payment OC} = \text{Deposit} \times [(1 + r_{opp})^T - 1]
    """)
    
    st.markdown("2. **Upfront Costs Opportunity Cost:** What buying fees would earn if invested")
    st.latex(r"""
    \text{Buy Costs OC} = \text{Buy Fees} \times [(1 + r_{opp})^T - 1]
    """)
    
    st.markdown("3. **Monthly Payments Opportunity Cost:** What monthly mortgage+running costs would earn if invested")
    st.latex(r"""
    \text{Monthly OC} = \text{FV}(\text{Monthly Payments}, r_{opp}) - \sum \text{Monthly Payments}
    """)
    
    st.markdown("**Renting Scenario:**")
    st.markdown("1. **Rent Opportunity Cost:** What rent payments would earn if invested instead")
    st.latex(r"""
    \text{Rent OC} = \text{FV}(\text{Rent Payments}, r_{opp}) - \sum \text{Rent Payments}
    """)
    
    st.markdown("2. **Alternative Investment Benefit:** Renter invests down payment + buy costs")
    st.latex(r"""
    \text{Alt Investment} = (\text{Deposit} + \text{Buy Fees}) \times (1 + r_{opp})^T
    """)
    
    st.markdown("**Final Opportunity-Adjusted Costs:**")
    st.latex(r"""
    \text{Net Cost Owning}_{opp} = \text{Net Cost Owning} + \text{All Buying OCs}
    """)
    st.latex(r"""
    \text{Net Cost Renting}_{opp} = \text{Net Cost Renting} + \text{Rent OC} - \text{Alt Investment Gains}
    """)
    
    st.markdown("**Where:**")
    st.markdown("- $r_{opp}$ = Annual opportunity rate")
    st.markdown("- $T$ = Holding period in years") 
    st.markdown("- $\\text{FV}(\\cdot)$ = Future value of payment stream")
    st.markdown("- OC = Opportunity Cost")

with st.expander("🏠 What's the difference between UK and Hong Kong modes?"):
    st.markdown("**UK Mode includes:**")
    st.markdown("- SDLT (Stamp Duty Land Tax) calculations with main residence rates")
    st.markdown("- Optional SDLT surcharge for additional properties")
    st.markdown("- Service charges, ground rent, and estate management fees")
    st.markdown("- Legal fees and surveys")
    
    st.markdown("**Hong Kong Mode includes:**")
    st.markdown("- Simplified AVD (Ad Valorem Duty) using Scale 2 rates")
    st.markdown("- Management fees calculated per square foot")
    st.markdown("- Government rent and rates calculations")
    st.markdown("- Buyer agent commissions")

with st.expander("💰 What does 'opportunity rate' mean?"):
    st.markdown("""
    The opportunity rate represents the return you could earn by investing your money elsewhere instead of buying property.
    
    **Examples:**
    - 3-4%: Government bonds or high-yield savings
    - 5-7%: Diversified index funds (long-term average)
    - 8-10%: Stock market (higher risk, higher potential return)
    
    **Why it matters:** When you buy property, you tie up capital (down payment + fees) that could be invested elsewhere. 
    The opportunity-adjusted mode accounts for this by comparing what you could earn from alternative investments.
    """)

with st.expander("📊 How should I interpret the results?"):
    st.markdown("""
    **Simple Mode:** Pure cash flow comparison - which option costs you less out of pocket over the holding period.
    
    **Opportunity-Adjusted Mode:** More sophisticated analysis that considers investment alternatives. Use this if you're 
    comfortable with investing and want to account for the opportunity cost of tying up capital in property.
    
    **The verdict:** Shows which option is financially better based on your inputs. Remember this is just one factor 
    in your decision - also consider lifestyle preferences, risk tolerance, and market timing.
    """)

st.caption("This tool is a decision aid, not financial advice. Always verify tax/fee rules for your exact case.")
