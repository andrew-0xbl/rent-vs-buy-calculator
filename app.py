import math
from dataclasses import dataclass
from typing import Optional, List
import pandas as pd
import streamlit as st

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
        "maintenance_rate": 1.0,  # percentage
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

def amortization_interest_paid(principal: float, annual_rate: float, years: int, months: int) -> float:
    """Calculate total interest paid over specified period using direct formula for efficiency"""
    r = annual_rate / 12.0
    if r == 0:
        return 0.0
    
    n = years * 12
    A = monthly_payment(principal, annual_rate, years)
    
    # Direct formula for interest paid in first 'months' payments
    # Interest = A * months - principal * (1 - (1+r)^(-months)) / r * r
    # Simplified: Interest = A * months - principal * (1 - (1+r)^(-months))
    return A * months - principal * (1 - (1 + r) ** (-months))

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
    total_interest = amortization_interest_paid(loan, inputs.rate, inputs.term_years, months)
    bal_T = remaining_balance(loan, inputs.rate, inputs.term_years, months)
    principal_paid = loan - bal_T

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

def simulate(inputs: Inputs) -> Results:
    # Compute once
    base = _compute_core(inputs)

    # Breakeven search using the core (no recursion)
    low, high = -0.10, 0.10
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
# Time series for charts
def monthly_series(inputs: Inputs):
    months = inputs.hold_years * 12
    deposit = inputs.price * (1 - inputs.ltv)
    loan = inputs.price - deposit
    A = monthly_payment(loan, inputs.rate, inputs.term_years)

    # Owner running monthly
    if inputs.jurisdiction.upper() == "UK":
        owner_extra_monthly = inputs.service_charge
        rates_like_annual = 0.0
        govrent_annual = 0.0
    else:
        owner_extra_monthly = inputs.mgmt_fee_psf * inputs.net_area_sqft + inputs.service_charge
        rv_annual = inputs.rent * 12
        rates_like_annual = inputs.hk_rates_override_annual if inputs.hk_rates_override_annual is not None else hk_rates_annual_from_rent(rv_annual)
        govrent_annual = inputs.hk_govrent_override_annual if inputs.hk_govrent_override_annual is not None else hk_government_rent_annual(rv_annual)

    maintenance_monthly = (inputs.maintenance_rate * inputs.price) / 12.0
    owner_running_monthly = maintenance_monthly + owner_extra_monthly + (rates_like_annual/12.0) + (govrent_annual/12.0)

    # Rent path (increase annually)
    rent_by_month = []
    current_rent = inputs.rent
    for m in range(1, months+1):
        rent_by_month.append(current_rent)
        if m % 12 == 0:
            current_rent *= (1 + inputs.rent_growth)

    # Mortgage amortization
    balances = [loan]
    interest_by_month = []
    principal_by_month = []
    bal = loan
    r = inputs.rate / 12.0
    for _ in range(1, months+1):
        interest = bal * r
        principal = A - interest
        bal = max(0.0, bal - principal)
        interest_by_month.append(interest)
        principal_by_month.append(principal)
        balances.append(bal)

    # Cumulative rent
    cum_rent = []
    s = 0.0
    for x in rent_by_month:
        s += x
        cum_rent.append(s)

    # One-offs
    if inputs.jurisdiction.upper() == "UK":
        stamp = sdlt_england_main_residence(inputs.price, surcharge=inputs.sdlt_surcharge) if inputs.sdlt_override is None else inputs.sdlt_override
        agent_buy = 0.0
    else:
        stamp = hk_avd_scale2(inputs.price) if inputs.sdlt_override is None else inputs.sdlt_override
        agent_buy = inputs.agent_buy_rate * inputs.price
    buy_one_offs = stamp + inputs.buy_legal + agent_buy

    # Cumulative buy (before equity refund)
    cum_buy = []
    s = deposit + buy_one_offs
    for _ in range(months):
        s += A + owner_running_monthly
        cum_buy.append(s)

    # Equity estimate over time
    g_month = (1 + inputs.price_growth) ** (1/12) - 1 if inputs.price_growth != 0 else 0.0
    price_path = []
    p = inputs.price
    for _ in range(1, months+1):
        p *= (1 + g_month)
        price_path.append(p)

    sale_cost_rate = inputs.sell_fee_rate
    equity_series = []
    for m in range(months):
        val = price_path[m]
        bal = balances[m+1]
        sale_cost_est = val * sale_cost_rate
        equity = max(0.0, val - sale_cost_est - bal)
        equity_series.append(equity)

    monthly_owner_payment = [monthly_payment(loan, inputs.rate, inputs.term_years) + owner_running_monthly] * months

    return pd.DataFrame({
        "Month": list(range(1, months+1)),
        "Monthly Rent": rent_by_month,
        "Monthly Owner Outflow": monthly_owner_payment,
        "Cumulative Rent": cum_rent,
        "Cumulative Buy (pre-refund)": cum_buy,
        "Estimated Equity": equity_series,
    }).set_index("Month")

# ------------------------- UI LAYOUT -------------------------

st.title("🏠 Rent vs Buy — UK & Hong Kong")
st.caption("Left: inputs. Right: breakdown (top) and charts (below).")

left, right = st.columns([1, 3], gap="large")

with left:
    st.markdown("### Inputs")
    jurisdiction = st.radio("Jurisdiction", ["UK", "HK"], horizontal=True)
    
    # Get defaults for selected jurisdiction
    defaults = DEFAULT_VALUES[jurisdiction]

    price = st.number_input("Purchase price", min_value=0.0, value=defaults["price"], step=1000.0, format="%.2f")
    rent = st.number_input("Current monthly rent", min_value=0.0, value=defaults["rent"], step=50.0, format="%.2f")
    ltv = st.slider("Loan-to-value (LTV)", 0.0, 0.95, defaults["ltv"], 0.01)
    rate = st.slider("Mortgage rate (annual %)", 0.0, 10.0, defaults["rate"], 0.05) / 100.0
    term_years = st.slider("Mortgage term (years)", 5, 40, defaults["term_years"], 1)
    hold_years = st.slider("Holding horizon (years)", 1, 15, defaults["hold_years"], 1)

    rent_growth = st.slider("Annual rent growth (%)", 0.0, 10.0, defaults["rent_growth"], 0.25) / 100.0
    price_growth = st.slider("Annual price growth (%)", -10.0, 10.0, defaults["price_growth"], 0.25) / 100.0
    opportunity_rate = st.slider("Opportunity rate (annual %, investable alt.)", 0.0, 10.0, defaults["opportunity_rate"], 0.25) / 100.0

    st.markdown("#### Owner running & one-offs")
    if jurisdiction == "UK":
        maintenance_rate = st.slider("Maintenance (annual % of price)", 0.0, 3.0, defaults["maintenance_rate"], 0.1) / 100.0
        service_charge = st.number_input("Service/ground/estate charges (monthly)", min_value=0.0, value=defaults["service_charge"], step=10.0)
        sdlt_surcharge = st.slider("SDLT surcharge (additional property)", 0.0, 5.0, defaults["sdlt_surcharge"], 0.5) / 100.0
        buy_legal = st.number_input("Buyer legal/surveys", min_value=0.0, value=defaults["buy_legal"], step=100.0)
        sell_fee_rate = st.slider("Selling costs (% of sale price)", 0.0, 3.0, defaults["sell_fee_rate"], 0.1) / 100.0
        sdlt_override = st.number_input("Override SDLT (0 = auto)", min_value=0.0, value=0.0, step=100.0)
        sdlt_override_val = None if sdlt_override == 0 else sdlt_override
        agent_buy_rate = 0.0
        mgmt_fee_psf = 0.0
        net_area_sqft = 0.0
        hk_rates_override_val = None
        hk_govrent_override_val = None
    else:
        maintenance_rate = st.slider("Maintenance (annual % of price)", 0.0, 3.0, defaults["maintenance_rate"], 0.1) / 100.0
        mgmt_fee_psf = st.number_input("Management fee (HKD/ft²/mo)", min_value=0.0, value=defaults["mgmt_fee_psf"], step=0.1)
        net_area_sqft = st.number_input("Net area (ft²)", min_value=0.0, value=defaults["net_area_sqft"], step=10.0)
        buy_legal = st.number_input("Buyer legal/misc (HKD)", min_value=0.0, value=defaults["buy_legal"], step=500.0)
        agent_buy_rate = st.slider("Buyer agent fee (% of price)", 0.0, 2.0, defaults["agent_buy_rate"], 0.1) / 100.0
        sell_fee_rate = st.slider("Selling costs (% of sale price)", 0.0, 3.0, defaults["sell_fee_rate"], 0.1) / 100.0
        sdlt_override = st.number_input("Override AVD (0 = auto)", min_value=0.0, value=0.0, step=1000.0)
        hk_rates_override = st.number_input("Override HK Rates (annual, 0=auto)", min_value=0.0, value=0.0, step=100.0)
        hk_govrent_override = st.number_input("Override Gov't Rent (annual, 0=auto)", min_value=0.0, value=0.0, step=100.0)
        sdlt_override_val = None if sdlt_override == 0 else sdlt_override
        service_charge = 0.0
        sdlt_surcharge = 0.0
        hk_rates_override_val = None if hk_rates_override == 0 else hk_rates_override
        hk_govrent_override_val = None if hk_govrent_override == 0 else hk_govrent_override

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

    # ---- Formulas (dropdown) ----
    with st.expander("📐 View Calculation Formulas & Methodology"):
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

    # ---- Breakdown (top area) ----
    st.markdown("### Breakdown")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Deposit", f"{cur}{inputs.price*(1-inputs.ltv):,.0f}")
    m2.metric("Upfront taxes & fees", f"{cur}{res.buy_one_offs:,.0f}")
    m3.metric("Interest paid (horizon)", f"{cur}{res.interest_paid:,.0f}")
    m4.metric("Principal repaid (horizon)", f"{cur}{res.principal_paid:,.0f}")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Owner running (horizon)", f"{cur}{res.owner_running:,.0f}")
    m6.metric("Sale price (horizon)", f"{cur}{res.final_sale_price:,.0f}")
    m7.metric("Sale costs", f"{cur}{res.sale_costs:,.0f}")
    m8.metric("Mortgage balance at sale", f"{cur}{res.outstanding_balance:,.0f}")

    st.markdown(f"**Breakeven annual price growth (simple, over {inputs.hold_years}y):** {res.breakeven_growth_simple:.2%}")
    # ---- Summary verdict box ----
    st.markdown("#### Summary verdict")
    basis = st.radio("Assess using", ["Simple (cash only)", "Opportunity-adjusted"], horizontal=True, key="basis_radio")
    if basis.startswith("Simple"):
        diff = res.net_cost_owning_simple - res.net_cost_renting_simple
        owning = res.net_cost_owning_simple
        renting = res.net_cost_renting_simple
    else:
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
        st.metric("Owning", f"{cur}{owning:,.0f}")
    
    with col2:
        st.metric("Renting", f"{cur}{renting:,.0f}")
    
    with col3:
        # Color coding for difference - green if positive, red if negative
        if abs(diff) <= eps:
            color = "#262730"  # default text color
        elif diff > 0:
            color = "#00C851"  # green for positive (owning more expensive)
        else:
            color = "#FF4444"  # red for negative (renting more expensive)
        
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
        
        st.markdown("**Buying Scenario Adjustments:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Down Payment Opportunity Cost", f"{cur}{down_payment_opp_cost:,.0f}")
            st.caption(f"What {cur}{deposit:,.0f} down payment would earn at {inputs.opportunity_rate:.1%}/year")
        with col2:
            st.metric("Buy Costs Opportunity Cost", f"{cur}{buy_costs_opp_cost:,.0f}")
            st.caption(f"What {cur}{buy_one_offs:,.0f} in buying fees would earn at {inputs.opportunity_rate:.1%}/year")
        
        st.markdown("**Renting Scenario Adjustments:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Alternative Investment Gains", f"{cur}{renter_alt_investment_gains:,.0f}")
            st.caption(f"Gains from investing {cur}{deposit + buy_one_offs:,.0f} at {inputs.opportunity_rate:.1%}/year")
        with col2:
            net_benefit_to_renter = renter_alt_investment_gains
            st.metric("Net Benefit to Renter", f"{cur}{net_benefit_to_renter:,.0f}")
            st.caption("Total investment gains reduce renting costs")
        
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
    
    

    # ---- Charts (remaining area) ----
    
st.download_button(
    "Download scenario JSON",
    data=str(inputs.__dict__),
    file_name="rent_vs_buy_scenario.json",
)
st.caption("This tool is a decision aid, not financial advice. Always verify tax/fee rules for your exact case.")
