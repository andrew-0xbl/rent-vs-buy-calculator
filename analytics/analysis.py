# Import required modules
from models import Inputs, Results
from finance.mortgage import monthly_payment
from finance.taxes import (
    sdlt_england_main_residence,
    hk_avd_scale2,
    hk_rates_annual_from_rent,
    hk_government_rent_annual,
)

# Import simulation function for breakeven calculations
from analytics.simulation import simulate, _compute_core


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
        owner_extra_monthly = (
            inputs.mgmt_fee_psf * inputs.net_area_sqft + inputs.service_charge
        )
        rv_annual = inputs.rent * 12
        rates_like_annual = (
            inputs.hk_rates_override_annual
            if inputs.hk_rates_override_annual is not None
            else hk_rates_annual_from_rent(rv_annual)
        )
        govrent_annual = (
            inputs.hk_govrent_override_annual
            if inputs.hk_govrent_override_annual is not None
            else hk_government_rent_annual(rv_annual)
        )

    maintenance_monthly = (inputs.maintenance_rate * inputs.price) / 12.0
    owner_running_monthly = (
        maintenance_monthly
        + owner_extra_monthly
        + (rates_like_annual / 12.0)
        + (govrent_annual / 12.0)
    )

    # Buy one-offs
    if inputs.jurisdiction.upper() == "UK":
        buy_one_offs = (
            sdlt_england_main_residence(inputs.price, surcharge=inputs.sdlt_surcharge)
            if inputs.sdlt_override is None
            else inputs.sdlt_override
        ) + inputs.buy_legal
    else:
        buy_one_offs = (
            (
                hk_avd_scale2(inputs.price)
                if inputs.sdlt_override is None
                else inputs.sdlt_override
            )
            + inputs.buy_legal
            + (inputs.agent_buy_rate * inputs.price)
        )

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


def breakeven_price_growth_simple(
    inputs: Inputs, tol=1e-6, lo=-0.20, hi=0.20, iters=60
):
    """Solve for annual price growth where SIMPLE net costs are equal: own == rent."""

    def diff_at(g):
        tmp = Inputs(**{**inputs.__dict__, "price_growth": g})
        r = _compute_core(tmp)
        return r.net_cost_owning_simple - r.net_cost_renting_simple

    a, b = lo, hi
    fa, fb = diff_at(a), diff_at(b)
    # Expand if needed
    k = 0
    while fa * fb > 0 and k < 8:
        a -= 0.05
        b += 0.05
        fa, fb = diff_at(a), diff_at(b)
        k += 1
    for _ in range(iters):
        m = 0.5 * (a + b)
        fm = diff_at(m)
        if abs(fm) < tol:
            return m
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def breakeven_price_growth_irr_nonnegative(
    inputs: Inputs, tol=1e-6, lo=-0.20, hi=0.20, iters=60
):
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
        a -= 0.05
        b += 0.05
        fa, fb = irr_at(a), irr_at(b)
        k += 1
        if k > 8:
            break
    for _ in range(iters):
        m = 0.5 * (a + b)
        fm = irr_at(m)
        if abs(fm) < 1e-5:
            return m
        if (fa <= 0 and fm >= 0) or (fa >= 0 and fm <= 0):
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)
