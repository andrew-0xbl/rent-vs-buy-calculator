# Import required modules
from models import Inputs, Results
from finance.mortgage import monthly_payment, remaining_balance
from finance.taxes import (
    sdlt_england_main_residence,
    hk_avd_scale2,
    hk_rates_annual_from_rent,
    hk_government_rent_annual,
)


def _compute_core(inputs: Inputs) -> Results:
    """Pure computation for a single scenario WITHOUT breakeven search (no recursion)."""
    months = inputs.hold_years * 12
    price_T = inputs.price * ((1 + inputs.price_growth) ** inputs.hold_years)
    deposit = inputs.price * (1 - inputs.ltv)
    loan = inputs.price - deposit

    if inputs.jurisdiction.upper() == "UK":
        tax_buy = (
            sdlt_england_main_residence(inputs.price, surcharge=inputs.sdlt_surcharge)
            if inputs.sdlt_override is None
            else inputs.sdlt_override
        )
        owner_extra_monthly = inputs.service_charge
        rates_like_annual = 0.0
        govrent_annual = 0.0
    else:
        tax_buy = (
            hk_avd_scale2(inputs.price)
            if inputs.sdlt_override is None
            else inputs.sdlt_override
        )
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

    agent_buy = inputs.agent_buy_rate * inputs.price
    buy_one_offs = tax_buy + inputs.buy_legal + agent_buy

    A = monthly_payment(loan, inputs.rate, inputs.term_years)
    bal_T = remaining_balance(loan, inputs.rate, inputs.term_years, months)
    principal_paid = loan - bal_T
    total_interest = A * months - principal_paid

    maintenance_monthly = (inputs.maintenance_rate * inputs.price) / 12.0
    hk_rates_monthly = rates_like_annual / 12.0
    hk_govrent_monthly = govrent_annual / 12.0
    owner_running_monthly = (
        maintenance_monthly
        + owner_extra_monthly
        + hk_rates_monthly
        + hk_govrent_monthly
    )
    owner_running_total = owner_running_monthly * months

    # Renting payments (with growth)
    rent_total = 0.0
    current_rent = inputs.rent
    for _ in range(inputs.hold_years):
        rent_total += current_rent * 12
        current_rent *= 1 + inputs.rent_growth

    # Selling costs and final equity
    sale_costs = inputs.sell_fee_rate * price_T
    equity_returned = max(0.0, price_T - sale_costs - bal_T)

    # SIMPLE mode: net out-of-pocket
    owner_outflow = (
        deposit + buy_one_offs + (A * months) + owner_running_total - equity_returned
    )
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
    down_payment_opportunity_cost = (
        deposit * ((1 + inputs.opportunity_rate) ** inputs.hold_years) - deposit
    )

    # 2. Buy-related one-off costs could have been invested
    buy_costs_opportunity_cost = (
        buy_one_offs * ((1 + inputs.opportunity_rate) ** inputs.hold_years)
        - buy_one_offs
    )

    # 3. Monthly mortgage + running costs opportunity cost
    owner_monthly_cost = A + owner_running_monthly
    owner_monthly_payments = [owner_monthly_cost] * months
    owner_monthly_opportunity_cost = future_value_stream(
        owner_monthly_payments, inputs.opportunity_rate
    ) - (owner_monthly_cost * months)

    # RENTING SCENARIO OPPORTUNITY COSTS:
    # 1. Monthly rent payments opportunity cost
    rent_payments = []
    current_rent = inputs.rent
    for month in range(1, months + 1):
        rent_payments.append(current_rent)
        if month % 12 == 0:
            current_rent *= 1 + inputs.rent_growth

    rent_opportunity_cost = future_value_stream(
        rent_payments, inputs.opportunity_rate
    ) - sum(rent_payments)

    # ALTERNATIVE INVESTMENT SCENARIO FOR RENTER:
    # Renter invests down payment + buy costs at opportunity rate
    renter_alternative_investment = (deposit + buy_one_offs) * (
        (1 + inputs.opportunity_rate) ** inputs.hold_years
    )

    # NET OPPORTUNITY-ADJUSTED COSTS:
    # Buying: base cost + opportunity costs - equity returned
    net_cost_owning_opp = (
        owner_outflow
        + down_payment_opportunity_cost
        + buy_costs_opportunity_cost
        + owner_monthly_opportunity_cost
    )

    # Renting: base cost + opportunity cost - alternative investment gains
    net_cost_renting_opp = (
        renter_outflow
        + rent_opportunity_cost
        - (renter_alternative_investment - deposit - buy_one_offs)
    )

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

    return Results(**{**base.__dict__, "breakeven_growth_simple": breakeven})
