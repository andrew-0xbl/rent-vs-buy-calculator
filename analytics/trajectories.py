# Import required modules
import pandas as pd
from models import Inputs, Results
from finance.mortgage import monthly_payment
from finance.taxes import (
    sdlt_england_main_residence,
    hk_avd_scale2,
    hk_rates_and_govrent_from_rent_monthly,
)

# Import analytics functions
from analytics.simulation import simulate
from analytics.analysis import (
    build_owner_cashflows_for_irr,
    irr_annual_from_monthly_cfs,
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

    return pd.DataFrame(
        {
            "Horizon (yrs)": horizons,
            "Equity IRR (annual %)": irr_vals,
            "Net Gain After Sale (%)": gain_vals_pct,
        }
    )


def wealth_trajectories(inputs: Inputs):
    """Return DataFrame with years, Buy_NetWorth, Rent_Investor_Wealth (FV basis and level cash).
    Buy_NetWorth(t): property_value(t) - mortgage_balance(t) - (optional sale cost fraction at t -> 0 mid-hold).
    Rent_Investor_Wealth(t): FV of initial (deposit+fees) plus FV of monthly deltas (rent cheaper than own).
    We keep sale costs out during hold (only at terminal) to avoid artificial dips mid-hold.
    """
    months = inputs.hold_years * 12
    dep = inputs.price * (1 - inputs.ltv)
    loan = inputs.price - dep
    r_m = inputs.rate / 12.0
    opp_m = inputs.opportunity_rate / 12.0

    # Property path
    price_path = [
        inputs.price * ((1 + inputs.price_growth) ** (t / 12.0))
        for t in range(months + 1)
    ]

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
        rates_m = 0.0
        govrent_m = 0.0
    else:
        rates_m, govrent_m = (
            hk_rates_and_govrent_from_rent_monthly(inputs.rent)
            if (
                inputs.hk_rates_override_annual is None
                and inputs.hk_govrent_override_annual is None
            )
            else (
                (inputs.hk_rates_override_annual or 0.0) / 12.0,
                (inputs.hk_govrent_override_annual or 0.0) / 12.0,
            )
        )
        owner_extra_monthly = (
            inputs.mgmt_fee_psf * inputs.net_area_sqft + inputs.service_charge
        )

    maint_m = (inputs.maintenance_rate * inputs.price) / 12.0
    owner_running_m = maint_m + owner_extra_monthly + rates_m + govrent_m

    # Buy net worth path (in level dollars, not FV) during hold:
    buy_networth = []
    for t in range(months + 1):
        eq = price_path[t] - balances[t]
        buy_networth.append(eq)

    # Rent investor wealth path (FV basis):
    # FV of initial (dep + buy_one_offs)
    if inputs.jurisdiction.upper() == "UK":
        tax_buy = (
            sdlt_england_main_residence(inputs.price, surcharge=inputs.sdlt_surcharge)
            if inputs.sdlt_override is None
            else inputs.sdlt_override
        )
        buy_one_offs = tax_buy + inputs.buy_legal
    else:
        tax_buy = (
            hk_avd_scale2(inputs.price)
            if inputs.sdlt_override is None
            else inputs.sdlt_override
        )
        buy_one_offs = tax_buy + inputs.buy_legal + inputs.agent_buy_rate * inputs.price

    lump = dep + buy_one_offs
    wealth_rent = []
    FV_lump = lump  # will compound forward
    rent_level = inputs.rent
    for t in range(months + 1):
        # compound lump
        if t > 0:
            FV_lump *= 1 + opp_m
        # monthly delta (if renting cheaper than owning)
        owner_monthly = A + owner_running_m
        monthly_saving = max(0.0, owner_monthly - rent_level)  # money renter can invest
        # invest saving at end of each month
        if t > 0:
            FV_lump += monthly_saving
        wealth_rent.append(FV_lump)
        # step rent annually
        if t > 0 and t % 12 == 0:
            rent_level *= 1 + inputs.rent_growth

    # Sample data points at year boundaries only (whole numbers)
    year_indices = [t for t in range(months + 1) if t % 12 == 0]  # 0, 12, 24, 36, etc.
    year_numbers = [t // 12 for t in year_indices]  # 0, 1, 2, 3, etc.
    year_buy_networth = [buy_networth[i] for i in year_indices]
    year_wealth_rent = [wealth_rent[i] for i in year_indices]

    df = pd.DataFrame(
        {
            "Years": year_numbers,  # Whole number years
            "Buy_NetWorth": year_buy_networth,
            "Rent_Investor_Wealth_FV": year_wealth_rent,
        }
    )
    return df


def equivalent_monthly_cost_from_fv(
    total_fv_cost: float, months: int, opp_annual: float
):
    """Convert end-of-horizon FV cost into a level monthly 'equivalent' using the FV annuity factor."""
    r = opp_annual / 12.0
    if months == 0:
        return 0.0
    if r == 0:
        return total_fv_cost / months
    fv_annuity_factor = ((1 + r) ** months - 1) / r
    return total_fv_cost / fv_annuity_factor


# --- A5. Ownership cost breakdown for stacked bar (small chart) ---


def ownership_cost_breakdown(inputs: Inputs, res: Results):
    """Return DataFrame with categories and amounts for pie chart."""
    # Create data with categories and amounts
    data = {
        "Category": [
            "Deposit",
            "Buy taxes & legal",
            "Interest",
            "Maintenance",
            "Sale costs",
        ],
        "Amount": [
            inputs.price * (1 - inputs.ltv),
            res.buy_one_offs,
            res.interest_paid,
            res.owner_running,
            res.sale_costs,
        ],
    }
    return pd.DataFrame(data)
