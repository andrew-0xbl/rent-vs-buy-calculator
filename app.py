import streamlit as st
import pandas as pd
import altair as alt

from config import DEFAULT_VALUES
from models import Inputs

from finance.taxes import (
    sdlt_england_main_residence,
    hk_avd_scale2,
)
from analytics.simulation import simulate
from analytics.analysis import (
    build_owner_cashflows_for_irr,
    irr_annual_from_monthly_cfs,
    breakeven_price_growth_simple,
    breakeven_price_growth_irr_nonnegative,
)
from analytics.trajectories import (
    horizon_profile_dataframe,
    wealth_trajectories,
    equivalent_monthly_cost_from_fv,
    ownership_cost_breakdown,
)

st.set_page_config(
    page_title="Rent or Buy? (UK & Hong Kong Perspective)",
    page_icon="💰",
    layout="wide",
)

st.title("💰 Rent or Buy ? A UK and Hong Kong Perspective")

st.markdown(
    """
**Making the wrong choice could cost you £50,000+ over a decade.** This comprehensive calculator compares the true financial impact of renting versus buying, accounting for factors most people miss.

**Beyond the obvious costs,** we factor in opportunity costs on your down payment, hidden ownership expenses, and the real mathematics behind whether rent is "money down the drain." Get data-driven insights to make one of your biggest financial decisions with confidence.
"""
)

left, right = st.columns([1, 3], gap="large")

with left:
    st.markdown("### Inputs")
    jurisdiction = st.radio("Jurisdiction", ["UK", "HK"], horizontal=True)

    # Get defaults for selected jurisdiction
    defaults = DEFAULT_VALUES[jurisdiction]

    price = st.number_input(
        "Purchase price",
        min_value=0.0,
        value=defaults["price"],
        step=1000.0,
        format="%.0f",
        help="Total price of the property you're considering buying",
    )
    rent = st.number_input(
        "Current monthly rent",
        min_value=1.0,
        value=defaults["rent"],
        step=50.0,
        format="%.0f",
        help="Monthly rent for a comparable property to the one you're buying",
    )
    ltv = st.slider(
        "Loan-to-value (LTV)",
        0.0,
        0.95,
        defaults["ltv"],
        0.01,
        help="Percentage of purchase price financed by mortgage (e.g., 0.80 = 80% mortgage, 20% down payment)",
    )
    rate = (
        st.slider(
            "Mortgage rate (annual %)",
            0.0,
            10.0,
            defaults["rate"],
            0.05,
            help="Annual interest rate for your mortgage",
        )
        / 100.0
    )
    term_years = st.slider(
        "Mortgage term (years)",
        5,
        40,
        defaults["term_years"],
        1,
        help="Length of mortgage repayment period",
    )
    hold_years = st.slider(
        "Holding horizon (years)",
        1,
        30,
        defaults["hold_years"],
        1,
        help="How long you plan to own the property before selling",
    )

    rent_growth = (
        st.slider(
            "Annual rent growth (%)",
            0.0,
            10.0,
            defaults["rent_growth"],
            0.25,
            help="Expected annual increase in rental prices",
        )
        / 100.0
    )
    price_growth = (
        st.slider(
            "Annual price growth (%)",
            -10.0,
            10.0,
            defaults["price_growth"],
            0.25,
            help="Expected annual increase in property values",
        )
        / 100.0
    )
    opportunity_rate = (
        st.slider(
            "Opportunity rate (annual %, investable alt.)",
            0.0,
            10.0,
            defaults["opportunity_rate"],
            0.25,
            help="Rate of return you could earn by investing money elsewhere (e.g., stocks, bonds)",
        )
        / 100.0
    )

    st.markdown("#### Owner running & one-offs")
    if jurisdiction == "UK":
        maintenance_rate = (
            st.slider(
                "Maintenance (annual % of price)",
                0.0,
                3.0,
                defaults["maintenance_rate"],
                0.1,
                help="Annual maintenance costs as percentage of property value",
            )
            / 100.0
        )
        service_charge = st.number_input(
            "Service/ground/estate charges (monthly)",
            min_value=0.0,
            value=defaults["service_charge"],
            step=10.0,
            help="Monthly service charges, ground rent, or estate management fees",
        )
        sdlt_surcharge = (
            st.slider(
                "SDLT surcharge (additional property)",
                0.0,
                5.0,
                defaults["sdlt_surcharge"],
                0.5,
                help="Additional SDLT rate if this is not your main residence (usually 3%)",
            )
            / 100.0
        )
        buy_legal = st.number_input(
            "Buyer legal/surveys",
            min_value=0.0,
            value=defaults["buy_legal"],
            step=100.0,
            help="Legal fees, surveys, and other costs when buying",
        )
        sell_fee_rate = (
            st.slider(
                "Selling costs (% of sale price)",
                0.0,
                3.0,
                defaults["sell_fee_rate"],
                0.1,
                help="Estate agent fees and legal costs when selling (typically 1-2%)",
            )
            / 100.0
        )
        sdlt_override = st.number_input(
            "Override SDLT (0 = auto)",
            min_value=0.0,
            value=0.0,
            step=100.0,
            help="Manual SDLT amount (leave 0 for automatic calculation based on price)",
        )
        sdlt_override_val = None if sdlt_override == 0 else sdlt_override
        agent_buy_rate = 0.0
        mgmt_fee_psf = 0.0
        net_area_sqft = 0.0
        hk_rates_override_val = None
        hk_govrent_override_val = None
    else:
        maintenance_rate = (
            st.slider(
                "Maintenance (annual % of price)",
                0.0,
                3.0,
                defaults["maintenance_rate"],
                0.1,
                help="Annual maintenance costs as percentage of property value",
            )
            / 100.0
        )
        mgmt_fee_psf = st.number_input(
            "Management fee (HKD/ft²/mo)",
            min_value=0.0,
            value=defaults["mgmt_fee_psf"],
            step=0.1,
            help="Monthly management fees charged per square foot",
        )
        net_area_sqft = st.number_input(
            "Net area (ft²)",
            min_value=0.0,
            value=defaults["net_area_sqft"],
            step=10.0,
            help="Net floor area of the property in square feet",
        )
        buy_legal = st.number_input(
            "Buyer legal/misc (HKD)",
            min_value=0.0,
            value=defaults["buy_legal"],
            step=500.0,
            help="Legal fees and miscellaneous costs when buying",
        )
        agent_buy_rate = (
            st.slider(
                "Buyer agent fee (% of price)",
                0.0,
                2.0,
                defaults["agent_buy_rate"],
                0.1,
                help="Real estate agent commission when buying (typically 1%)",
            )
            / 100.0
        )
        sell_fee_rate = (
            st.slider(
                "Selling costs (% of sale price)",
                0.0,
                3.0,
                defaults["sell_fee_rate"],
                0.1,
                help="Agent fees and legal costs when selling (typically 2%)",
            )
            / 100.0
        )
        sdlt_override = st.number_input(
            "Override AVD (Simplified) (0 = auto)",
            min_value=0.0,
            value=0.0,
            step=1000.0,
            help="Manual Ad Valorem Duty amount (leave 0 for automatic calculation)",
        )

        # HK Rates & Government Rent toggle
        auto_rates = st.checkbox(
            "Auto-calc Rates & Gov't Rent from rent",
            value=True,
            help="If checked, uses current rent to estimate rateable value. Otherwise, enter annual amounts manually.",
        )

        if auto_rates:
            hk_rates_override_val = None
            hk_govrent_override_val = None
            st.caption(
                "Using auto: Rates ≈ stepped % of RV; Gov't Rent = 3% of RV. RV derived from current rent."
            )
        else:
            hk_rates_override = st.number_input(
                "HK Rates (annual)", min_value=0.0, value=0.0, step=100.0
            )
            hk_govrent_override = st.number_input(
                "Gov't Rent (annual)", min_value=0.0, value=0.0, step=100.0
            )
            hk_rates_override_val = hk_rates_override
            hk_govrent_override_val = hk_govrent_override

        sdlt_override_val = None if sdlt_override == 0 else sdlt_override
        service_charge = 0.0
        sdlt_surcharge = 0.0

    inputs = Inputs(
        jurisdiction=jurisdiction,
        price=price,
        rent=rent,
        ltv=ltv,
        rate=rate,
        term_years=term_years,
        hold_years=hold_years,
        rent_growth=rent_growth,
        price_growth=price_growth,
        maintenance_rate=maintenance_rate,
        service_charge=service_charge,
        mgmt_fee_psf=mgmt_fee_psf,
        net_area_sqft=net_area_sqft,
        buy_legal=buy_legal,
        agent_buy_rate=agent_buy_rate,
        sell_fee_rate=sell_fee_rate,
        sdlt_surcharge=sdlt_surcharge,
        sdlt_override=sdlt_override_val,
        hk_rates_override_annual=hk_rates_override_val,
        hk_govrent_override_annual=hk_govrent_override_val,
        opportunity_rate=opportunity_rate,
    )

with right:
    res = simulate(inputs)
    cur = "£" if jurisdiction == "UK" else "$"
    # ---- Summary verdict box ----
    st.markdown("### Summary verdict")
    basis = st.radio(
        "Assess using",
        ["Simple (cash only)", "Opportunity-adjusted"],
        horizontal=True,
        key="basis_radio",
    )

    if basis.startswith("Simple"):
        st.caption(
            "Simple Mode: Economic profit (excludes principal as expense; includes interest, running, buy/sell costs; no discounting; pre-tax)."
        )
    else:
        st.caption(
            "Opportunity-Adjusted Mode: Economic profit with opportunity costs on tied-up capital and rent payments. All opportunity adjustments are expressed as end-of-horizon future values (no discounting)."
        )

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
        st.success("**Renting is cheaper**")
    else:
        st.error("**Buying is cheaper**")

    # Three metric boxes in a single row
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Owning",
            f"{cur}{owning:,.0f}",
            help=f"Total net cost of owning over {inputs.hold_years} years",
        )

    with col2:
        st.metric(
            "Renting",
            f"{cur}{renting:,.0f}",
            help=f"Total cost of renting over {inputs.hold_years} years",
        )

    with col3:
        if abs(diff) <= eps:
            color = "#262730"
        elif diff > 0:
            color = "#00C851"
        else:
            color = "#FF4444"

        st.markdown(
            f'<p style="font-size:14px; color:#8892b0; margin:0;">Difference</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p style="font-size:36px; font-weight:600; color:{color}; margin:0; line-height:1;">{cur}{diff:,.0f}</p>',
            unsafe_allow_html=True,
        )

    # ---- Opportunity Cost Breakdown (when in opportunity mode) ----
    if not basis.startswith("Simple"):
        st.markdown("### 🔍 Opportunity Cost Analysis")
        st.markdown("**Understanding the Opportunity-Adjusted Calculation:**")

        # Calculate the individual components for display
        deposit = inputs.price * (1 - inputs.ltv)

        # Recalculate components for display (matching the calculation logic)
        if inputs.jurisdiction.upper() == "UK":
            buy_one_offs = (
                sdlt_england_main_residence(
                    inputs.price, surcharge=inputs.sdlt_surcharge
                )
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

        down_payment_opp_cost = (
            deposit * ((1 + inputs.opportunity_rate) ** inputs.hold_years) - deposit
        )
        buy_costs_opp_cost = (
            buy_one_offs * ((1 + inputs.opportunity_rate) ** inputs.hold_years)
            - buy_one_offs
        )

        renter_alt_investment_total = (deposit + buy_one_offs) * (
            (1 + inputs.opportunity_rate) ** inputs.hold_years
        )
        renter_alt_investment_gains = renter_alt_investment_total - (
            deposit + buy_one_offs
        )

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
                current_rent *= 1 + inputs.rent_growth

        rent_opportunity_cost_display = fv_rent_stream - sum_rent
        net_benefit_to_renter = (
            renter_alt_investment_gains - rent_opportunity_cost_display
        )

        st.markdown("**Buying Scenario Adjustments:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Down Payment Opportunity Cost", f"{cur}{down_payment_opp_cost:,.0f}"
            )
            st.caption(
                f"What {cur}{deposit:,.0f} down payment would earn at {inputs.opportunity_rate:.1%}/year"
            )
        with col2:
            st.metric("Buy Costs Opportunity Cost", f"{cur}{buy_costs_opp_cost:,.0f}")
            st.caption(
                f"What {cur}{buy_one_offs:,.0f} in buying fees would earn at {inputs.opportunity_rate:.1%}/year"
            )

        st.markdown("**Renting Scenario Adjustments:**")
        # Show both components + net
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Investment gains on deposit + fees",
                f"{cur}{renter_alt_investment_gains:,.0f}",
            )
        with col2:
            st.metric(
                "Opportunity cost on rent stream",
                f"{cur}{rent_opportunity_cost_display:,.0f}",
            )
        with col3:
            st.metric(
                "Net Benefit to Renter (opportunity)",
                f"{cur}{net_benefit_to_renter:,.0f}",
            )
        st.caption(
            "Net = gains from investing the initial lump sum minus the opportunity cost applied to the rent stream."
        )

        st.markdown("---")
        st.markdown("**Interpretation:**")
        if inputs.opportunity_rate > 0:
            st.markdown(
                f"""
            - **Buying** becomes more expensive when you consider that your down payment and fees could earn {inputs.opportunity_rate:.1%}/year elsewhere
            - **Renting** becomes relatively cheaper because you can invest the down payment money instead
            - The opportunity rate of {inputs.opportunity_rate:.1%}/year represents your best alternative investment (e.g., index funds, bonds)
            """
            )
        else:
            st.markdown(
                "- With 0% opportunity rate, both modes show the same results (no alternative investment consideration)"
            )

    # ---- Breakdown (top area) ----
    st.markdown("### Breakdown")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Deposit",
        f"{cur}{inputs.price*(1-inputs.ltv):,.0f}",
        border=True,
        help="Down payment required upfront",
    )
    m2.metric(
        "Upfront taxes & fees",
        f"{cur}{res.buy_one_offs:,.0f}",
        border=True,
        help="One-time costs when purchasing: stamp duty, legal fees, surveys",
    )
    m3.metric(
        "Interest paid (horizon)",
        f"{cur}{res.interest_paid:,.0f}",
        border=True,
        help=f"Total mortgage interest payments over {inputs.hold_years} years",
    )
    m4.metric(
        "Principal repaid (horizon)",
        f"{cur}{res.principal_paid:,.0f}",
        border=True,
        help=f"Mortgage principal paid down over {inputs.hold_years} years",
    )

    m5, m6, m7, m8 = st.columns(4)
    m5.metric(
        "Owner running (horizon)",
        f"{cur}{res.owner_running:,.0f}",
        border=True,
        help=f"Ongoing costs over {inputs.hold_years} years: maintenance, service charges, insurance",
    )
    m6.metric(
        "Estimated Sale costs",
        f"{cur}{res.sale_costs:,.0f}",
        border=True,
        help="Estate agent fees and legal costs when selling",
    )
    m7.metric(
        "Mortgage balance at sale",
        f"{cur}{res.outstanding_balance:,.0f}",
        border=True,
        help=f"Remaining mortgage debt after {inputs.hold_years} years",
    )

    # Sale Analysis Section
    st.markdown("### Sale Analysis")
    st.caption(
        "Economic profit shown above treats principal as equity, not expense (no discounting, pre-tax). Equity IRR uses cash flows (including principal timing) and provides a time-weighted return on equity."
    )

    # Calculate percentage changes for deltas
    price_change_pct = ((res.final_sale_price - inputs.price) / inputs.price) * 100
    net_sale_change_pct = ((res.net_sale_price - inputs.price) / inputs.price) * 100
    net_gain = res.net_gain_after_sale(inputs.price, inputs.ltv)

    # Equity contributed = deposit + cumulative principal repaid
    deposit = inputs.price * (1 - inputs.ltv)
    equity_contributed = deposit + res.principal_paid
    ending_equity = res.net_sale_price - res.outstanding_balance
    equity_multiple = (
        (ending_equity / equity_contributed) if equity_contributed > 0 else float("nan")
    )

    # IRR on equity cash flows
    owner_cfs = build_owner_cashflows_for_irr(inputs, res)
    equity_irr_annual = irr_annual_from_monthly_cfs(owner_cfs)

    s1, s2, s3 = st.columns(3)
    s1.metric(
        "Sale price (horizon)",
        f"{cur}{res.final_sale_price:,.0f}",
        delta=f"{price_change_pct:+.1f}%",
        border=True,
        help=f"Expected property value after {inputs.hold_years} years of growth",
    )
    s2.metric(
        "Net sale price",
        f"{cur}{res.net_sale_price:,.0f}",
        delta=f"{net_sale_change_pct:+.1f}%",
        border=True,
        help="Sale price after deducting selling costs",
    )
    s3.metric(
        "Net gain after sale",
        f"{cur}{net_gain:,.0f}",
        delta=f"{(net_gain/inputs.price)*100:+.1f}%" if net_gain != 0 else "0.0%",
        border=True,
        help="Economic profit (excludes principal as expense; includes interest, running, buy/sell costs; no discounting; pre-tax)",
    )

    t1, t2 = st.columns(2)

    t1.metric(
        "Equity Multiple",
        f"{equity_multiple:,.2f}",
        border=True,
        help="A simple measure of how much money an investment returns relative to the initial equity invested. Calculated using (Ending equity) / (Deposit + Principal repaid)",
    )

    # Add a new row for IRR
    t2.metric(
        "Equity IRR (annual)",
        f"{equity_irr_annual:.2%}",
        help="IRR computed on owner equity cash flows (principal treated as capital, not expense). Equity IRR reflects actual cash flow timing (principal included in cash flows), annualised.",
        border=True,
    )

    st.markdown(
        f"**Breakeven annual price growth (simple, over {inputs.hold_years}y):** {res.breakeven_growth_simple:.2%}"
    )

    # ---- Horizon vs Return Chart ----
    st.markdown("### Return Profile vs. Holding Horizon")
    hp_df = horizon_profile_dataframe(inputs, max_years=30)
    st.line_chart(hp_df.set_index("Horizon (yrs)"), use_container_width=True)
    st.caption(
        "Equity IRR is annualised and cashflow-based (includes principal timing). Net Gain After Sale is an economic profit (% of purchase price) that excludes principal as an expense but includes interest, running, and transaction costs."
    )

    # ---- Breakeven Calculators ----
    st.markdown("### Breakeven Calculators")
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        g_simple = breakeven_price_growth_simple(inputs)
        st.metric(
            "Breakeven price growth (Simple)",
            f"{g_simple:.2%}",
            help="Annual property growth where simple net costs are equal (own = rent).",
            border=True,
        )
    with bcol2:
        g_irr0 = breakeven_price_growth_irr_nonnegative(inputs)
        st.metric(
            "Breakeven price growth (IRR = 0%)",
            f"{g_irr0:.2%}",
            help="Annual property growth where equity IRR crosses 0%.",
            border=True,
        )

    # ---- Consumption Premium & Equivalent Monthly Cost ----
    st.markdown("### Consumption Premium & Equivalent Monthly Cost")
    months = inputs.hold_years * 12
    # Use OPPORTUNITY-ADJUSTED totals for EAC on FV basis (consistent with your app's FV opportunity framing)
    own_fv = (
        res.net_cost_owning_opp
        if res.net_cost_owning_opp is not None
        else res.net_cost_owning_simple
    )
    rent_fv = (
        res.net_cost_renting_opp
        if res.net_cost_renting_opp is not None
        else res.net_cost_renting_simple
    )
    eac_own = equivalent_monthly_cost_from_fv(own_fv, months, inputs.opportunity_rate)
    eac_rent = equivalent_monthly_cost_from_fv(rent_fv, months, inputs.opportunity_rate)
    consumption_premium = (
        eac_own - eac_rent
    )  # >0 means paying extra per month to own vs rent

    p1, p2, p3, p4 = st.columns(4)
    cur = "£" if inputs.jurisdiction == "UK" else "$"
    with p1:
        st.metric(
            "Equivalent Monthly Cost (Owning)", f"{cur}{eac_own:,.0f}", border=True
        )
    with p2:
        st.metric(
            "Equivalent Monthly Cost (Renting)", f"{cur}{eac_rent:,.0f}", border=True
        )
    with p3:
        st.metric(
            "Consumption Premium (per month)",
            f"{cur}{consumption_premium:,.0f}",
            help="Positive = you pay this extra per month to own (for stability/lifestyle).",
            border=True,
        )
    with p4:
        # Calculate percentage difference
        if eac_rent != 0:
            premium_pct = (consumption_premium / eac_rent) * 100
        else:
            premium_pct = 0

        st.markdown(
            f'<p style="font-size:14px; color:#8892b0; margin:0;">Own Premium Percentage</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p style="font-size:28px; font-weight:600; margin:0; line-height:1;">{premium_pct:.1f}%</p>',
            unsafe_allow_html=True,
        )

    # ---- Charts Row: Wealth Trajectory and Ownership Cost Mix ----
    chart_left, chart_right = st.columns([2, 1], gap="medium")

    with chart_left:
        st.markdown("### Wealth Trajectory (Rent & Invest vs Buy)")
        wdf = wealth_trajectories(inputs)

        # Prepare data for stacked bar chart
        chart_data = pd.DataFrame(
            {
                "Years": wdf["Years"],
                "Buy Net Worth": wdf["Buy_NetWorth"],
                "Rent Investor Wealth FV": wdf["Rent_Investor_Wealth_FV"],
            }
        )

        # Melt data for stacked bar chart
        melted_data = pd.melt(
            chart_data,
            id_vars=["Years"],
            value_vars=["Buy Net Worth", "Rent Investor Wealth FV"],
            var_name="Scenario",
            value_name="Wealth",
        )

        # Create grouped bar chart using Altair
        bar_chart = (
            alt.Chart(melted_data)
            .mark_bar()
            .encode(
                x=alt.X("Years:O", title="Years", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Wealth:Q", title="Wealth", axis=alt.Axis(format=",.0f")),
                color=alt.Color(
                    "Scenario:N",
                    scale=alt.Scale(range=["#FF6B6B", "#4ECDC4"]),
                    legend=alt.Legend(orient="left", titleLimit=0),
                ),
                xOffset="Scenario:N",
                tooltip=[
                    alt.Tooltip("Years:O"),
                    alt.Tooltip("Scenario:N"),
                    alt.Tooltip("Wealth:Q", format=",.0f"),
                ],
            )
            .properties(height=400)
        )

        st.altair_chart(bar_chart, use_container_width=True)
        st.caption(
            "Buy Net Worth = property value − mortgage balance (no sale costs mid-hold)."
        )
        st.caption(
            "Rent Investor Wealth FV = FV of deposit+fees invested plus invested monthly savings, at the opportunity rate."
        )

    with chart_right:
        st.markdown("### Ownership Cost Mix")
        odf = ownership_cost_breakdown(inputs, res)

        # Create pie chart with diverse color palette
        colors = ["#ff6e61", "#708dff", "#70ffb0", "#ffd900", "#ff4d4d"]

        pie_chart = (
            alt.Chart(odf)
            .mark_arc(innerRadius=50, outerRadius=120)
            .encode(
                theta=alt.Theta("Amount:Q"),
                color=alt.Color(
                    "Category:N",
                    scale=alt.Scale(range=colors),
                    legend=alt.Legend(orient="left", titleLimit=0, labelLimit=0),
                ),
                tooltip=["Category:N", alt.Tooltip("Amount:Q", format=",.0f")],
            )
            .resolve_scale(color="independent")
        )

        st.altair_chart(pie_chart, use_container_width=False)

    # ---- Sensitivity Analysis ----
    st.markdown("### Sensitivity — Annual Price Growth")

    # Slider
    g_sens = (
        st.slider(
            "Test price growth (%)", -5.0, 8.0, float(inputs.price_growth * 100), 0.25
        )
        / 100.0
    )

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
        chart_data = pd.DataFrame(
            {"Year": wdf_s["Years"], "Buy Net Worth": wdf_s["Buy_NetWorth"]}
        )

        bar_chart = (
            alt.Chart(chart_data)
            .mark_bar(color="#45B7D1")
            .encode(
                x=alt.X("Year:O", title="Year", axis=alt.Axis(labelAngle=0)),
                y=alt.Y(
                    "Buy Net Worth:Q",
                    title="Buy Net Worth",
                    axis=alt.Axis(format=",.0f"),
                ),
                tooltip=[
                    alt.Tooltip("Year:O"),
                    alt.Tooltip("Buy Net Worth:Q", format=",.0f"),
                ],
            )
            .properties(height=300)
        )

        st.altair_chart(bar_chart, use_container_width=True)

    with sens_right:
        # Two metrics in separate rows
        st.metric("Equity IRR (sens)", f"{tmp_irr:.2%}", border=True)
        st.metric(
            "Net gain after sale (sens)",
            f"{tmp_res.net_gain_after_sale(tmp_inputs.price, tmp_inputs.ltv)/tmp_inputs.price*100:.2f}%",
            border=True,
        )

# ---- FAQ Section ----
st.markdown("---")
st.markdown("## FAQ")
with st.expander("How are the calculations performed?"):
    # Simple mode formulas
    st.markdown("**Simple Mode (Cash-Only):**")
    st.markdown(
        "This mode just adds up actual cash flows. It does not account for what your money could have earned elsewhere."
    )
    st.latex(
        r"""
    \text{Net Cost Owning} = \text{Deposit} + \text{Buy Fees} + \text{Mortgage} + \text{Running Costs} - \text{Equity Returned}
    """
    )
    st.latex(
        r"""
    \text{Net Cost Renting} = \sum_{t=1}^{T} Rent_t \cdot (1 + g_r)^{t-1}
    """
    )
    st.caption(
        "⚠️ Caveat: assumes fixed mortgage payments, no early repayment, and rent grows steadily at rate $g_r$."
    )

    st.markdown("---")

    # Opportunity cost methodology
    st.markdown("**Opportunity-Adjusted Mode (Investment Alternative):**")
    st.markdown(
        """
    This mode also considers *opportunity cost* — what your money could have earned if invested instead of tied up in property.
    """
    )

    st.markdown("**Buying Scenario Opportunity Costs:**")
    st.latex(
        r"""
    \text{Deposit OC} = \text{Deposit} \cdot [(1 + r_{opp})^T - 1]
    """
    )
    st.latex(
        r"""
    \text{Fees OC} = \text{Buy Fees} \cdot [(1 + r_{opp})^T - 1]
    """
    )
    st.latex(
        r"""
    \text{Monthly OC} = FV(\text{Payments}, r_{opp}) - \sum \text{Payments}
    """
    )

    st.markdown("**Renting Scenario:**")
    st.latex(
        r"""
    \text{Rent OC} = FV(\text{Rent}, r_{opp}) - \sum \text{Rent}
    """
    )
    st.latex(
        r"""
    \text{Alt Invest Gain} = (\text{Deposit} + \text{Fees}) \cdot (1 + r_{opp})^T
    """
    )

    st.markdown("**Final Opportunity-Adjusted Costs:**")
    st.latex(
        r"""
    \text{Cost}_{own}^{adj} = \text{Cost}_{own} + \text{All Buying OCs}
    """
    )
    st.latex(
        r"""
    \text{Cost}_{rent}^{adj} = \text{Cost}_{rent} + \text{Rent OC} - \text{Alt Invest Gain}
    """
    )

    st.markdown("**Where:**")
    st.markdown("- $r_{opp}$ = annual opportunity rate")
    st.markdown("- $T$ = holding period (years)")
    st.markdown("- $FV(·)$ = future value of payments")
    st.caption(
        "⚠️ Caveat: assumes constant $r_{opp}$, no investment volatility, and reinvestment at same rate."
    )

with st.expander("What's the difference between UK and Hong Kong modes?"):
    st.markdown("**UK Mode includes:**")
    st.markdown("- SDLT (Stamp Duty Land Tax, main residence rates)")
    st.markdown("- Optional SDLT surcharge for extra properties")
    st.markdown("- Service charges, ground rent, estate fees")
    st.markdown("- Legal fees and surveys")

    st.markdown("**Hong Kong Mode includes:**")
    st.markdown("- AVD (Ad Valorem Duty) using Scale 2 rates")
    st.markdown("- Management fees (per square foot) + service charge")
    st.markdown("- Government rent and property rates")
    st.markdown("- Buyer’s agent commission")
    st.caption(
        "⚠️ Caveat: does not cover all tax types (e.g. BSD, SSD in HK; reliefs in UK). Simplified for typical buyers."
    )

with st.expander("What does 'opportunity rate' mean?"):
    st.markdown(
        """
    The opportunity rate ($r_{opp}$) is the annual return you think you could earn by investing instead of buying.
    
    **Examples:**
    - 3–4%: government bonds, savings
    - 5–7%: index funds
    - 8–10%: stock market (higher risk)
    """
    )
    st.caption("⚠️ Caveat: assumes stable returns and no investment losses.")

with st.expander("How should I interpret the results?"):
    st.markdown(
        """
    - **Simple Mode:** Out-of-pocket comparison only.  
    - **Opportunity-Adjusted:** Adds the “what if I had invested instead?” factor.  
    - **Verdict:** The cheaper option under your assumptions is shown, but remember: lifestyle, risk, and stability also matter.
    """
    )
    st.caption("⚠️ Caveat: results are sensitive to assumptions (rates, growth, costs).")

with st.expander("What is Equity IRR and how is it calculated?"):
    st.markdown(
        """
    **Equity IRR** = annualised return on your equity in the property, considering timing of all inflows/outflows.
    
    Includes:
    - Deposit + fees at purchase (negative)
    - Monthly mortgage + running costs (negative)
    - Principal repayments (positive — builds equity)
    - Sale proceeds net of mortgage (positive)
    """
    )
    st.latex(
        r"""
    \text{IRR} = r \ \text{ where } \sum_{t=0}^{N} \frac{CF_t}{(1+r)^t} = 0
    """
    )
    st.markdown("Where $CF_t$ = monthly cash flow.")
    st.caption(
        "⚠️ Caveat: assumes you could reinvest at the same rate as IRR, which is rarely realistic."
    )

with st.expander("What are Breakeven Price Growth rates?"):
    st.markdown(
        """
    These show how much property prices must rise (per year) for buying to match renting:
    
    - **Breakeven (Simple):** Total owning cost = renting cost.  
    - **Breakeven (IRR = 0%):** Property price growth that makes IRR exactly 0%.  
    """
    )
    st.caption(
        "⚠️ Caveat: assumes smooth price growth, no shocks, and ignores taxes on sale."
    )

with st.expander("What is Consumption Premium and Equivalent Monthly Cost?"):
    st.markdown(
        """
    These express results as a “monthly bill” so it’s easier to compare with rent:
    
    - **Equivalent Monthly Cost (EMC):** Turns all future costs into a monthly equivalent.  
    """
    )
    st.latex(
        r"""
    EMC = \frac{FV \cdot r}{(1+r)^{12T} - 1} \cdot \tfrac{1}{12}
    """
    )
    st.markdown(
        """
    - **Consumption Premium:** The extra monthly cost of owning vs renting, i.e. what you pay for stability and control.  
    """
    )
    st.caption(
        "⚠️ Caveat: EMC is not actual cash flow — it’s an equivalent, assumes fixed $r$ and horizon $T$."
    )

with st.expander("How is Wealth Trajectory calculated?"):
    st.markdown(
        """
    Compares how your wealth changes under each scenario:
    
    **Buy Scenario:**
    """
    )
    st.latex(
        r"""
    \text{Wealth}_{buy} = \text{Property Value} - \text{Mortgage Balance}
    """
    )
    st.markdown(
        """
    **Rent + Invest Scenario:**
    """
    )
    st.latex(
        r"""
    \text{Wealth}_{rent} = FV(\text{Deposit + Fees}) + FV(\text{Monthly Savings})
    """
    )
    st.caption(
        "⚠️ Caveat: assumes steady compounding, ignores market downturns, ignores taxes on investment gains."
    )

st.caption(
    "This tool is a decision aid, not financial advice. It assumes constant rates, simplified fees/taxes, no personal taxation, no shocks (repairs, vacancy), and reinvestment at fixed $r_{opp}$."
)
