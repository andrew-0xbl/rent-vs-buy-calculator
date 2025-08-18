from dataclasses import dataclass
from typing import Optional

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
