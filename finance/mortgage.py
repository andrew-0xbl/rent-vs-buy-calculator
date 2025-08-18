def monthly_payment(principal: float, annual_rate: float, years: int) -> float:
    n = years * 12
    if n <= 0:
        raise ValueError("years must be > 0")
    r = annual_rate / 12.0
    if abs(r) < 1e-12:
        return principal / n
    pow_ = (1 + r) ** n
    return principal * (r * pow_) / (pow_ - 1)


def remaining_balance(
    principal: float, annual_rate: float, years: int, months_paid: int
) -> float:
    n = years * 12
    if n <= 0:
        raise ValueError("years must be > 0")
    m = max(0, min(months_paid, n))
    r = annual_rate / 12.0

    if abs(r) < 1e-12:
        bal = principal * (1 - m / n)
        return max(0.0, bal)

    A = monthly_payment(principal, annual_rate, years)
    pow_m = (1 + r) ** m
    bal = principal * pow_m - A * ((pow_m - 1) / r)
    return max(0.0, bal)
