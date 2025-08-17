# Rent vs Buy Calculator

A comprehensive financial analysis tool to help you decide whether to rent or buy property in the UK and Hong Kong markets. This interactive Streamlit application provides detailed cash flow analysis, opportunity cost calculations, and breakeven scenarios.

## Features

- **Dual Market Support**: Calculations for both UK and Hong Kong property markets with region-specific taxes and fees
- **Comprehensive Analysis**: Simple cash-only mode and opportunity-adjusted mode considering alternative investments
- **Interactive Interface**: Real-time calculations with adjustable parameters via sliders and inputs
- **Visual Analytics**: Charts showing cumulative costs and equity growth over time
- **Breakeven Analysis**: Automatic calculation of required property appreciation for buying to break even with renting

## Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip

## Installation & Setup

1. Clone or download this repository
2. Navigate to the project directory:
   ```bash
   cd rent_vs_buy
   ```

3. Install dependencies using uv (recommended):
   ```bash
   uv sync
   ```
   
   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

### Using uv (recommended):
```bash
uv run streamlit run app.py
```

### Using pip:
```bash
streamlit run app.py
```

The application will open in your default web browser, typically at `http://localhost:8501`.

## Usage

1. **Select Jurisdiction**: Choose between UK or Hong Kong to load appropriate defaults
2. **Configure Property Details**: Set purchase price, current rent, loan terms, and holding period
3. **Adjust Market Assumptions**: Configure growth rates for rent and property values
4. **Set Running Costs**: Input maintenance, service charges, and other ongoing expenses
5. **Review Analysis**: Compare simple cash flow vs. opportunity-adjusted costs
6. **Explore Scenarios**: Use the breakeven analysis to understand required appreciation rates

### Key Calculations

- **Simple Mode**: Direct cash flow comparison without considering opportunity costs
- **Opportunity-Adjusted Mode**: Factors in what your down payment could earn in alternative investments
- **Breakeven Growth**: Calculates the annual property appreciation needed for buying to match renting costs

## License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2024 Rent vs Buy Calculator

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Disclaimer

This tool is provided for educational and informational purposes only. It is not intended as financial advice. Always consult with qualified financial advisors and verify tax rules and fees for your specific situation before making property investment decisions.
