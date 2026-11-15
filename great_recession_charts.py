"""
Great Recession charts (2008) for PowerPoint.

Downloads eight series from FRED's public CSV endpoint (no API key needed),
draws six charts in one consistent style, saves them as PNGs in ./charts,
and prints a table of the key numbers used in the annotations.

How to rerun:
    pip install pandas matplotlib
    python great_recession_charts.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw straight to files; no window needed
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------------------------------------------------------
# Settings shared by every chart
# ----------------------------------------------------------------------------
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
OUT_DIR = Path(__file__).resolve().parent / "charts"

START, END = "2005-01-01", "2012-12-31"            # default date window
REC_START, REC_END = pd.Timestamp("2007-12-01"), pd.Timestamp("2009-06-30")
REC_LABEL = "Recession (Dec 2007 to Jun 2009)"
BASE_DATE = pd.Timestamp("2007-12-01")              # start of the recession

NAVY, RED, GRAY = "#1E2A44", "#B3261E", "#6B6B6B"
FIGSIZE, DPI = (8, 5.75), 200

key_numbers = []  # rows for the summary table printed at the end


# ----------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------
def fetch(series_id):
    """Download one FRED series and return it as a float Series indexed by date."""
    # FRED marks missing values with "."
    df = pd.read_csv(FRED_CSV.format(series_id), na_values=".")
    # The date column has been called both "DATE" and "observation_date";
    # it is always the first column, and the values are always the second.
    dates = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    values = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s = pd.Series(values.to_numpy(), index=dates, name=series_id)
    return s[s.index.notna()].dropna().sort_index()


def month_label(date):
    return date.strftime("%b %Y")


def quarter_label(date):
    return f"Q{date.quarter} {date.year}"


def record(figure, item, date, value):
    """Remember a number for the summary table."""
    key_numbers.append({"Figure": figure, "Item": item, "Date": date, "Value": value})


# ----------------------------------------------------------------------------
# Style helpers
# ----------------------------------------------------------------------------
def new_chart(title, ylabel, series_ids, start=START, end=END):
    """Create a figure with the shared style, recession shading and footnote."""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, facecolor="white")
    ax.set_facecolor("white")

    ax.set_title(title, fontsize=14, fontweight="bold", color=NAVY, pad=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel("Year", fontsize=11)
    ax.tick_params(labelsize=10, length=0)

    # Light horizontal gridlines only, no top/right spines
    ax.grid(axis="y", color="#E3E3E3", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9A9A9A")

    # One tick per year (every two years on the longer home-price chart)
    span_years = pd.Timestamp(end).year - pd.Timestamp(start).year
    ax.xaxis.set_major_locator(mdates.YearLocator(2 if span_years > 10 else 1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(pd.Timestamp(start), pd.Timestamp(end))

    # NBER recession shading with a small label at the top of the band
    ax.axvspan(REC_START, REC_END, color="#E6E6E6", zorder=0, linewidth=0)
    ax.text(
        REC_START + (REC_END - REC_START) / 2, 0.985, REC_LABEL,
        transform=ax.get_xaxis_transform(), ha="center", va="top",
        fontsize=7, color=GRAY,
    )

    # Source footnote, bottom-left
    fig.text(
        0.01, 0.005,
        f"Source: FRED, Federal Reserve Bank of St. Louis ({', '.join(series_ids)})",
        fontsize=8, color="gray", ha="left", va="bottom",
    )
    return fig, ax


def annotate(ax, date, value, text, offset, color=RED):
    """Mark a point and label it with a small arrow. offset = (dx, dy) in points."""
    ax.plot([date], [value], "o", color=color, markersize=5, zorder=5)
    ax.annotate(
        text, xy=(date, value), xytext=offset, textcoords="offset points",
        fontsize=9, color=color, ha="center", va="center",
        arrowprops=dict(arrowstyle="-|>", color=color, linewidth=0.9,
                        mutation_scale=8, shrinkA=2, shrinkB=4),
    )


def save(fig, ax, filename, headroom=0.18):
    """Add space above the data (for the recession label), a little below, and save the PNG."""
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - (hi - lo) * 0.04, hi + (hi - lo) * headroom)
    fig.tight_layout(rect=(0, 0.03, 1, 1))           # leave room for the footnote
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / filename, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved charts/{filename}")


# ----------------------------------------------------------------------------
# Figure 1: Real GDP
# ----------------------------------------------------------------------------
def fig1_real_gdp():
    gdp = fetch("GDPC1")[START:END]
    peak_d, trough_d = pd.Timestamp("2007-10-01"), pd.Timestamp("2009-04-01")  # Q4 2007, Q2 2009
    peak_v, trough_v = gdp[peak_d], gdp[trough_d]
    change = (trough_v / peak_v - 1) * 100

    fig, ax = new_chart("Figure 1: Real GDP", "Billions of chained 2017 dollars", ["GDPC1"])
    ax.plot(gdp.index, gdp, color=NAVY, linewidth=2)
    annotate(ax, peak_d, peak_v, f"{quarter_label(peak_d)} peak\n${peak_v:,.0f}B", (-70, 22))
    annotate(ax, trough_d, trough_v, f"{quarter_label(trough_d)} trough\n${trough_v:,.0f}B", (62, -24))
    save(fig, ax, "fig1_real_gdp.png")

    record("1 Real GDP", "Q4 2007 peak ($B, chained 2017)", quarter_label(peak_d), f"{peak_v:,.1f}")
    record("1 Real GDP", "Q2 2009 trough ($B, chained 2017)", quarter_label(trough_d), f"{trough_v:,.1f}")
    record("1 Real GDP", "Peak-to-trough change", "", f"{change:+.1f}%")
    # Cross-check: the highest/lowest quarters actually in the data around the recession
    win = gdp["2007-07-01":"2009-12-31"]
    record("1 Real GDP", "Highest quarter, 2007Q3-2009Q4 (check)", quarter_label(win.idxmax()), f"{win.max():,.1f}")
    record("1 Real GDP", "Lowest quarter, 2007Q3-2009Q4 (check)", quarter_label(win.idxmin()), f"{win.min():,.1f}")


# ----------------------------------------------------------------------------
# Figure 2: Unemployment rate
# ----------------------------------------------------------------------------
def fig2_unemployment():
    un = fetch("UNRATE")[START:END]
    start_v = un[BASE_DATE]
    peak_d, peak_v = un.idxmax(), un.max()           # first month at the highest rate

    fig, ax = new_chart("Figure 2: Unemployment Rate", "Percent of labor force", ["UNRATE"])
    ax.plot(un.index, un, color=NAVY, linewidth=2)
    annotate(ax, BASE_DATE, start_v, f"{month_label(BASE_DATE)}: {start_v:.1f}%", (-62, 26))
    annotate(ax, peak_d, peak_v, f"{month_label(peak_d)} peak: {peak_v:.1f}%", (66, 20))
    save(fig, ax, "fig2_unemployment.png")

    record("2 Unemployment", "Rate at start of recession (%)", month_label(BASE_DATE), f"{start_v:.1f}")
    record("2 Unemployment", "Peak rate (%)", month_label(peak_d), f"{peak_v:.1f}")
    record("2 Unemployment", "Increase (percentage points)", "", f"{peak_v - start_v:+.1f}")


# ----------------------------------------------------------------------------
# Figure 3: CPI inflation, year-over-year
# ----------------------------------------------------------------------------
def fig3_inflation():
    # 12-month % change, computed on the full history before trimming the window
    headline = (fetch("CPIAUCSL").pct_change(12) * 100)[START:END]
    core = (fetch("CPILFESL").pct_change(12) * 100)[START:END]
    peak_d, peak_v = headline["2008"].idxmax(), headline["2008"].max()
    low_d, low_v = headline["2009"].idxmin(), headline["2009"].min()

    fig, ax = new_chart("Figure 3: CPI Inflation (Year-over-Year %)",
                        "Percent change from a year earlier", ["CPIAUCSL", "CPILFESL"])
    ax.axhline(0, color="#555555", linewidth=1)
    ax.plot(headline.index, headline, color=NAVY, linewidth=2, label="Headline CPI")
    ax.plot(core.index, core, color=RED, linewidth=2, label="Core CPI (excl. food & energy)")
    annotate(ax, peak_d, peak_v, f"{month_label(peak_d)} peak\n{peak_v:.1f}%", (-62, 6))
    annotate(ax, low_d, low_v, f"{month_label(low_d)} low\n{low_v:.1f}%", (66, 16))
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    save(fig, ax, "fig3_cpi_inflation.png")

    record("3 CPI inflation", "Headline YoY peak (%)", month_label(peak_d), f"{peak_v:.1f}")
    record("3 CPI inflation", "Headline YoY low (%)", month_label(low_d), f"{low_v:.1f}")
    record("3 CPI inflation", "Core YoY in the same month as headline peak (%)", month_label(peak_d), f"{core[peak_d]:.1f}")
    record("3 CPI inflation", "Core YoY in the same month as headline low (%)", month_label(low_d), f"{core[low_d]:.1f}")


# ----------------------------------------------------------------------------
# Figure 4: Capacity utilization
# ----------------------------------------------------------------------------
def fig4_capacity_utilization():
    tcu = fetch("TCU")[START:END]
    pre_v = tcu[BASE_DATE]
    low_d, low_v = tcu["2009"].idxmin(), tcu["2009"].min()

    fig, ax = new_chart("Figure 4: Capacity Utilization", "Percent of capacity", ["TCU"])
    ax.plot(tcu.index, tcu, color=NAVY, linewidth=2)
    annotate(ax, BASE_DATE, pre_v, f"{month_label(BASE_DATE)}: {pre_v:.1f}%", (-66, 22))
    annotate(ax, low_d, low_v, f"{month_label(low_d)} low: {low_v:.1f}%", (86, 8))
    save(fig, ax, "fig4_capacity_utilization.png")

    record("4 Capacity utilization", "Pre-recession level (%)", month_label(BASE_DATE), f"{pre_v:.1f}")
    record("4 Capacity utilization", "2009 low (%)", month_label(low_d), f"{low_v:.1f}")
    record("4 Capacity utilization", "Drop (percentage points)", "", f"{low_v - pre_v:+.1f}")


# ----------------------------------------------------------------------------
# Figure 5: Real personal income, with and without transfers
# ----------------------------------------------------------------------------
def fig5_real_personal_income():
    rpi, ex = fetch("RPI"), fetch("W875RX1")
    # Index both to 100 in Dec 2007 so they share one axis
    rpi = (rpi / rpi[BASE_DATE] * 100)[START:END]
    ex = (ex / ex[BASE_DATE] * 100)[START:END]
    after = slice(BASE_DATE, pd.Timestamp("2010-12-31"))   # search window for the troughs
    rpi_d, rpi_v = rpi[after].idxmin(), rpi[after].min()
    ex_d, ex_v = ex[after].idxmin(), ex[after].min()

    fig, ax = new_chart("Figure 5: Real Personal Income",
                        "Index (Dec 2007 = 100)", ["RPI", "W875RX1"])
    ax.plot(rpi.index, rpi, color=NAVY, linewidth=2, label="Real personal income (total)")
    ax.plot(ex.index, ex, color=RED, linewidth=2, label="Excluding government transfers")
    ax.fill_between(rpi.index, ex, rpi, where=rpi.index >= BASE_DATE, color=RED,
                    alpha=0.12, linewidth=0, label="Gap = government transfers")
    annotate(ax, ex_d, ex_v, f"Excl. transfers low\n{month_label(ex_d)}: {ex_v:.1f}", (84, -12))
    annotate(ax, rpi_d, rpi_v, f"Total low\n{month_label(rpi_d)}: {rpi_v:.1f}", (-6, 62))
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    save(fig, ax, "fig5_real_personal_income.png")

    record("5 Real personal income", "Total: low (Dec 2007 = 100)", month_label(rpi_d), f"{rpi_v:.1f}")
    record("5 Real personal income", "Total: change from Dec 2007", "", f"{rpi_v - 100:+.1f}%")
    record("5 Real personal income", "Excl. transfers: low (Dec 2007 = 100)", month_label(ex_d), f"{ex_v:.1f}")
    record("5 Real personal income", "Excl. transfers: change from Dec 2007", "", f"{ex_v - 100:+.1f}%")


# ----------------------------------------------------------------------------
# Figure 6: Case-Shiller national home price index
# ----------------------------------------------------------------------------
def fig6_home_prices():
    start, end = "2000-01-01", "2013-12-31"
    hpi = fetch("CSUSHPINSA")[start:end]
    peak_d, peak_v = hpi["2006"].idxmax(), hpi["2006"].max()
    low_d, low_v = hpi["2012"].idxmin(), hpi["2012"].min()
    decline = (low_v / peak_v - 1) * 100

    fig, ax = new_chart("Figure 6: Case-Shiller U.S. National Home Price Index",
                        "Index (Jan 2000 = 100)", ["CSUSHPINSA"], start, end)
    ax.plot(hpi.index, hpi, color=NAVY, linewidth=2)
    annotate(ax, peak_d, peak_v, f"{month_label(peak_d)} peak: {peak_v:.1f}", (-76, 14))
    annotate(ax, low_d, low_v, f"{month_label(low_d)} low: {low_v:.1f}\n({decline:.1f}% from peak)", (-6, -34))
    save(fig, ax, "fig6_home_prices.png")

    record("6 Home prices", "2006 peak (index)", month_label(peak_d), f"{peak_v:.1f}")
    record("6 Home prices", "2012 low (index)", month_label(low_d), f"{low_v:.1f}")
    record("6 Home prices", "Peak-to-low decline", "", f"{decline:+.1f}%")
    print(f"Home prices fell {abs(decline):.1f}% from {month_label(peak_d)} to {month_label(low_d)}.")


# ----------------------------------------------------------------------------
# Run everything
# ----------------------------------------------------------------------------
def main():
    fig1_real_gdp()
    fig2_unemployment()
    fig3_inflation()
    fig4_capacity_utilization()
    fig5_real_personal_income()
    fig6_home_prices()

    print("\nKEY NUMBERS FOR THE SLIDES")
    print(pd.DataFrame(key_numbers).to_string(index=False))


if __name__ == "__main__":
    main()
