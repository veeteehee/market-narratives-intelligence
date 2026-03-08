"""
data_generator.py
Generates a realistic synthetic CSV matching the all-data.csv format.
Run standalone: python data_generator.py
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

NARRATIVES = {
    "inflation": [
        "Inflation continues to rise as consumer prices hit a 40-year high.",
        "Central banks struggle to contain inflationary pressures across global markets.",
        "Core CPI exceeds analyst expectations amid persistent supply chain disruptions.",
        "Fed signals aggressive rate hikes to tame runaway inflation expectations.",
        "Wage growth failing to keep pace with soaring headline inflation rates.",
        "Energy costs drive inflation to multi-decade highs across European markets.",
        "Investors brace for prolonged inflation cycle despite consecutive rate increases.",
        "Housing inflation remains sticky even as broader CPI index moderates slightly.",
        "Food price inflation hits hardest among lower-income household segments.",
        "Import price index surges as currency weakness amplifies domestic inflation.",
    ],
    "tech_layoffs": [
        "Major tech companies announce mass layoffs amid broader economic uncertainty.",
        "Silicon Valley workforce shrinks as the post-pandemic growth era ends.",
        "Meta cuts eleven thousand jobs in largest single layoff in company history.",
        "Amazon reduces headcount by eighteen thousand as cloud revenue growth slows.",
        "Tech sector unemployment spikes sharply as startup funding rounds dry up.",
        "Former engineers from big tech flood the job market after sweeping cuts.",
        "AI-driven automation cited as key reason for accelerating technology layoffs.",
        "Recruitment freezes spread across large tech firms as operating margins compress.",
        "Severance packages under scrutiny as laid-off workers organise collectively.",
        "Junior engineers hit hardest as firms restructure towards senior specialist roles.",
    ],
    "ai_boom": [
        "Artificial intelligence investments reach record highs in the third quarter.",
        "Generative AI startups attracting billions in new venture capital funding rounds.",
        "ChatGPT launch sparks renewed investor interest in AI infrastructure stocks.",
        "Nvidia shares surge on unprecedented demand for AI accelerator chips.",
        "Enterprise AI adoption accelerates rapidly across the financial services sector.",
        "AI productivity tools are reshaping workforce dynamics at major technology companies.",
        "Machine learning cloud platforms see exponential growth in monthly active usage.",
        "OpenAI valuation soars as the AI arms race intensifies among technology giants.",
        "Foundation model licensing becomes new revenue stream for AI research labs.",
        "Sovereign wealth funds allocate strategic positions to artificial intelligence sector.",
    ],
    "energy_crisis": [
        "European energy crisis deepens as winter demand surges beyond forecasts.",
        "Natural gas spot prices hit record highs following upstream supply disruptions.",
        "OPEC plus cuts production targets as oil markets face uncertain demand outlook.",
        "Renewable energy investment accelerates sharply amid fossil fuel price volatility.",
        "Grid stability concerns mount as extreme weather events strain power infrastructure.",
        "Energy companies report record quarterly profits while consumers face high bills.",
        "LNG export capacity expands as Europe urgently seeks alternative fuel supply.",
        "Nuclear energy revival gains political momentum amid energy security debates.",
        "Electricity rationing discussed in several European countries facing supply gaps.",
        "Carbon credit prices spike as energy firms seek to offset emissions increases.",
    ],
    "banking_stress": [
        "Regional banks face significant deposit outflows following high-profile bank collapse.",
        "Emergency rescue of major Swiss bank shakes global banking sector confidence.",
        "Federal Reserve emergency lending facilities see unprecedented drawdown demand.",
        "Bank stocks plunge sharply as contagion fears spread through financial sector.",
        "Deposit insurance coverage limits debated as bank run concerns threaten stability.",
        "Commercial real estate loan exposure raises alarm bells at mid-size lenders.",
        "Regulators tighten capital adequacy requirements following stress testing events.",
        "Unrealised bond portfolio losses mount at community banks amid rate increases.",
        "Credit default swap spreads on bank debt widen to post-financial-crisis highs.",
        "Interbank lending rates spike as counterparty risk appetite deteriorates sharply.",
    ],
}

SECTORS  = ["Technology","Finance","Energy","Healthcare","Consumer","Industrials"]
SOURCES  = ["Reuters","Bloomberg","Financial Times","Wall Street Journal",
            "CNBC","MarketWatch","Seeking Alpha","Twitter/X"]


def make_demo_csv(output_path: str, n_rows: int = 400, seed: int = 42) -> pd.DataFrame:
    """Generate a demo CSV with text, date, sentiment, sector, return columns."""
    random.seed(seed)
    np.random.seed(seed)

    start_date = datetime(2023, 1, 1)
    nar_keys   = list(NARRATIVES.keys())
    weights    = [25, 20, 30, 15, 10]
    rows       = []

    for _ in range(n_rows):
        nar_key   = random.choices(nar_keys, weights=weights)[0]
        base_text = random.choice(NARRATIVES[nar_key])
        modifiers = ["", " Analysts warn.", " Markets react swiftly.",
                     " Experts remain divided.", " Officials respond."]
        text      = base_text + random.choice(modifiers)

        date      = (start_date + timedelta(days=random.randint(0, 179))).strftime("%Y-%m-%d")
        sector    = random.choice(SECTORS)
        ret_val   = round(np.random.normal(0, 0.012), 5)

        # Assign approximate sentiment based on narrative
        pos_nars = {"ai_boom"}
        neg_nars = {"tech_layoffs", "banking_stress", "energy_crisis"}
        if nar_key in pos_nars:
            sent = random.choice(["positive","neutral"])
        elif nar_key in neg_nars:
            sent = random.choice(["negative","neutral"])
        else:
            sent = random.choice(["positive","negative","neutral"])

        rows.append({"text": text, "date": date, "sentiment": sent,
                     "sector": sector, "return": ret_val,
                     "source": random.choice(SOURCES)})

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Demo CSV → {output_path}  ({len(df)} rows)")
    return df


if __name__ == "__main__":
    make_demo_csv("demo_narratives.csv")
