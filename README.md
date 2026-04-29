# From Capital War to Fragmented Civilian Insecurity

**Course:** POLI3148 Data Science in Politics and Public Administration  
**Assignment:** Data-Driven Analysis of Armed Conflict Using ACLED  
**Author:** Kenny Chow

**Github Repo:** https://github.com/Kenny-Choww/HKU-POLI3148-Assignment1
**Online Report URL:** https://kenny-choww.github.io/HKU-POLI3148-Assignment1/
**Report:** [docs/index.html](docs/index.html)

## Project Overview

This project analyzes how Sudan's civil war has shifted geographically and politically since the outbreak of fighting on 15 April 2023. The main research question is:

> How has Sudan's civil war shifted geographically and politically since April 2023, and can ACLED event patterns help identify where violence against civilians is most likely to intensify?

The project uses ACLED event data as the primary dataset and Python as the primary programming language. It builds from cleaning and exploratory analysis to spatial visualization, actor-network analysis, and an explainable machine-learning risk model. The final output is an interactive analytical HTML report with a linked dashboard layer: shared controls update the time series, state ranking, event-type mix, actor ranking, and narrative insight together.

## Data Sources

The primary dataset is `data/raw/acled_sudan_2023_2025.csv`, downloaded from the [ACLED Data Export Tool](https://acleddata.com/conflict-data/data-export-tool). The export covers Sudan from **2023-04-15 to 2025-04-22** and includes all ACLED event types in the file.

The project also uses Sudan admin1 boundary geometry from [geoBoundaries](https://www.geoboundaries.org/) through the gbOpen ADM1 API. These boundaries are used only for state-level maps; ACLED remains the primary analytical data source.

Important scope note: the proposed topic referenced 2023-2026, but the available raw ACLED export ends on 2025-04-22. The report states this limitation clearly. To extend the project to 2026, replace the raw CSV with a newer ACLED export and rerun the notebooks or `code/Z_generate_report.py`.

## Methodology

The workflow has four layers:

1. **Data loading and cleaning:** standardizes ACLED columns, validates country/date coverage, converts dates and numeric fields, creates time variables, flags civilian targeting, and builds an admin1-month panel.
2. **Descriptive and spatial analysis:** examines monthly event and fatality trends, maps event-level geography, and aggregates conflict intensity to Sudanese states.
3. **Actor analysis:** creates an actor co-involvement network from ACLED `actor1` and `actor2` fields to show the structure around SAF, RSF, civilians, militias, and local armed groups.
4. **Machine learning:** trains a transparent logistic classifier to predict whether an admin1-month will experience high civilian-targeting risk in the following month using lagged conflict features.

## Key Findings

- The war begins as a capital-centered conflict, with Khartoum recording the largest overall event volume.
- Reported lethality is more geographically fragmented than event volume; Darfur, especially North and West Darfur, carries a major fatality burden.
- Violence against civilians is not only a battlefield byproduct. Al Jazirah, Khartoum, North Darfur, and other states show substantial civilian-targeting patterns.
- The actor network is centered on SAF and RSF but includes civilians, unidentified armed groups, communal militias, joint forces, police, and local armed movements.
- The risk model suggests that lagged civilian targeting, recent conflict intensity, actor diversity, and location spread are useful warning indicators for next-month civilian-targeting risk.

## Project Structure

```text
asm1/
|-- README.md
|-- note_on_ai_use.md
|-- requirements.txt
|-- POLI3148_Assignment1_Instruction.pdf
|-- code/
|   |-- 01_data_loading_cleaning.ipynb
|   |-- 02_exploratory_spatial_actor_analysis.ipynb
|   |-- 03_machine_learning_and_report.ipynb
|   |-- project_utils.py
|   `-- Z_generate_report.py
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- external/
`-- docs/
    `-- index.html
```

## How to Run

Install the Python dependencies:

```powershell
pip install -r requirements.txt
```

Run the notebooks in order:

```text
code/01_data_loading_cleaning.ipynb
code/02_exploratory_spatial_actor_analysis.ipynb
code/03_machine_learning_and_report.ipynb
```

Or regenerate all summary outputs and the HTML report directly:

```powershell
python code\Z_generate_report.py
```

The final interactive report is saved at `docs/index.html`.

## Limitations

ACLED records reported events, so the data reflect source availability, reporting access, and coding decisions. Fatalities are reported estimates and should be interpreted cautiously. The machine-learning model is designed for interpretable early warning, not causal inference or operational prediction. Finally, the current raw data file ends in April 2025, so the project should be refreshed with a newer ACLED export before making claims about 2026.
