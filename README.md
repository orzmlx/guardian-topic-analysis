# Event-Driven Topic Shifts in The Guardian: A Comparative Analysis of BERTopic and LDA

This project analyzes how *The Guardian*'s news agenda shifted around Queen Elizabeth II's passing (September 8, 2022). By using topic modelling (LDA vs. BERTopic) on articles from 2020 to 2025, we investigate the "agenda reset" phenomenon, excluding direct event coverage to capture indirect thematic shifts.

## 📂 Project Structure

The repository is organized as follows:

```
├── src/                # Source code
│   ├── guardian_news_scraper.py      # Script to fetch data from Guardian API
│   ├── guardian_lord_analysis_change.ipynb  # Main analysis notebook (Preprocessing, Modeling, Visualization)
│   └── sample.py       # Helper scripts
├── data/               # Data files
│   ├── guardian_news_std.csv         # (Not included in repo due to size) Full dataset
│   └── guardian_news_std_sample.csv  # Sample dataset for reproducibility
├── report_latex/       # LaTeX source for the final report
│   ├── project-acl.tex # Main LaTeX file
│   └── project-acl.pdf # Compiled PDF report
├── images/             # Generated figures and wordclouds
│   ├── topic_time_series.png
│   ├── lda_combined_time_series.png
│   └── wordclouds/     # Topic wordclouds for LDA and BERTopic
├── results_output/     # Intermediate analysis results (CSV)
└── course_materials/   # Course-related documents and examples
```

## 🚀 Getting Started

### Prerequisites

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 1. Data Collection

To fetch the news data yourself (requires a Guardian Open Platform API key), run:

```bash
python src/guardian_news_scraper.py
```

*Note: The project already includes a sample dataset in `data/guardian_news_std_sample.csv`.*

### 2. Analysis & Modeling

The core analysis is performed in the Jupyter Notebook:

```bash
jupyter notebook src/guardian_lord_analysis_change.ipynb
```

This notebook covers:
- **Preprocessing**: Cleaning text, removing stopwords, and filtering event-related keywords.
- **Modeling**: Training LDA and BERTopic models on Pre-event and Post-event datasets.
- **Alignment**: Semantically aligning topics between time periods.
- **Visualization**: Generating time series plots and word clouds (saved to `images/`).

## 📊 Results

The full analysis and findings are detailed in the **[Final Report](report_latex/project-acl.pdf)**.

Key findings include:
- A significant "agenda reset" was observed after the Queen's passing.
- **Economic Policy** and **Political Leadership** topics consolidated and increased in prominence post-event.
- **BERTopic** provided superior granularity and coherence compared to LDA in capturing these shifts.

## 📝 License

This project is part of the Text Mining course at Linköping University.
