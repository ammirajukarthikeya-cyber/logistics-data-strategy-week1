# Strategic Planning and Data Exploration in Logistics
**An End-to-End Analytical Framework for Multi-Echelon Supply Chain Optimization, Predictive Lead-Time Analytics, and Route Efficiency**

---

## 📌 Project Overview
This repository contains the strategic planning report, data architecture design, and Python analytical pipelines for **Week 1: Strategic Planning and Data Exploration in Logistics**. 

The project addresses core operational bottlenecks in modern enterprise logistics—including lead-time variance, bullwhip effects, SLA breach risks, and vehicle routing inefficiencies—by applying statistical profiling, supervised machine learning, unsupervised geospatial clustering, and combinatorial route heuristics.

---

## 🎯 Key Performance Indicators (KPIs)
| KPI Metric | Formulation | Operational Benchmark |
| :--- | :--- | :--- |
| **On-Time In-Full (OTIF) Rate** | `(N_compliant / N_total) * 100` | $\ge 98.5\%$ across all tiers |
| **Order Lead Time & Delay Variance** | $T_{\text{actual}} - T_{\text{scheduled}}$ | $\le 24\text{h}$ urban; $\text{Var} \le 1.5\text{h}$ |
| **Inventory Turnover Ratio (ITR)** | $\frac{\text{COGS}}{\text{Avg. Inventory Value}}$ | $\ge 8.0\times$ annual velocity |
| **Cost Per Delivery / Ton-Mile** | $\frac{\text{Fuel + Labor + Maint}}{\text{Total Distance}}$ | $\le \$1.85 / \text{delivery stop}$ |
| **Fleet Capacity Utilization** | $\frac{\text{Actual Volume}}{\text{Max Capacity}} \times 100$ | $\ge 88\%$ fill rate |

---

## 🏗️ Analytical Architecture & Modules

```
├── Strategic_Planning_and_Data_Exploration_in_Logistics.docx   <- Comprehensive Word (.docx) Strategic Report
├── logistics_strategy_pipeline.py                             <- Standalone executable Python pipeline
├── generate_docx_report.py                                    <- Automated Word document builder
└── README.md                                                  <- Project documentation & instructions
```

### Pipeline Workflows:
1. **Module 1 (Data Ingestion & Hygiene):** Ingests transactional logs, handles anomalous values, standardizes datetime formats, and computes operational delta features.
2. **Module 2 (KPI Performance Engine):** Evaluates multi-tier supply chain metrics (OTIF, delay variance, SLA breach rates) segmented by fulfillment tier.
3. **Module 3 (Predictive Delay Classifier):** Random Forest Classifier predicting delivery delay risks based on distance, payload weight, traffic congestion indices, and shipping tier.
4. **Module 4 (Spatial Fulfillment Clustering):** K-Means spatial clustering algorithm grouping delivery coordinates to determine optimal micro-fulfillment hub centroids.
5. **Module 5 (Heuristic Route Optimization):** Nearest-Neighbor Traveling Salesperson Problem (TSP) solver minimizing cumulative fleet transit distance and carbon emissions.

---

## 🚀 Getting Started & Execution

### Prerequisites
- Python 3.9+
- Required libraries:
  ```bash
  pip install numpy pandas scikit-learn scipy python-docx
  ```

### Running the End-to-End Pipeline
```bash
python logistics_strategy_pipeline.py
```

### Generating the Formatted Word (.docx) Report
```bash
python generate_docx_report.py
```

---

## 📈 Expected Business Outcomes
- **18–22% Reduction in Transit Delays** through proactive ETA alerting and dynamic dispatching.
- **12–15% Drop in Inventory Carrying Costs** via synchronized multi-echelon demand forecasting.
- **14% Mileage & Emissions Reduction** through algorithmic vehicle route sequencing and micro-hub zoning.
