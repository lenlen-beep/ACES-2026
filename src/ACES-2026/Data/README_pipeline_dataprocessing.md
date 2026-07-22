#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 15:58:55 2026

@author: nele
"""

# Data Processing Pipeline — ACES 2026
## District Heating Network Optimization | Europa-Universität Flensburg

---
## 1. Heat Demand Data (Aalborg 2019)
## 1.1 Software Environment
- Python version: 3.11.5
- Full package list: see requirements.txt

---

## 1.2 Random Seed
- Stratified sampling uses fixed seed: 42
- Set as RANDOM_SEED = 42 in step2_267_profiles_stratified.py

---

## 1.3 Script Order & Purpose

| Step | Script                                     | What it does                                                                                                     |
|------|--------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| 1    | step1_full_heat_demand.py                  | Loads raw Aalborg CSVs, runs coverage and quality checks, produces visual presentation of the data as an output  |
| 2    | step2_267_profiles_stratified.py           | Produces selected_267_profiles_*.csv and visual presentation of the reduced results as an output                 |
| 3    | compare_aalborg_flensburg_temperature.py   | Compares both temperatures for evidence of comparability of the heat demand vs weather data as model input       |
---

## 1.4 Input / Output Files

Input:
- 25 raw Aalborg CSV files (anonymised smart heat meter data in hourly resolution from 2018-2020)
- contextual_data.csv (building type, construction year & energy label per meter_id)

- temperature data from Aalborg and Flensburg

Output:
- selected_267_profiles_2019_long.csv (full results including hourly load, annual load, housing type with meter ID)
- selected_267_profiles_2019_wide.csv (hourly load in kWh per house (1-267) only organized in the correct structure for optimization model)
- selected_267_profiles_meta.csv      (background information data on meter ID, housing type, load & demand, quantile bins, coverage fraction)
- figures with all validation plots
- validation/statistical checks/interim results in the console after running scripts (step1: hourly coverage, share of building types, step2: amount of excluded data, allocation per building type, validation of sample vs. whole population, quantile comparison, KS-test, p-value)

- figures of temperature comparisons and therefor validation

---
## 2. Weather Data (for PV Generation)

---

## 3. Electricity Prices (for HP/Compressor)
- wholesale spot-market prices in hourly resolution in MWh/year from 2024
- weekly rythm was aligned to 2019 by shifting the series so that weekdays match

---

## 4. Gas Price (for Gas Boiler)
- set as a constant/fixed price because gas boiler only serves as a peak shaving/backup

---

## 5. Notes / Changes
- description above based on final presentation status (until 17.07.2026)
- [Date]: [Any change you made, e.g. "switched PV data source from X to Y"]