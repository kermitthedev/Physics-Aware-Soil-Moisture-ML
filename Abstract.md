# Abstract — Draft v1.0

**Paper Title:**
Non-Causal Temporal Sampling Inflates Sequence Model 
Advantage by 3× in Soil Moisture Prediction: 
A Methodological Analysis

**Target Venue:** 
IEEE Geoscience and Remote Sensing Letters (IEEE GRSL)
or IGARSS 2027

**Authors:**
Adi Singh — MS Cybersecurity, Mississippi State University

**Status:** Draft — pending experimental completion
- ✅ Phase 1-5 complete
- ⬜ Issue #5 Downsampling regime study
- ⬜ Issue #6 Temporal feature ablation  
- ⬜ Issue #7 CAF field scale validation
- ⬜ Issue #9 Statistical significance testing

**Last Updated:** April 2026

---
---

## Abstract (Draft)

Deep learning model selection for environmental time 
series prediction is critically dependent on evaluation 
methodology. This study demonstrates that random 
train/test splits — the dominant evaluation strategy 
in soil moisture deep learning — introduce non-causal 
temporal sampling that fundamentally distorts 
architecture comparison results.

Using a multi-depth soil moisture sensor dataset 
augmented with 30 days of physics-informed synthetic 
data (47,609 total samples), we compare Artificial 
Neural Networks (ANN) and Long Short-Term Memory 
networks (LSTM) under both random and chronological 
evaluation regimes across a sequence length sweep 
(SEQ_LEN = 10, 30, 60, 120 minutes).

Under random splits, LSTM performance improvements 
increase from ~6% to ~17%, suggesting that commonly 
used evaluation strategies substantially overestimate 
the benefits of temporal modeling. Under chronological 
evaluation — the correct paradigm for operational 
forecasting — LSTM advantage follows a physically 
meaningful inverted-U pattern peaking at SEQ_LEN=60 
minutes (6.1% improvement), corresponding to the 
duration of a complete irrigation event cycle, and 
collapsing at SEQ_LEN=120 minutes (0.3% improvement).

SHAP explainability analysis reveals that random 
splits cause models to rely on temporal features 
(Hour ranked #1) while chronological evaluation 
forces reliance on physical sensor relationships 
(Layer 4 ranked #1) — confirming that leakage 
produces spurious temporal dependencies rather than 
genuine physical learning.

Additionally, a 1D Richards Equation finite difference 
solver revealed preferential flow dynamics in the 
sensor system, providing a physical explanation for 
the bounded temporal dependency observed under 
honest evaluation.

These results demonstrate that evaluation methodology 
— not model architecture — determines the apparent 
magnitude and persistence of temporal learning in 
soil moisture prediction, with direct implications 
for remote sensing ML evaluation protocols including 
NASA CYGNSS soil moisture retrieval.
