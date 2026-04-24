# Physics-Aware Soil Moisture Prediction: ANN vs LSTM with Monte Carlo Uncertainty Quantification and SHAP Explainability

*Inspired by Boyd et al. (2019) — "High Spatio-Temporal Resolution CYGNSS Soil Moisture Estimates Using Artificial Neural Networks"*

---

## Overview

This project implements and compares two deep learning architectures — a fully connected Artificial Neural Network (ANN) and a Long Short-Term Memory network (LSTM) — for soil moisture prediction from multi-sensor time series data.

The methodology is directly inspired by Boyd et al. (2019), which demonstrated that physics-aware machine learning can effectively retrieve soil moisture from NASA's CYGNSS satellite GPS reflectometry signals. This project replicates the core ML pipeline concept using ground-based sensor data: using shallow/surface sensor readings as observable inputs to predict deeper, harder-to-measure moisture values — mirroring the satellite approach of using GPS surface reflections to estimate subsurface soil moisture.

---

## Research Question

> Does temporal memory (LSTM) improve soil moisture prediction over a static feedforward network (ANN) when using multi-depth sensor observations?

---

## Dataset

- **Source:** Multi-sensor plant vase soil moisture dataset
- **Duration:** 4 days (March 6-9, 2020), minute-level resolution
- **Samples:** 4,409 timestamped readings
- **Features:** 5 moisture sensors at different depths + temporal context (hour, minute)
- **Target:** moisture4 (deepest sensor — hardest to measure directly)

| Column | Description |
|--------|-------------|
| moisture0 | Shallowest sensor (surface observable) |
| moisture1 | Second layer sensor |
| moisture2 | Third layer sensor |
| moisture3 | Fourth layer sensor |
| moisture4 | Deepest sensor **(prediction target)** |
| hour, minute | Temporal context features |
| irrigation | Irrigation event flag |

---

## Methodology

### Feature Design — Physics-Aware Approach
Following Boyd et al. (2019), features were selected based on physical understanding of the system rather than purely statistical methods. Shallow sensor readings serve as surface observables analogous to CYGNSS GPS reflectometry signals, while the deep sensor target mirrors subsurface soil moisture retrieval.

### ANN Architecture
- Parameters: 2,561
- Approach: Treats each timestep independently
- Loss: MSE | Optimizer: Adam (lr=0.001)

### LSTM Architecture
- Sequence length: 10 timesteps
- Approach: Leverages temporal memory across previous readings
- Loss: MSE | Optimizer: Adam (lr=0.001)

---

## Results

| Model | RMSE (cm³/cm³) | R² |
|-------|---------------|-----|
| **ANN** | **0.0038** | **0.8434** |
| LSTM | 0.0056 | 0.6041 |

**Winner: ANN — outperforms LSTM by 47.2%**

### Key Finding

Contrary to the hypothesis that temporal memory would improve predictions, the ANN significantly outperformed the LSTM. This suggests that at minute-level sampling frequency, the cross-sensor spatial relationships between depth layers carry more predictive signal than temporal history. The problem structure — predicting moisture at depth from simultaneous multi-channel surface observations — is more analogous to Boyd's CYGNSS setup where simultaneous multi-channel observations drive prediction rather than sequential patterns.

This finding aligns with Boyd et al. (2019)'s choice of a fully connected ANN over sequential architectures for GNSS reflectometry based soil moisture retrieval.

---

## Visualizations

![ANN vs LSTM Comparison](ann_vs_lstm_comparison.png)
![Remote Sensing Style Analysis](remote_sensing_analysis.png)
![Temporal Soil Moisture Snapshots](temporal_snapshots.png)
![Inter-Sensor Correlation Analysis](sensor_correlation_analysis.png)

*Six panel comparison showing: training loss curves, predicted vs actual scatter plots, RMSE bar chart, and time series overlay for both models.*

---
## Uncertainty Quantification — Monte Carlo Dropout

Standard neural networks produce single-point predictions with no measure of 
confidence. In remote sensing applications, uncertainty estimates are critical 
— a soil moisture reading without confidence bounds is scientifically incomplete.

This study extends the ANN with Monte Carlo Dropout uncertainty quantification,
running 100 stochastic forward passes to generate a full predictive distribution
rather than a single estimate.

### Method
- Monte Carlo Dropout: dropout remains active during inference
- 100 forward passes per prediction
- 95% confidence intervals computed from sample statistics
- Calibration evaluated against perfect calibration diagonal

### Results

| Model | RMSE (cm³/cm³) | R² | Notes |
|-------|---------------|-----|-------|
| ANN (Standard) | 0.0038 | 0.8434 | Single point prediction |
| LSTM | 0.0056 | 0.6041 | Sequential memory |
| **ANN (MC Dropout)** | **0.0037** | **best** | **+ uncertainty estimates** |

Mean uncertainty ±2σ: 0.0031 cm³/cm³

### Key Physical Finding

Model uncertainty peaks precisely during irrigation events — the moments of 
greatest physical change in the soil system. This demonstrates physically 
meaningful uncertainty behavior: the model correctly identifies when it is 
operating outside its comfort zone, analogous to how satellite retrieval 
algorithms flag low-confidence retrievals during rainfall events in CYGNSS 
data products.

This behavior directly mirrors the uncertainty quantification challenges 
discussed in Boyd et al. (2019) for GNSS reflectometry soil moisture retrieval.

### Uncertainty Visualization

![Uncertainty Quantification](uncertainity_over_time.png)

---
## SHAP Explainability Analysis

Deep learning models are often criticized as black boxes — producing 
predictions without physical justification. This study applies SHAP 
(SHapley Additive exPlanations) to open the black box and validate 
that the model learned physically meaningful relationships rather than 
spurious correlations.

### Global Feature Importance

| Rank | Feature | Mean |SHAP| | Physical Interpretation |
|------|---------|-------------|------------------------|
| 1 | Hour of Day | 0.01985 | Irrigation timing context |
| 2 | Surface (m0) | 0.01940 | Water infiltration entry point |
| 3 | Layer 4 (m3) | 0.01823 | Adjacent layer to target |
| 4 | Layer 2 (m1) | 0.01202 | Intermediate transport layer |
| 5 | Layer 3 (m2) | 0.00554 | Near-constant layer, correctly ignored |
| 6 | Minute | 0.00151 | Noise, correctly ignored |

### Physics Validation

Three key findings confirm physically meaningful model behavior:

**1. Surface sensor ranked #2** — confirms model learned water 
infiltration pathway. Water enters from the surface downward, 
consistent with known soil physics.

**2. Layer 4 ranked #3** — the sensor directly adjacent to the 
prediction target is the third most important feature. Physically 
correct — nearest neighbor carries most signal.

**3. Minute ranked last** — sub-minute temporal variation is noise. 
Model correctly learned to ignore this, relying on physical 
sensor readings instead.

**4. Hour ranked #1 with 245.9% importance spike during irrigation** 
— rather than a spurious time-of-day correlation, SHAP reveals the 
model uses hour specifically to contextualize irrigation events, 
which occur at predictable times. This represents intelligent 
temporal-physical feature interaction.

### Feature Importance Shift During Irrigation

| Feature | Shift During Irrigation | Physical Meaning |
|---------|------------------------|-----------------|
| Surface (m0) | ↑ 129.3% | Surface entry point activates |
| Layer 4 (m3) | ↑ 80.6% | Deep layer response detected |
| Hour | ↑ 245.9% | Irrigation timing context critical |
| Layer 2 (m1) | ↓ 29.7% | Intermediate layers less critical |
| Layer 3 (m2) | ↓ 10.5% | Near-constant layer ignored |
| Minute | ↑ 100.8% | Fine temporal resolution activated |

The feature importance shift during irrigation events reveals the 
model's implicit understanding of water infiltration physics — 
increased reliance on surface entry point and adjacent deep layer 
during active water movement mirrors the physical process of 
downward water percolation through soil layers.

This directly validates the physics-aware methodology of Boyd et al. 
(2019), where physical understanding guides both feature selection 
and model interpretation.

![SHAP Analysis](shap_explainability.png)

---




## Repository Structure
---

## How to Run

1. Open `soil_moisture_ann_lstm.ipynb` in Google Colab
2. Upload `plant_vase1(2).csv` when prompted
3. Run all cells sequentially
4. Results and visualizations generate automatically

---

## References

Boyd, D. R., Senyurek, V., Lei, F., Gurbuz, A. C., Kurum, M., & Moorhead, R. (2019). High Spatio-Temporal Resolution CYGNSS Soil Moisture Estimates Using Artificial Neural Networks. *Remote Sensing*, 11(19), 2272. https://doi.org/10.3390/rs11192272

---

## Author

**Adi Singh**
MS in Cybersecurity Operations and Defense — Mississippi State University
GitHub: [@kermitthedev](https://github.com/kermitthedev)

---

*This project was developed as a learning exercise to understand physics-aware machine learning methodology for geophysical remote sensing applications.*
