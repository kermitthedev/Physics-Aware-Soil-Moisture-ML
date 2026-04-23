# Physics-Aware Soil Moisture Prediction: ANN vs LSTM Comparison Study

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
![Uncertainty Quantification](uncertainity_over_time.png)

*Six panel comparison showing: training loss curves, predicted vs actual scatter plots, RMSE bar chart, and time series overlay for both models.*

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
