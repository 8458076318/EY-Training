# PPG EDA Run Report

## Input

- CSV: `C:\Training\AI-ML-Training-Projects\date_11_06\data\raw\sakshi_ppg_20260611T074737_len148s.csv`
- Rows: `7423`
- Columns: `102`
- Sampling interval: `20.000 ms`
- Sampling frequency: `50.000 Hz`
- Dominant period: `2474` samples

## Time Continuity

- Missing timestamp groups: `560`
- Estimated missing rows from gaps: `664`

## Missing Values

- Total missing cells: `0`
- Strategy: linear interpolation for future gaps in dense sensor channels, after reindexing to the expected cadence

## Stationarity

- Test: `ADF-like`
- Statistic: `-3.141746`
- P-value: `0.001686`
- Stationary: `True`

## Outliers

- `red`: `276`
- `ir`: `993`
- `red_corrected`: `49`
- `ir_corrected`: `664`

## Artifacts

- time_series_plot: `C:\Training\AI-ML-Training-Projects\date_11_06\artifacts\03_series_plots\time_series.png`
- superimposed_red_ir: `C:\Training\AI-ML-Training-Projects\date_11_06\artifacts\03_series_plots\red_ir_superimposed.png`
- normalized_red_ir_peaks: `C:\Training\AI-ML-Training-Projects\date_11_06\artifacts\03_series_plots\red_ir_normalized_peaks.png`
- decomposition_plot: `C:\Training\AI-ML-Training-Projects\date_11_06\artifacts\04_decomposition\ir_corrected_decomposition.png`
- weekly_boxplot: `C:\Training\AI-ML-Training-Projects\date_11_06\artifacts\05_boxplots\weekly_boxplot.png`
- monthly_boxplot: `C:\Training\AI-ML-Training-Projects\date_11_06\artifacts\05_boxplots\monthly_boxplot.png`
- stationarity_plot: `C:\Training\AI-ML-Training-Projects\date_11_06\artifacts\06_stationarity\rolling_stationarity.png`
- acf_pacf_plot: `C:\Training\AI-ML-Training-Projects\date_11_06\artifacts\07_acf_pacf\acf_pacf.png`
- outlier_red_plot: `C:\Training\AI-ML-Training-Projects\date_11_06\artifacts\04_decomposition\red_corrected_outliers.png`
- outlier_ir_plot: `C:\Training\AI-ML-Training-Projects\date_11_06\artifacts\04_decomposition\ir_corrected_outliers.png`
