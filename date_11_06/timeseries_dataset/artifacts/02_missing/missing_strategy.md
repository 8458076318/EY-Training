# Missing Value Strategy

No missing values were detected in the supplied dataset. If future rows contain gaps, use linear interpolation for dense sensor signals (`red`, `ir`, `red_corrected`, `ir_corrected`) and preserve `seq` / `timestamp_ms` after reindexing on the expected sampling interval.
