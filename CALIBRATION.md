# CHARISMA Calibration protocol

```python
# Create an instance of CalibrationModel
# x-calibration
calmodel = CalibrationModel.calibration_model_factory(laser_wl=785,spe_neon,spe_sil,neon_wl = rc2const.NEON_WL,
                                            find_kw={"wlen" : 100, "width" :  1}, fit_peaks_kw={},should_fit=True)
# Save
calmodel.save(modelfile)
# Load
calmodel = CalibrationModel.from_file(modelfile)
# Apply to new spectrum
calmodel.apply_calibration_x(
    spe_to_calibrate,
    spe_units='cm-1'
    )
```
