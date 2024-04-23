# CHARISMA Calibration protocol

```python
# Create an instance of CalibrationModel
calmodel = CalibrationModel(laser_wl=785)
calmodel.derive_model_x(
    spe_neon,
    spe_neon_units='cm-1',
    ref_neon=None,
    ref_neon_units='nm',
    spe_sil=None,
    spe_sil_units='cm-1',
    ref_sil=None,
    ref_sil_units='cm-1'
    )
# Store
calmodel.save(modelfile)
# Load
calmodel = CalibrationModel.from_file(modelfile)
# Apply to new spectrum
calmodel.apply_calibration_x(
    spe_to_calibrate,
    spe_units='m-1'
    )
```
