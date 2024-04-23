# CHARISMA Calibration protocol

```
        # Create an instance of CalibrationModel
        calmodel = CalibrationModel(laser_wl=785)
        calmodel.derive_model_x(spe_neon,spe_neon_units="cm-1",ref_neon=None,ref_neon_units="nm",spe_sil=None,spe_sil_units="cm-1",ref_sil=None,ref_sil_units="cm-1")
        #store
        calmodel.save(modelfile)
        #load
        calmodel = CalibrationModel.from_file(modelfile)   
        #apply to new spectrum
        calmodel.apply_calibration_x(spe_to_calibrate,spe_units="cm-1")`
```        