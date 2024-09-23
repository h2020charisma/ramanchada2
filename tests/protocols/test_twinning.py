import ramanchada2 as rc2
from ramanchada2.auxiliary.spectra.datasets2 import (filtered_df,get_filenames,
                                                     prepend_prefix)
import pytest
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.linear_model import LinearRegression

def load_spectrum(row):
    return rc2.spectrum.Spectrum.from_local_file(row["filename"])
   

@pytest.fixture(scope="module")
def reference_with_replicates():
    df_ref = filtered_df(sample=['TiO2'],device=['BWtek'],provider=['FNMT-B'],laser_wl=['785'])    
    df_ref["filename"] = prepend_prefix(df_ref["filename"])
    df_ref["device_id"] = prepend_prefix(df_ref["filename"])
    df_ref['device_id'] = df_ref[['provider', 'device', 'laser_wl']].agg(' '.join, axis=1)
    checkmetadata4twinning(df_ref,25)
    df_ref["spectrum_original"] = df_ref.apply(load_spectrum, axis=1)
    return df_ref

@pytest.fixture(scope="module")
def twinned_with_replicates():
    df_test = filtered_df(sample=['TiO2'],device=['BWtek'],provider=['ICV'],laser_wl=['785'])    
    df_test["filename"] = prepend_prefix(df_test["filename"])
    df_test['device_id'] = df_test[['provider', 'device', 'laser_wl']].agg(' '.join, axis=1)
    checkmetadata4twinning(df_test,25)
    df_test["spectrum_original"] = df_test.apply(load_spectrum, axis=1)
    return df_test

def checkmetadata4twinning(df,expected_rows=25):
    assert "laser_power_mW" in df.columns
    assert "laser_power_percent" in df.columns
    assert "time_ms" in df.columns
    assert "replicate" in df.columns
    assert expected_rows == df.shape[0]

def process_replicates(df,grouping_cols = ['sample','provider','laser_wl','laser_power_percent','laser_power_mW','time_ms'],title="reference"):
    processed_rows = []
    
    for group_keys, group in df.groupby(grouping_cols):
        # Iterate over each row in the group
        x = None 
        y = None
        for index, row in group.iterrows():
            if x is None:
                x = row["spectrum_original"].x
            if y is None:            
                y = row["spectrum_original"].y
            else:
                y = y + row["spectrum_original"].y
        spe_average = rc2.spectrum.Spectrum(x,y/ group.shape[0])
        
        processed_row = row.copy()  # Make a copy of the row
        processed_row["spectrum"] = spe_average
        processed_row["type"] = title
        processed_rows.append(processed_row)
    
    df =  pd.DataFrame(processed_rows)
    df.sort_values(by='laser_power_percent')
    return df

def trim(df,boundaries=(50,3000),source='spectrum',target="spectrum"):
    for index, row in df.iterrows():
        df.at[index,target] = row[source].trim_axes(method='x-axis',boundaries=boundaries)
    return df

def baseline(df,baseline_kwargs=None,source='spectrum',target="spectrum"):
    if baseline_kwargs is None:
        baseline_kwargs = {"niter" : 100 }
    for index, row in df.iterrows():
        df.at[index,target] = row[source].subtract_baseline_rc1_snip(**baseline_kwargs)       
    return df

def calc_peak_intensity(spe,peak=144,boundaries=None,prominence_coeff=0.01,no_fit=False):
    try:
        peak_intensity="height"
        if boundaries is None:
            boundaries=(peak-50, peak+50)
        spe = spe.trim_axes(method='x-axis', boundaries=boundaries)
        prominence = spe.y_noise_MAD() * prominence_coeff
        candidates = spe.find_peak_multipeak(prominence=prominence)
        
        
        fit_res = spe.fit_peak_multimodel(profile='Voigt', candidates=candidates,no_fit=no_fit)
        df = fit_res.to_dataframe_peaks()
        df["sorted"] = abs(df["center"] - peak) #closest peak to 144
        df_sorted = df.sort_values(by='sorted')
        index_left = np.searchsorted(spe.x, df_sorted["center"][0] , side='left', sorter=None)
        index_right = np.searchsorted(spe.x, df_sorted["center"][0] , side='right', sorter=None)
        intensity_val = (spe.y[index_right] + spe.y[index_left])/2.0
        _label = "intensity = {:.3f} {} ={:.3f} amplitude={:.3f} center={:.1f}".format(
            intensity_val,peak_intensity,df_sorted.iloc[0][peak_intensity],
            df_sorted.iloc[0]["amplitude"],df_sorted.iloc[0]["center"])
        
        return intensity_val, fit_res
    except Exception as err:
        print(err)
        return None,None

def spe_area(df,title="area",boundaries=(50, 3000)):
    for index, row in df.iterrows():
        spe = row['spectrum']
        sc = spe.trim_axes(method='x-axis', boundaries=boundaries)  
        df.at[index,title] = np.sum(sc.y * np.diff(sc.x_bin_boundaries))


def laser_power_regression(df,peak_at = 144,boundaries=None,no_fit=False,title=""):
    fig, ax = plt.subplots(df.shape[0],1,figsize=(12,12))
    r = 0
    for index, row in df.iterrows():
        spe = row['spectrum']
        if boundaries is None:
            boundaries=(peak_at-50, peak_at+50)
        df.at[index,'peak_intensity'],fit_res = calc_peak_intensity(spe,peak=peak_at,boundaries=boundaries,no_fit=no_fit)
        spe.trim_axes(method='x-axis',boundaries=boundaries).plot(ax=ax[r], fmt=':',label=row["laser_power_mW"])
        if fit_res is not None:
            fit_res.plot(ax=ax[r])            
        r = r+1

    plt.savefig("test_twinning_peaks_{}.png".format(title))        
    print(df[['laser_power_mW','peak_intensity']])
    return LinearRegression().fit(df[["laser_power_mW"]].values,df["peak_intensity"].values)
    #print("Intercept:", model.intercept_)
    #print("Slope (Coefficient):", model.coef_[0])    


def normalize_by_laserpower_time(reference,twinned,source='spectrum',target='spectrum'):
    for (index_refere4nce, row_reference), (index_twinned, row_twinned) in zip(reference.iterrows(), twinned.iterrows()):
        laser_power_ratio = row_reference['laser_power_mW'] / row_twinned['laser_power_mW']
        time_ratio = row_reference['time_ms'] / row_twinned['time_ms']
        spe = row_twinned[source]
        twinned.at[index_twinned, target] = rc2.spectrum.Spectrum(spe.x,spe.y*laser_power_ratio*time_ratio)

def apply_correction_factor(df,CF=1,source='spectrum',target='spectrum'):
    for index, row in df.iterrows():
        df.at[index,target] = row[source]*CF

def plot_spectra(df,title="spectra",source='spectrum'):
    fig, ax = plt.subplots(1, 1, figsize=(12, 2))
    for index, row in df.iterrows():
        row[source].plot(ax=ax)
    plt.savefig("test_twinning_{}.png".format(title))        

def plot_model(A,B,correction_factor =1, regression_A=(0,1), regression_B=(0,1)):
    fig, axes = plt.subplots(1,2, figsize=(10,4)) 
    axes[0].plot(A["laser_power_mW"],A["peak_intensity"],'o',label=A["device_id"].unique())
    
    A_pred = A["laser_power_mW"]*regression_A[1] + regression_A[0]
    axes[0].plot(A["laser_power_mW"],A_pred,'-',label="{:.2e} * LP + {:.2e}".format(regression_A[1],regression_A[0]))
    
    axes[0].plot(B["laser_power_mW"],B["peak_intensity"],'+',label=B["device_id"].unique())

    #axes[0].plot(B["laser_power_mW"],B["peak_intensity"]*correction_factor,'+',label="{} corrected".format(B["device_id"].unique()))

    B_pred =B["laser_power_mW"]*regression_B[1] + regression_B[0]
    axes[0].plot(B["laser_power_mW"],B_pred,'-',label="{:.2e} * LP + {:.2e}".format(regression_B[1],regression_B[0]))

    axes[0].set_ylabel("Peak intensity of the (fitted) peak @ 144cm-1")
    axes[0].set_xlabel("laser power, %")
    axes[0].legend()
    bar_width = 0.2  # Adjust this value to control the width of the groups
    bar_positions = np.arange(len(A["laser_power_percent"].values))
    axes[1].bar(bar_positions -bar_width,A["area"], width=bar_width,label=str(A["device_id"].unique()))
    bar_positions = np.arange(len(B["laser_power_percent"].values))
    axes[1].bar(bar_positions  ,B["area"],width=bar_width,label=str(B["device_id"].unique()))
    axes[1].bar(bar_positions + bar_width,B["area_harmonized"],width=bar_width,label="{} harmonized CF={:.2e}".format(B["device_id"].unique(),correction_factor))
    axes[1].set_ylabel("spectrum area")
    axes[1].set_xlabel("laser power, %")
    # Set the x-axis positions and labels
    plt.xticks(bar_positions, B["laser_power_percent"])
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("test_twinning_evaluation.png")        

def test_metadata4twinning(reference_with_replicates,twinned_with_replicates):
    reference = process_replicates(reference_with_replicates,title="reference")
    boundaries=(50, 2000)
    reference = trim(reference,boundaries=boundaries)
    #
    reference = baseline(reference)
    assert 5 == reference.shape[0]
    plot_spectra(reference,"reference")
    twinned = process_replicates(twinned_with_replicates,title="twinned")
    twinned = trim(twinned,boundaries=boundaries)
    assert 5 == twinned.shape[0]
    plot_spectra(reference,"twinned")
    normalize_by_laserpower_time(reference,twinned)
    plot_spectra(twinned,"twinned_normalized")
    twinned = baseline(twinned)
    plot_spectra(twinned,"twinned_normalized_baseline")

    boundaries4peak = boundaries # (50, 500)
    boundaries4area = boundaries
    spe_area(reference,title="area",boundaries=boundaries4area)
    spe_area(twinned,title="area",boundaries=boundaries4area)

    
    model_reference = laser_power_regression(reference,title="reference",no_fit=False)
    print("slope reference",model_reference.coef_[0])
    model_twinned = laser_power_regression(twinned,title="twinned",no_fit=False)
    print("slope twinned",model_twinned.coef_[0])
    CF = model_reference.coef_[0] / model_twinned.coef_[0]
    print("correction factor",CF)
    apply_correction_factor(twinned,CF)

    spe_area(twinned,title="area_harmonized",boundaries=boundaries4area)
    plot_spectra(twinned,"twinned_corrected")

    plot_model(reference,twinned,correction_factor=CF,
               regression_A=(model_reference.intercept_,model_reference.coef_[0]),
               regression_B=(model_twinned.intercept_,model_twinned.coef_[0]))
    