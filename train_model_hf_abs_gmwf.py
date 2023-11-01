"""
17/02/2022 
- Zonal gravity wave momentum flux modelling
- 1h prediction
"""

import sys 
import netCDF4
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pandas as pd

# Data from ERA5
# --------------
## Input
## -----
AnaPath = 'path_to_the_datasets/'
AnaInput = netCDF4.Dataset(AnaPath + 'Input_ERA5_data_all_balloons.nc')
print('\n* Input data from reanalysis\n==========================')
print('- Number of observations: ' + str(AnaInput.variables['time'].shape[0]))
print('- Variables: ' + str(list(AnaInput.variables.keys())))

# Ballon data
# -----------
print('\n* Hourly balloon data from Rani\n===============================')
Bal_data_h = netCDF4.Dataset(AnaPath + 'All_STRATEOLE2_Balloon_1day15min.nc')
print('- Number of observations: ' + str(Bal_data_h.variables['time'].shape[0]))
print('- Variables: ' + str(list(Bal_data_h.variables.keys())))

image_format = 'png'
bal_levels = [49, 49, 51, 51, 51, 49, 49, 51]#[49, 51, 55, 53, 53, 49, 49, 51]  # Balloon levels from 1 to 8
index_level = [int((n-1)/2) for n in bal_levels]
balloon_indices = [[1,2565], [2566,5036], [5037,7464], [7465,9056], [9057,10974], [10975,12356], [12357,14351], [14352,16197]]
balloon_sizes = [balloon_indices[i][1]-balloon_indices[i][0] + 1 for i in range(len(balloon_indices))]


# ============================================================================================

bal_ = 1                         # Balloon number
resoluton = 3                    # time resolution            
typ = 'u1d'                      # waves with period from 15mn to 1day
sub = "abs"                      # absolute momentum fluxes
va_ = "qdm_u_" + sub             # name of variables
if sub != 'west':
    tranf = True
else:
    tranf = False
namefig = ['rf', 'extra', 'boost']
fignames = ['Random Forest', 'Extra Trees', 'Adaboost']
addname = 'fig_' + typ + '_bal' + str(bal_)  + '_3h_24h_' + sub + '.png'

# ============================================================================================
    
# Extract the inputs
# ------------------
input_lookup = list(AnaInput.variables.keys())
input_dict = {}
leng = len(AnaInput.variables['time'])
for va in input_lookup:
    if va in ['lnsp']:
        input_dict[va] = np.array(AnaInput.variables[va][:,2,2]).astype('float32')
    if va in ['temp', 'vitu', 'vitv']:
        for lev in [int((n-1)/2) for n in [51, 85, 111, 137]]:
            input_dict[va+str(int(2*lev+1))] = np.array(AnaInput.variables[va][:, lev, 2, 2]).astype('float32')
    if va == 'tp':
        input_dict['tp'] = np.array(AnaInput.variables[va][:,2,2]).astype('float32')
input_dict['sza'] = Bal_data_h['sza'][:]
input_dict['tp_mean'] = np.mean(AnaInput.variables['tp'][:,:,:], axis=(1,2))
input_dict['tp_sd'] = np.std(AnaInput.variables['tp'][:,:,:], axis=(1,2))

# All inputs
X_u = pd.DataFrame(input_dict)
# Input for zonal wind (u)
u = pd.DataFrame({
    'qdm_u': Bal_data_h.variables['qdm_u'][:], 
    'qdm_u_abs': Bal_data_h.variables['qdm_u_abs'][:],
    'qdm_u_west': Bal_data_h.variables['qdm_u_west'][:],
    'qdm_u_east': Bal_data_h.variables['qdm_u_east'][:]})
print(X_u.shape)
print(u.shape)
print(X_u.columns)
# Predictive models
# =================
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor

# Modifying time resolution
def modify_resolution(resolution, X, y):
    group = []
    start = 0
    balloon = [0]
    for i in range(8):
        s = balloon_sizes[i] // resolution
        r = balloon_sizes[i] % s
        group = group + list(np.repeat(range(start, start + s), resolution)) +  list(np.repeat(start + s,r))
        l1 = len(group)
        if r == 0:
            start += s
        else:
            start += (s + 1)
        balloon.append(max(group)+1)
    x_ = X.groupby(by = group).mean()
    y_ = y.groupby(by = group).mean()
    return x_, y_, balloon

def av_resolution(resulution, y):
    a = len(y) // resulution
    r = len(y) % a
    if r < resulution/2:
        group = np.concatenate((np.repeat(list(range(0,a)), resulution), np.repeat(a-1, r)))
    else:
        group = np.concatenate((np.repeat(list(range(0,a)), resulution), np.repeat(a, r)))
    return y.groupby(by = group).mean()

# modified time resolution
x_, y_, cut_id = modify_resolution(resoluton, X_u, u)
id_ = np.full(len(y_[va_][:]), True)
id_[cut_id[bal_-1]:cut_id[bal_]] = False
x_train = x_.iloc[id_,:].reset_index(drop='index')
y_train = y_[va_][id_].reset_index(drop='index')

x_test = x_.iloc[~id_,:].reset_index(drop='index')
y_test = y_[va_][~id_].reset_index(drop='index')
scaler = StandardScaler()
x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns).reset_index(drop='index')
x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_train.columns).reset_index(drop='index')

print(x_train.shape)
print(x_test.shape)

result_path = 'path_for_saving_results/'
img_path = 'path_for_saving_figure/'

all_predictions = {}
opt_param = {}


### Random Forest 
### =============
grid = { 
    'n_estimators':  [300, 500, 800, 1000],
    'max_features': np.int32(np.linspace(2,x_train_scaled.shape[1], 8)),
    'min_samples_leaf' : [10, 30, 50, 100, 150, 300]
}
rf_cv = GridSearchCV(
    estimator=RandomForestRegressor(), 
    param_grid = grid, 
    cv = 10)
rf_cv.fit(x_train_scaled, y_train)
print(rf_cv.best_params_)
rf = RandomForestRegressor(
    n_estimators=rf_cv.best_params_['n_estimators'],
    max_features = rf_cv.best_params_['max_features'],
    min_samples_leaf = rf_cv.best_params_['min_samples_leaf'])
rf_fit = rf.fit(x_train_scaled, y_train)
y_hat_rf = rf_fit.predict(x_test_scaled).reshape(-1)
np.mean(np.abs(y_hat_rf-y_test))/np.mean(y_test)
opt_param['min_samples_leaf_rf'] = rf_cv.best_params_['min_samples_leaf']
opt_param['max_features_rf'] = rf_cv.best_params_['max_features']
opt_param['n_estimators_rf'] = rf_cv.best_params_['n_estimators']
print(opt_param)

## Extra Tree
# ===========
grid = { 
    'n_estimators':  [300, 500, 800, 1000],
    'max_features': np.int32(np.linspace(2,x_train_scaled.shape[1], 8)),
    'min_samples_leaf' : [2, 3, 5, 10, 15]
}
extra_cv = GridSearchCV(
    estimator=ExtraTreesRegressor(), 
    param_grid = grid,
    cv = 10)
extra_cv.fit(x_train_scaled, y_train)
print(extra_cv.best_params_)
extra = ExtraTreesRegressor(
    n_estimators=extra_cv.best_params_['n_estimators'],
    max_features = extra_cv.best_params_['max_features'],
    min_samples_leaf = extra_cv.best_params_['min_samples_leaf'])
extra_fit = extra.fit(x_train_scaled, y_train)
y_hat_extra = extra_fit.predict(x_test_scaled).reshape(-1)
np.mean(np.abs(y_hat_extra-y_test))/np.mean(y_test)
opt_param['min_samples_leaf_extra'] = extra_cv.best_params_['min_samples_leaf']
opt_param['max_features_extra'] = extra_cv.best_params_['max_features']
opt_param['n_estimators_extra'] = extra_cv.best_params_['n_estimators']
print(opt_param)

### Boosting
### ========

nfolds = 5
nrepeats = 2
k_cv = RepeatedKFold(n_repeats=nrepeats, n_splits=nfolds)
max_depth = [30, 50, 75, 100, 300, 500] #[10, 30, 50, 75, 100, 200, 300, 400, 500]
max_features = np.int32(np.linspace(2,x_train_scaled.shape[1], 8))
ntree = [300, 500, 750, 1000]
cv_error = {}
for i, (id_train, id_test) in enumerate(k_cv.split(x_train_scaled)):
    temp = np.zeros((len(ntree), len(max_depth), len(max_features)))
    x_tr = x_train_scaled.iloc[id_train,:].reset_index(drop=True)
    y_tr = y_train.values[id_train]
    x_te = x_train_scaled.iloc[id_test,:].reset_index(drop=True)
    y_te = y_train.values[id_test]
    print('CV boost :', i)
    m = 1
    for j in range(len(ntree)):
        for k in range(len(max_depth)):
            for l in range(len(max_features)):
                print('process :', m)
                m += 1
                boost = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=int(max_depth[k]), max_features=int(max_features[l])))
                boost.n_estimators = int(ntree[j])
                boost_fit = boost.fit(x_tr, y_tr)
                pred = boost_fit.predict(x_te)
                temp[j,k,l] = np.sqrt(mean_squared_error(y_te, pred))/np.mean(y_te)
    cv_error['fold'+str(i+1)] = temp
    
av_cv_err = np.zeros((len(ntree), len(max_depth), len(max_features)))
for key, value in cv_error.items():
    av_cv_err += np.array(value)/(nfolds*nrepeats)
opt_id = np.where(av_cv_err == np.min(av_cv_err))
print(opt_id)

opt_parameters = {'n_estimators': ntree[opt_id[0][0]], 'max_depth': max_depth[opt_id[1][0]], 'max_features' : max_features[opt_id[2][0]]}

boost_fit = AdaBoostRegressor(
    base_estimator=DecisionTreeRegressor(max_depth=opt_parameters['max_depth'], max_features=opt_parameters['max_features']),
    n_estimators=opt_parameters['n_estimators']).fit(x_train_scaled, y_train)
y_hat_boost = boost_fit.predict(x_test_scaled).reshape(-1)
np.mean(np.abs(y_hat_boost-y_test))/np.mean(y_test)

opt_param['max_depth_boost'] = opt_parameters['max_depth']
opt_param['max_features_boost'] = opt_parameters['max_features']
opt_param['n_estimators_boost'] = opt_parameters['n_estimators']
print(opt_param)


# Save optimal parameters
param = xr.Dataset(opt_param)
param.to_netcdf(result_path + typ + '_parameters_bal' + str(bal_) + '_' + str(resoluton) + 'h_24h' + sub + '.nc', 'w')
param.close()

y_test1 = av_resolution(8, pd.DataFrame(y_test))
y_hat_rf1 = av_resolution(8, pd.DataFrame(y_hat_rf))
y_hat_extra1 = av_resolution(8, pd.DataFrame(y_hat_extra))
y_hat_boost1 = av_resolution(8, pd.DataFrame(y_hat_boost))

rmae_rf = np.mean(np.abs(y_test1.values - y_hat_rf1.values))/np.average(np.abs(y_test1))
rmae_extra = np.mean(np.abs(y_test1.values - y_hat_extra1.values))/np.average(np.abs(y_test1))
rmae_boost = np.mean(np.abs(y_test1.values - y_hat_boost1.values))/np.average(np.abs(y_test1))
  
coef_rf = np.corrcoef(y_test1.values.reshape(-1), y_hat_rf1.values.reshape(-1))
coef_extra = np.corrcoef(y_test1.values.reshape(-1), y_hat_extra1.values.reshape(-1))
coef_boost = np.corrcoef(y_test1.values.reshape(-1), y_hat_boost1.values.reshape(-1))
  
rmae = [rmae_rf, rmae_extra, rmae_boost]
r2 = [coef_rf[0,1], coef_extra[0,1], coef_boost[0,1]]

def save_images(data, namefig):
    plt.clf()
    fig = plt.figure(figsize=(18,5))
    axs = {}
    for i in range(len(namefig)):
        sub_data = data[[namefig[i], 'y_test']]
        axs[namefig[i]] = fig.add_subplot(1,3,i+1)
        axs[namefig[i]].plot(sub_data['y_test'], label = 'y_test')
        axs[namefig[i]].plot(sub_data[namefig[i]], label = 'prediction')
        axs[namefig[i]].set_title(fignames[i] + ' (RMAE='+str(round(rmae[i],3))+' ; Cor='+str(round(r2[i], 3))+')' )
        axs[namefig[i]].set_xlabel('index')
        axs[namefig[i]].set_ylabel('value')
        plt.legend()
    plt.savefig(img_path + addname, format = 'png', dpi=300, bbox_inches='tight')
    plt.clf()

all_pred24h = pd.DataFrame(
      {
          'rf' : y_hat_rf1.values.reshape(-1),
          'extra' : y_hat_extra1.values.reshape(-1),
          'boost' : y_hat_boost1.values.reshape(-1),
          'y_test' : y_test1.values.reshape(-1)
      }
)

save_images(all_pred24h, namefig)

df = xr.Dataset(all_pred24h)
df.to_netcdf(result_path + "all_prediction_" + typ +"_bal" + str(bal_) + "_3h_24h" + sub + ".nc")
df.close()

print('-------------- DONE ----------------')