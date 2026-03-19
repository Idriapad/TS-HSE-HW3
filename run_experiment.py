import pandas as pd
import numpy as np
import os
import src.config as config
from tqdm import tqdm
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, AutoETS, AutoTheta
from catboost import CatBoostRegressor
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from neuralforecast.models import PatchTST
from neuralforecast import NeuralForecast

def prepare_train_test(df, horizon):
    df = df.sort_values(by=['unique_id', 'ds']).reset_index(drop=True)
    
    test = df.groupby('unique_id').tail(horizon).copy()
    train = df.drop(test.index).copy()
    
    test['step'] = test.groupby('unique_id').cumcount() + 1
    
    return train, test

def run_baselines(train_df, horizon):
    models = [
        Naive(),
        SeasonalNaive(season_length=12),
        AutoETS(season_length=12),
        AutoTheta(season_length=12)
    ]
    
    sf = StatsForecast(
        models=models,
        freq='ME',
        n_jobs=-1
    )
    
    forecasts = sf.forecast(df=train_df, h=horizon)
    forecasts = forecasts.reset_index()
    
    forecasts['step'] = forecasts.groupby('unique_id').cumcount() + 1
    return forecasts

def run_catboost_recursive(train_df, horizon, lags=24):
    train_wide = train_df.pivot(index='ds', columns='unique_id', values='y')
    train_wide = train_wide.asfreq('ME')
    
    forecaster = ForecasterAutoregMultiSeries(
        regressor = CatBoostRegressor(iterations=200, verbose=0, random_state=config.RANDOM_SEED),
        lags = lags,
        encoding='ordinal'
    )
    
    forecaster.fit(series=train_wide)
    
    all_preds = []
    for uid in tqdm(train_df['unique_id'].unique(), desc="CatBoost Recursive"):
        df_window = train_df[train_df['unique_id'] == uid].set_index('ds')[['y']].copy()
        df_window.columns = [uid]
        df_window = df_window.asfreq('ME')
        
        pred = forecaster.predict(steps=horizon, levels=uid, last_window=df_window)
        pred_df = pd.DataFrame({
            'unique_id': uid,
            'step': list(range(1, horizon + 1)),
            'CatBoost_Recursive': pred[uid].values
        })
        all_preds.append(pred_df)
        
    return pd.concat(all_preds, ignore_index=True)

def run_catboost_direct(train_df, horizon, lags=24):
    train_df = train_df.copy()

    features = []
    for i in range(lags):
        col = f'lag_{i}'
        train_df[col] = train_df.groupby('unique_id')['y'].shift(i)
        features.append(col)
        
    targets = []
    for h in range(1, horizon + 1):
        target_col = f'target_step_{h}'
        train_df[target_col] = train_df.groupby('unique_id')['y'].shift(-h)
        targets.append(target_col)
        
    last_rows = train_df.groupby('unique_id').tail(1).copy()
    X_predict = last_rows[features]
    
    train_df = train_df.dropna(subset=features + targets)
    
    all_preds = []
    for h in tqdm(range(1, horizon + 1), desc="CatBoost Direct"):
        target_col = f'target_step_{h}'
        
        X_train = train_df[features]
        y_train = train_df[target_col]
        
        model = CatBoostRegressor(iterations=200, depth=6, verbose=0, random_state=config.RANDOM_SEED)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_predict)
        
        step_preds = pd.DataFrame({
            'unique_id': last_rows['unique_id'].values,
            'step': h,
            'CatBoost_Direct': preds
        })
        all_preds.append(step_preds)
        
    return pd.concat(all_preds, ignore_index=True)

def run_patchtst(train_df, horizon):
    model = PatchTST(
        h=horizon,
        input_size=36,
        patch_len=12,
        stride=12,
        revin=True,
        hidden_size=64,
        n_heads=4,
        batch_size=32,
        max_steps=300,
        scaler_type='standard',
        random_seed=config.RANDOM_SEED
    )
    
    nf = NeuralForecast(
        models=[model],
        freq='ME'
    )
    
    nf.fit(df=train_df)
    
    preds = nf.predict()
    preds = preds.reset_index()
    
    preds['step'] = preds.groupby('unique_id').cumcount() + 1
    
    return preds

def main():
    df = pd.read_csv(config.PROCESSED_DATA_PATH)
    df['ds'] = pd.to_datetime(df['ds'])
    
    train, test = prepare_train_test(df, config.HORIZON)
    
    base_preds = run_baselines(train, config.HORIZON)
    
    cb_rec_preds = run_catboost_recursive(train, config.HORIZON)
    
    cb_dir_preds = run_catboost_direct(train, config.HORIZON)

    patchtst_preds = run_patchtst(train, config.HORIZON)
    
    results = test.merge(base_preds.drop(columns=['ds']), on=['unique_id', 'step'], how='left')
    results = results.merge(cb_rec_preds, on=['unique_id', 'step'], how='left')
    results = results.merge(cb_dir_preds, on=['unique_id', 'step'], how='left')

    patchtst_preds = patchtst_preds[['unique_id', 'step', 'PatchTST']]
    results = results.merge(patchtst_preds, on=['unique_id', 'step'], how='left')

    if 'index' in results.columns:
        results = results.drop(columns=['index'])

    os.makedirs('results', exist_ok=True)
    results.to_csv('results/forecasts_final.csv', index=False)
    print("Прогнозы сохранены в results/forecasts_final.csv")

if __name__ == "__main__":
    main()
