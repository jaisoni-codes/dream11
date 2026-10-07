import lightgbm as lgb
import pandas as pd
import numpy as np

class LGBMFantasyPointsModel:
    def __init__(self, params):
        self.params = params
        self.model = None
        self.features = None
        self.categorical_features = ['role', 'role_confidence', 'role_source', 'format']
        
    def _prepare_features(self, X):
        # Exclude IDs and target
        exclude = ['match_id', 'player', 'date', 'fantasy_points']
        features = [c for c in X.columns if c not in exclude]
        
        # Keep features list to ensure order at inference
        if self.features is None:
            self.features = features
        else:
            features = self.features
            
        X_feats = X[features].copy()
        
        # Handle categoricals
        for cat in self.categorical_features:
            if cat in X_feats.columns:
                X_feats[cat] = X_feats[cat].astype('category')
                
        return X_feats
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train_feats = self._prepare_features(X_train)
        
        train_data = lgb.Dataset(X_train_feats, label=y_train)
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            X_val_feats = self._prepare_features(X_val)
            val_data = lgb.Dataset(X_val_feats, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')
            
        # LightGBM uses early stopping through callbacks in newer versions
        callbacks = []
        if X_val is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=20))
            
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
    def predict(self, X_test):
        X_test_feats = self._prepare_features(X_test)
        return self.model.predict(X_test_feats)
        
    def get_feature_importance(self):
        if self.model is None:
            return pd.DataFrame()
            
        gain = self.model.feature_importance(importance_type='gain')
        split = self.model.feature_importance(importance_type='split')
        
        df = pd.DataFrame({
            'feature': self.features,
            'gain_importance': gain,
            'split_importance': split
        })
        
        df = df.sort_values('gain_importance', ascending=False).reset_index(drop=True)
        return df
