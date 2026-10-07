import numpy as np

class BasePredictor:
    def fit(self, X_train, y_train):
        pass
    def predict(self, X_test):
        raise NotImplementedError

class BaselineB1(BasePredictor):
    # Global Mean
    def __init__(self):
        self.global_mean = 0.0
        
    def fit(self, X_train, y_train):
        self.global_mean = y_train.mean()
        
    def predict(self, X_test):
        return np.full(len(X_test), self.global_mean)

class BaselineB2(BasePredictor):
    # Player Career Mean (with B1 fallback)
    def __init__(self):
        self.b1 = BaselineB1()
        
    def fit(self, X_train, y_train):
        self.b1.fit(X_train, y_train)
        
    def predict(self, X_test):
        preds = X_test['fantasy_points_career_mean'].fillna(self.b1.global_mean).values
        return preds

class BaselineB3(BasePredictor):
    # Recent Mean (last 5, fallback to B2)
    def __init__(self):
        self.b2 = BaselineB2()
        
    def fit(self, X_train, y_train):
        self.b2.fit(X_train, y_train)
        
    def predict(self, X_test):
        preds = X_test['fantasy_points_last_5']
        b2_preds = self.b2.predict(X_test)
        return np.where(preds.isna(), b2_preds, preds)

class BaselineB4(BasePredictor):
    # EWMA (fallback to B2)
    def __init__(self):
        self.b2 = BaselineB2()
        
    def fit(self, X_train, y_train):
        self.b2.fit(X_train, y_train)
        
    def predict(self, X_test):
        preds = X_test['fantasy_points_ewma_short']
        b2_preds = self.b2.predict(X_test)
        return np.where(preds.isna(), b2_preds, preds)

class BaselineB5(BasePredictor):
    # Role/Form Baseline: 0.6 * EWMA + 0.4 * Career + Role adjustment
    def __init__(self):
        self.b2 = BaselineB2()
        self.b4 = BaselineB4()
        
    def fit(self, X_train, y_train):
        self.b2.fit(X_train, y_train)
        self.b4.fit(X_train, y_train)
        
    def predict(self, X_test):
        ewma_preds = self.b4.predict(X_test)
        career_preds = self.b2.predict(X_test)
        
        base_preds = 0.6 * ewma_preds + 0.4 * career_preds
        
        # Simple role adjustment
        # Pure bowlers and batters might get slightly more opportunity in T20s than part-time ARs if weakly inferred
        # We will keep it simple: +2 for BAT/BOWL/WK
        role_adj = np.zeros(len(X_test))
        if 'role' in X_test.columns:
            # Just a tiny heuristic to show "role" inclusion
            role_adj = np.where(X_test['role'].isin(['BAT', 'BOWL']), 2.0, 0.0)
            
        return base_preds + role_adj
