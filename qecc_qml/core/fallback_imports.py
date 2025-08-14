"""
Fallback imports for optional dependencies to ensure core functionality works.
"""

# Fallback for torch
try:
    import torch
except ImportError:
    class FallbackTorch:
        @staticmethod
        def tensor(data):
            import numpy as np
            return np.array(data)
        
        @staticmethod
        def from_numpy(data):
            return data
            
        class optim:
            class Adam:
                def __init__(self, params, lr=0.01):
                    self.params = list(params)
                    self.lr = lr
                    
                def zero_grad(self):
                    pass
                    
                def step(self):
                    pass
    
    torch = FallbackTorch()

# Fallback for sklearn
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error
except ImportError:
    def train_test_split(X, y, test_size=0.2, random_state=None):
        import numpy as np
        if random_state:
            np.random.seed(random_state)
        
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        indices = np.random.permutation(n_samples)
        
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
    
    def accuracy_score(y_true, y_pred):
        import numpy as np
        return np.mean(y_true == y_pred)
    
    def mean_squared_error(y_true, y_pred):
        import numpy as np
        return np.mean((y_true - y_pred) ** 2)

# Fallback for pandas
try:
    import pandas as pd
except ImportError:
    class FallbackPandas:
        @staticmethod
        def DataFrame(data=None, columns=None):
            import numpy as np
            if data is None:
                return {}
            if isinstance(data, dict):
                return data
            return {"data": np.array(data)}
    
    pd = FallbackPandas()