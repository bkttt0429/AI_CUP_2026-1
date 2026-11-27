import importlib.metadata as md, sys
import numpy, pandas, scipy, sklearn, statsmodels, pywt, torch, tsfel
print(sys.prefix); print({m: md.version(m) for m in ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "PyWavelets", "torch", "tsfel")})
print({m.__name__: m.__file__ for m in (numpy, pandas, torch, tsfel)})
