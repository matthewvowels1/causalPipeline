# PyHal9001
Python Bending for HAL_9001 R Package

This package only works with R calls (done with rpy2 package).
The Dockerfile permits you to run PyHal9001 safely. You can add python or R packages by adding them in requirements.txt/R_Requirements.txt (#TODO)

## How To:

```python
from hal9001.HAL import HAL9001

hal = HAL9001()
hal.fit(train_data, train_labels)
preds = hal.predict(test_data)

```

### TODO:
- pip package this

### DONE:
- Complete package with docs available here: https://cran.r-project.org/web/packages/hal9001/index.html
- 