Rebuild transformers and run Streamlit app

1) Rebuild preprocessing transformers (recommended if you run the app in your current environment):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python create_transformers.py
```

This fits a QuantileTransformer-based scaler and a OneHotEncoder-based encoder using `train.csv` and saves `scaler_transformer.pkl`, `encoder_transformer.pkl`, and `unique_classes.pkl` into the `model/` folder.

2) Run the app:

```bash
streamlit run app.py
```

3) If you see unpickling errors when loading existing pickles, they are usually due to scikit-learn version mismatch. Two options:
- Run the app in the original environment (if you know it): install `scikit-learn==1.6.1` and other packages matching that environment and run the app.
- Rebuild transformers using `create_transformers.py` in your current environment (recommended).

Notes:
- `create_transformers.py` uses the same `Feature_E` function as the app to ensure compatible columns.
- If you prefer, I can also add a small script to re-save the trained model pipeline using the current environment.
