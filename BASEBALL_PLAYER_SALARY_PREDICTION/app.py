import os
import pandas as pd
import streamlit as st
from joblib import load


@st.cache_resource
def load_artifact():
	"""Load the persisted artifact (model + optional scaler)."""
	base_dir = os.path.dirname(os.path.abspath(__file__))
	model_path = os.path.join(base_dir, "models", "best_model.joblib")

	if not os.path.exists(model_path):
		raise FileNotFoundError(
			f"Model file not found at {model_path}. "
			"Run the training notebook cell that saves the best model first."
		)

	artifact = load(model_path)

	# Backwards compatibility: if only a model was saved, wrap it
	if not isinstance(artifact, dict) or "model" not in artifact:
		artifact = {"model": artifact, "scaler": None}

	return artifact


def main():
	st.title("IPL Player Salary Prediction")
	st.write(
		"This app uses the best trained model from the notebook "
		"to predict IPL player salaries."
	)

	# Load artifact once
	try:
		artifact = load_artifact()
	except Exception as e:
		st.error(str(e))
		return

	model = artifact.get("model")
	scaler = artifact.get("scaler")
	artifact_feature_names = artifact.get("feature_names")

	# Try to infer feature names (for single-input form)
	feature_names = None
	# 1) Prefer feature names saved in the artifact from the notebook
	if isinstance(artifact_feature_names, (list, tuple)) and artifact_feature_names:
		feature_names = list(artifact_feature_names)
	# 2) Fallback: try to read from the model itself
	elif hasattr(model, "feature_names_in_"):
		feature_names = list(model.feature_names_in_)

	tab_single, tab_batch = st.tabs(["Single input form", "Batch CSV upload"])

	with tab_single:
		st.markdown(
			"""
			Enter feature values below. Defaults are zero; change them as needed
			and click **Predict** to see the salary estimate.
			"""
		)

		if feature_names is None:
			st.info(
				"Single-input form is unavailable for this model because feature "
				"names are not stored. Use the batch CSV upload tab instead."
			)
		else:
			with st.form("single_input_form"):
				values = {}
				for name in feature_names:
					# Default 0.0; user can adjust
					values[name] = st.number_input(name, value=0.0)

				submitted = st.form_submit_button("Predict")
				if submitted:
					try:
						X_single = pd.DataFrame([values], columns=feature_names)
						if scaler is not None:
							X_scaled = scaler.transform(X_single)
							preds = model.predict(X_scaled)
						else:
							preds = model.predict(X_single)
						st.subheader("Predicted Salary")
						st.write(float(preds[0]))
					except Exception as e:
						st.error(
							"Error during prediction. Check that values are valid for the "
							"trained model.\n" + str(e)
						)

	with tab_batch:
		st.markdown(
			"""
			### Batch prediction from CSV
			1. Prepare a CSV file with the **same feature columns** used for training (excluding the target `Salary`).
			2. Upload the CSV below.
			3. Click **Predict Salaries** to get model predictions.
			"""
		)

		uploaded_file = st.file_uploader("Upload feature CSV", type=["csv"], key="csv_uploader")

		if uploaded_file is not None:
			try:
				input_df = pd.read_csv(uploaded_file)
			except Exception as e:
				st.error(f"Error reading CSV: {e}")
				return

			st.subheader("Input Data Preview")
			st.dataframe(input_df.head())

			if st.button("Predict Salaries", key="predict_csv"):
				try:
					X = input_df
					# If a separate scaler was saved, apply it before prediction
					if scaler is not None:
						X_scaled = scaler.transform(X)
						preds = model.predict(X_scaled)
					else:
						# For Pipeline models (with internal scaler), use raw features
						preds = model.predict(X)
				except Exception as e:
					st.error(
						"Error during prediction. Make sure the uploaded CSV has the "
						"same columns and preprocessing as the training data.\n" + str(e)
					)
					return

				result_df = input_df.copy()
				result_df["Predicted_Salary"] = preds

				st.subheader("Predicted Salaries")
				st.dataframe(result_df)

				csv_out = result_df.to_csv(index=False).encode("utf-8")
				st.download_button(
					label="Download Predictions as CSV",
					data=csv_out,
					file_name="ipl_salary_predictions.csv",
					mime="text/csv",
				)


if __name__ == "__main__":
	main()

