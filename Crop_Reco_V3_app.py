from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
	page_title="Biswarup | Crop Advisor",
	page_icon="🌱",
	layout="wide",
	initial_sidebar_state="expanded",
)

st.markdown(
	"""
	<style>
	@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
	:root { --forest:#173f35; --leaf:#2c7654; --mint:#eaf4ec; --ink:#18342c; }
	html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
	.stApp { background:linear-gradient(180deg,#f7faf6 0%,#f4f7f2 100%); color:var(--ink); }
	[data-testid="stSidebar"] { background:#f0f5ef; border-right:1px solid #e0e9df; }
	[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2 { color:var(--forest); }
	.hero { padding:2.2rem 2.5rem; border-radius:24px; color:white; min-height:235px;
	  background:linear-gradient(90deg,rgba(13,48,38,.91),rgba(18,57,42,.57)),
	  url('https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1800&q=85') center 48%/cover; }
	.hero-kicker { color:#c8e7bd; text-transform:uppercase; letter-spacing:.16em; font-size:.76rem; font-weight:700; }
	.hero h1 { font-family:'Manrope',sans-serif; font-size:2.55rem; line-height:1.12; margin:.65rem 0; color:white; }
	.hero p { max-width:560px; color:#e6f0e8; font-size:1.02rem; margin:0; }
	.section-title { font-family:'Manrope',sans-serif; font-weight:800; color:var(--forest); font-size:1.3rem; margin:.4rem 0 .2rem; }
	.muted { color:#6d8177; font-size:.92rem; }
	.result-card { background:white; border:1px solid #e2ece2; border-radius:20px; padding:1.5rem; box-shadow:0 12px 32px rgba(28,67,48,.07); }
	.crop-name { color:var(--forest); font-family:'Manrope',sans-serif; font-size:2rem; font-weight:800; text-transform:capitalize; }
	div.stButton > button[kind="primary"] { background:#287451; border:0; border-radius:12px; padding:.72rem 1.2rem; font-weight:700; }
	div.stButton > button[kind="primary"]:hover { background:#1d5d40; border:0; }
	[data-testid="stMetric"] { background:white; border:1px solid #e3ece2; padding:1rem; border-radius:16px; }
	hr { border-color:#e5ece4; }
	</style>
	""",
	unsafe_allow_html=True,
)


@st.cache_resource
def load_artifact(artifact_path):
	import joblib

	return joblib.load(artifact_path)


def feature_defaults(name):
	key = str(name).lower().replace("_", " ").strip()
	presets = {
		"n": (0.0, 140.0, 1.0, 50.0), "nitrogen": (0.0, 140.0, 1.0, 50.0),
		"p": (0.0, 145.0, 1.0, 50.0), "phosphorus": (0.0, 145.0, 1.0, 50.0),
		"k": (0.0, 205.0, 1.0, 50.0), "potassium": (0.0, 205.0, 1.0, 50.0),
		"temperature": (0.0, 55.0, 0.1, 25.0), "temp": (0.0, 55.0, 0.1, 25.0),
		"humidity": (0.0, 100.0, 0.1, 65.0), "moisture": (0.0, 100.0, 0.1, 60.0),
		"ph": (0.0, 14.0, 0.1, 6.5), "rainfall": (0.0, 500.0, 1.0, 100.0),
		"rain": (0.0, 500.0, 1.0, 100.0), "soil moisture": (0.0, 100.0, 0.1, 60.0),
	}
	return presets.get(key, (0.0, 1000.0, 0.1, 1.0))


model_path = Path(__file__).resolve().parent / "best_crop_model.pkl"
feature_columns_path = Path(__file__).resolve().parent / "feature_columns.pkl"
label_encoder_path = Path(__file__).resolve().parent / "label_encoder.pkl"
scaler_path = Path(__file__).resolve().parent / "scaler.pkl"
soil_encoder_path = Path(__file__).resolve().parent / "soil_encoder.pkl"

with st.sidebar:
	st.markdown("## 🌿 Field settings")
	st.caption("Add your current soil and climate readings to find a suitable crop.")
	artifact_paths = {
		"Crop model": model_path,
		"Feature columns": feature_columns_path,
		"Crop label encoder": label_encoder_path,
		"Feature scaler": scaler_path,
	}
	for artifact_name, artifact_path in artifact_paths.items():
		if not artifact_path.exists():
			st.error(f"{artifact_name} file not found")
			st.code(str(artifact_path), language=None)
			st.stop()
	try:
		model = load_artifact(str(model_path))
		feature_names = load_artifact(str(feature_columns_path))
		label_encoder = load_artifact(str(label_encoder_path))
		scaler = load_artifact(str(scaler_path))
	except Exception as exc:
		st.error(f"Could not load the crop prediction artifacts: {exc}")
		st.stop()

	feature_names = list(feature_names)
	soil_feature = next(
		(feature for feature in feature_names if str(feature).strip().lower() == "soil_type"),
		None,
	)
	soil_encoder = None
	if soil_feature is not None:
		if not soil_encoder_path.exists():
			st.error("Soil encoder file not found")
			st.code(str(soil_encoder_path), language=None)
			st.stop()
		try:
			soil_encoder = load_artifact(str(soil_encoder_path))
		except Exception as exc:
			st.error(f"Could not load the soil encoder: {exc}")
			st.stop()

	st.markdown("### Your field readings")
	st.caption("Values are pre-filled with sensible starting points. Adjust them to match your field.")
	values = {}
	columns = st.columns(2)
	for index, feature in enumerate(feature_names):
		with columns[index % 2]:
			if feature == soil_feature:
				soil_type = st.selectbox(
					"Soil Type",
					options=list(soil_encoder.classes_),
					key=f"field_{index}",
				)
				values[feature] = soil_encoder.transform([soil_type])[0]
			else:
				low, high, step, default = feature_defaults(feature)
				values[feature] = st.number_input(
					str(feature).replace("_", " ").title(), min_value=low, max_value=high,
					value=default, step=step, key=f"field_{index}",
				)
	predict = st.button("🌱  Find my best crop", type="primary", use_container_width=True)

st.markdown(
	"""
	<div class="hero">
	  <div class="hero-kicker">Smarter choices · healthier harvests</div>
	  <h1>Grow what your land<br>is ready for.</h1>
	  <p>Personalized crop recommendations powered by your field conditions and our prediction model.</p>
	</div>
	""",
	unsafe_allow_html=True,
)
st.write("")

left, right = st.columns([1.2, 0.8], gap="large")
with left:
	st.markdown('<div class="section-title">A recommendation rooted in your data</div>', unsafe_allow_html=True)
	st.markdown('<div class="muted">Enter the readings in the panel, then let your conditions guide the next planting.</div>', unsafe_allow_html=True)
	st.write("")
	info_a, info_b, info_c = st.columns(3)
	
with right:
	st.image(
		"https://images.unsplash.com/photo-1464226184884-fa280b87c399?auto=format&fit=crop&w=1000&q=85",
		caption="Healthy soil is the start of every good harvest.",
		use_container_width=True,
	)

if predict:
	try:
		input_frame = pd.DataFrame([[values[name] for name in feature_names]], columns=feature_names)
		scaled_input = pd.DataFrame(
			scaler.transform(input_frame),
			columns=feature_names,
		)
		prediction = label_encoder.inverse_transform(model.predict(scaled_input))[0]
		st.divider()
		st.markdown('<div class="section-title">Your crop match</div>', unsafe_allow_html=True)
		st.markdown(
			f'<div class="result-card"><div class="muted">TOP RECOMMENDATION</div>'
			f'<div class="crop-name">🌾 {prediction}</div>'
			'<div class="muted">This crop is the closest match for the conditions you entered.</div></div>',
			unsafe_allow_html=True,
		)

		if hasattr(model, "predict_proba"):
			probabilities = model.predict_proba(scaled_input)[0]
			classes = getattr(model, "classes_", None)
			if classes is not None:
				crop_names = label_encoder.inverse_transform(classes)
				rankings = sorted(
					zip(crop_names, probabilities),
					key=lambda item: item[1],
					reverse=True,
				)[:3]
				if len(rankings) > 1:
					st.markdown("### 2nd and 3rd best crop recommendations")
					next_best = rankings[1:]
					rank_cols = st.columns(len(next_best))
					for rank, (column, (crop, score)) in enumerate(
						zip(rank_cols, next_best),
						start=2,
					):
						column.metric(
							f"{rank}{'nd' if rank == 2 else 'rd'} best: {str(crop).capitalize()}",
							f"{score:.1%} probability",
						)
	except Exception as exc:
		st.error(f"Prediction could not be completed. Check that the model accepts these inputs. Details: {exc}")

st.markdown("<br><div style='text-align:center;color:#829188;font-size:.82rem'>BISWARUP · CROP ADVISOR</div>", unsafe_allow_html=True)
