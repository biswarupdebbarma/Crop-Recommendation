from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
	page_title="Verdant | Crop Advisor",
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
def load_model(model_path):
	import joblib

	return joblib.load(model_path)


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

with st.sidebar:
	st.markdown("## 🌿 Field settings")
	st.caption("Add your current soil and climate readings to find a suitable crop.")
	if not model_path.exists():
		st.error("Model file not found")
		st.code(str(model_path), language=None)
		st.stop()
	try:
		model = load_model(str(model_path))
	except Exception as exc:
		st.error(f"Could not load the crop model: {exc}")
		st.stop()

	model_features = getattr(model, "feature_names_in_", None)
	feature_names = list(model_features) if model_features is not None else [
		"N", "P", "K", "temperature", "humidity", "ph", "rainfall"
	]
	st.markdown("### Your field readings")
	st.caption("Values are pre-filled with sensible starting points. Adjust them to match your field.")
	values = {}
	columns = st.columns(2)
	for index, feature in enumerate(feature_names):
		low, high, step, default = feature_defaults(feature)
		with columns[index % 2]:
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
	info_a.metric("Inputs", f"{len(feature_names)} readings")
	info_b.metric("Approach", "Field-aware")
	info_c.metric("Made for", "Better harvests")
with right:
	st.image(
		"https://images.unsplash.com/photo-1464226184884-fa280b87c399?auto=format&fit=crop&w=1000&q=85",
		caption="Healthy soil is the start of every good harvest.",
		use_container_width=True,
	)

if predict:
	try:
		input_frame = pd.DataFrame([[values[name] for name in feature_names]], columns=feature_names)
		prediction = model.predict(input_frame)[0]
		st.divider()
		st.markdown('<div class="section-title">Your crop match</div>', unsafe_allow_html=True)
		st.markdown(
			f'<div class="result-card"><div class="muted">TOP RECOMMENDATION</div>'
			f'<div class="crop-name">🌾 {prediction}</div>'
			'<div class="muted">This crop is the closest match for the conditions you entered.</div></div>',
			unsafe_allow_html=True,
		)

		if hasattr(model, "predict_proba"):
			probabilities = model.predict_proba(input_frame)[0]
			classes = getattr(model, "classes_", None)
			if classes is not None:
				rankings = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)[:3]
				st.markdown("### Other promising options")
				rank_cols = st.columns(len(rankings))
				for column, (crop, score) in zip(rank_cols, rankings):
					column.metric(str(crop).capitalize(), f"{score:.0%} match")
	except Exception as exc:
		st.error(f"Prediction could not be completed. Check that the model accepts these inputs. Details: {exc}")

st.markdown("<br><div style='text-align:center;color:#829188;font-size:.82rem'>VERDANT · CROP ADVISOR</div>", unsafe_allow_html=True)
