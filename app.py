from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="YOLO Segmentation", page_icon="🔍", layout="wide")
st.title("YOLO Instance Segmentation")
st.caption("อัปโหลดภาพ (นก / แมว / ไก่ / คน) ปรับ Confidence แล้วทดลองทำนายด้วยโมเดลของคุณ")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = BASE_DIR / "best.pt"

@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_model(weights_path: str):
    return YOLO(weights_path)

with st.sidebar:
    st.header("ตั้งค่าการทำนาย")
    weights_path = st.text_input("ตำแหน่งไฟล์โมเดล (.pt)", value=str(DEFAULT_WEIGHTS))
    conf = st.slider("Confidence threshold", min_value=0.01, max_value=1.0, value=0.25, step=0.01)
    st.caption("Conf ต่ำ: พบวัตถุมากขึ้น แต่อาจมีผลบวกลวงเพิ่มขึ้น")
    st.caption("Conf สูง: คัดเฉพาะผลที่โมเดลมั่นใจมากขึ้น")

uploaded_file = st.file_uploader("เลือกภาพสำหรับทดสอบ", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:
        st.error(f"ไม่สามารถอ่านไฟล์ภาพได้: {exc}")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ภาพต้นฉบับ")
        st.image(image, use_container_width=True)

    if st.button("เริ่ม Prediction", type="primary", use_container_width=True):
        model_path = Path(weights_path).expanduser()
        if not model_path.is_file():
            st.error(f"ไม่พบไฟล์โมเดล: {model_path}\n\nคัดลอก best.pt มาไว้ข้าง app.py หรือระบุ Path ที่ถูกต้องในแถบด้านข้าง")
            st.stop()
        try:
            model = load_model(str(model_path.resolve()))
            if model.task != "segment":
                st.warning("ไฟล์โมเดลนี้ไม่ใช่ Segmentation Model กรุณาเลือก best.pt ที่เทรนด้วย YOLO Segment")
                st.stop()
            with st.spinner("กำลังประมวลผล..."):
                result = model.predict(source=np.array(image), conf=conf, verbose=False)[0]
            plotted_bgr = result.plot()
            plotted_rgb = plotted_bgr[:, :, ::-1]
            with col2:
                st.subheader("ผลลัพธ์ Segmentation")
                st.image(plotted_rgb, use_container_width=True)

            if result.boxes is not None and len(result.boxes) > 0:
                counts = {}
                rows = []
                for i, box in enumerate(result.boxes, start=1):
                    cls_id = int(box.cls.item())
                    cls_name = result.names[cls_id]
                    score = float(box.conf.item())
                    counts[cls_name] = counts.get(cls_name, 0) + 1
                    rows.append({"ลำดับ": i, "คลาส": cls_name, "Confidence": round(score, 3)})
                st.success(f"ตรวจพบ {len(result.boxes)} วัตถุ")
                st.write("จำนวนแยกตามคลาส: " + ", ".join(f"{k}: {v}" for k, v in counts.items()))
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.info("ไม่พบวัตถุที่ผ่านค่า Confidence ที่กำหนด ลองลดค่า conf แล้วทดสอบอีกครั้ง")
        except Exception as exc:
            st.error(f"เกิดข้อผิดพลาดระหว่าง Prediction: {exc}")
else:
    st.info("อัปโหลดภาพเพื่อเริ่มต้น")
