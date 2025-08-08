import streamlit as st
from ultralytics import YOLO
from PIL import Image


model = YOLO("best.pt")

st.title("Cat Dog Detection App")
st.subheader("Detect cats and dogs in images using YOLOv8")

uploaded = st.file_uploader("Upload image...", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Image uploaded", use_container_width=True)

    if st.button("Detect"):
        results = model.predict(img)  
        res = results[0]

        st.image(res.plot(), caption="Result", use_container_width=True)

        
        counts = {}
        for cid in res.boxes.cls:
            name = res.names[int(cid)]
            counts[name] = counts.get(name, 0) + 1

        if counts:
            st.success("Detect: " + ", ".join(f"{v} {k}" for k, v in counts.items()))
        else:
            st.info("Nothing is detected.")