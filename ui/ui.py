import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import tempfile
from firstprototype.cloudinary import upload_image
from firstprototype.first import graph

st.title("Zero Waste AI Chef")

uploaded_file = st.file_uploader(
    "Upload fridge image",
    type=["jpg","jpeg","png","webp"]
)

if uploaded_file:

    # Save temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    st.image(temp_path)

    if st.button("Analyze Fridge"):

        with st.spinner("Uploading image..."):
            image_url = upload_image(temp_path)

        st.write("Image URL:", image_url)

        with st.spinner("Running AI Vision..."):

            result = graph.invoke({
                "image_url": image_url,
                "budget": 500,
                "nutrition_goal": "high_protein"
            })

        st.subheader("Detected Inventory")
        st.json(result["inventory"])
        st.subheader("⚠ At Risk Foods")

        risk_items = result.get("risk_items", [])

        if risk_items:
            for item in risk_items:
                st.warning(f"{item['item']} — {item['reason']}")
        else:
            st.success("No high-risk food detected.")