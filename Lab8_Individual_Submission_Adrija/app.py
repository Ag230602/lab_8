import streamlit as st
import requests

st.title("🌪️ Disaster Forecast Explanation UI")
st.write("This tool uses our LoRA-adapted TinyLlama model to generate domain-specific explanations of disaster forecasts for emergency planners.")

instruction = st.text_input("Instruction:", "Explain hurricane forecast uncertainty for emergency planners.")
user_input = st.text_area("Forecast Input:", "Storm probability 74%, confidence interval ±85 km, coastal population exposure 390000.")

if st.button("Generate Explanation"):
    if not instruction or not user_input:
        st.warning("Please provide both instruction and input.")
    else:
        with st.spinner("Generating..."):
            try:
                response = requests.post(
                    "http://localhost:8000/generate",
                    json={"instruction": instruction, "input": user_input}
                )
                if response.status_code == 200:
                    explanation = response.json().get("explanation", "")
                    st.success("Explanation Generated:")
                    st.write(explanation)
                else:
                    st.error(f"Error from API: {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the FastAPI server. Is it running on port 8000?")
