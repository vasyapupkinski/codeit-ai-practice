import streamlit as st
import time
import random
import pandas as pd

st.title("Hyper Parameter Tuning Simulatior")

if "history" not in st.session_state:
    st.session_state.history = []

with st.form("training_form"):
    st.subheader("Model Parameters Setting")

    col1, col2, col3 = st.columns(3)
    with col1:
        learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01)
    with col2:
        epochs = st.slider("Epochs", 1, 100, 10)
    with col3:
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128])

    submitted = st.form_submit_button("Start Training")

if submitted:
    st.write(f"Start Training with LR: {learning_rate} and Epochs: {epochs}")    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(100):
        time.sleep(0.1)
        progress_bar.progress(i + 1)
        status_text.text(f"Training {i + 1}%")

    accuracy = random.uniform(0.70, 0.99)
    loss = random.uniform(0.1, 0.5)

    st.success(f"Training Completed")
    st.write(f"Accuracy: {accuracy}")
    st.write(f"Loss: {loss}")

    st.session_state.history.append({
        "lr": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "accuracy": accuracy,
        "loss": loss
    })

if len(st.session_state.history) > 0:
    st.markdown("___")
    st.subheader("Training History(Session State Remained)")
    
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df)

    max_acc = history_df['accuracy'].max()
    st.metric(label="Best Accuracy", value=f"{max_acc:.4f}")

    st.subheader("Accuracy Trend")
    st.line_chart(history_df["accuracy"])

if st.button("Clear Session State"):
    st.session_state.history = []
    st.success("Session State Cleared")
    st.rerun()