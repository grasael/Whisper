from langchain_community.llms import OpenAI  # Updated import
from langchain_experimental.agents import create_csv_agent
from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd

def main():
    load_dotenv()

    # Load the OpenAI API key from the environment variable
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    st.set_page_config(page_title="Ask me anything.")
    st.header("Nissan Whisper 🌬️")

    csv_file = st.file_uploader("Upload your CSV file here:", type="csv")
    if csv_file is not None:
        # Read and display the CSV content
        df = pd.read_csv(csv_file)
        st.text("CSV Preview")
        st.dataframe(df)

        # Rewind the file to ensure it's read from the beginning again
        csv_file.seek(0)

        agent = create_csv_agent(
            OpenAI(temperature=0), csv_file, verbose=True, allow_dangerous_code=True
        )

        # Initialize session state to store chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Get user input
        user_input = st.chat_input("Ask a question about your CSV: ")

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.spinner(text="In progress..."):
                response = agent.run(user_input)
                st.session_state.messages.append({"role": "assistant", "content": response})

        # Display the chat messages
        for message in st.session_state.messages:
            st.chat_message(message["role"]).markdown(message["content"])

if __name__ == "__main__":
    main()
