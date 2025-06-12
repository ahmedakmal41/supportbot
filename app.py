import os
import streamlit as st
from openai import AzureOpenAI

# Streamlit page configuration
st.set_page_config(page_title="Dave - AI Assistant", page_icon="🤖", layout="centered")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Azure OpenAI configuration from environment variables
endpoint = os.getenv("ENDPOINT_URL")
deployment = os.getenv("DEPLOYMENT_NAME")
search_endpoint = os.getenv("SEARCH_ENDPOINT")
search_key = os.getenv("SEARCH_KEY")
search_index = os.getenv("SEARCH_INDEX_NAME")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

# Validate environment variables
required_env_vars = {
    "ENDPOINT_URL": endpoint,
    "DEPLOYMENT_NAME": deployment,
    "SEARCH_ENDPOINT": search_endpoint,
    "SEARCH_KEY": search_key,
    "SEARCH_INDEX_NAME": search_index,
    "AZURE_OPENAI_API_KEY": subscription_key
}
missing_vars = [key for key, value in required_env_vars.items() if not value]
if missing_vars:
    st.error(f"Missing required environment variables: {', '.join(missing_vars)}. Please set them in .streamlit/secrets.toml or the deployment platform's secrets management.")
    st.stop()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2025-01-01-preview",
)

# System prompt
system_prompt = {
    "role": "system",
    "content": "You are an AI assistant named Dave who is a very friendly assistant, greets in a good manner, asks the user with how he can help them that helps people find information from the knowledge sources it has present otherwise it politely says to contact the support representative through email, also DONT mention document like [doc1] JUST ANSWER NO REFERENCE, if you have name of the pdf mention that otherwise dont"
}

# Function to get response from Azure OpenAI
def get_response(user_input):
    messages = [system_prompt] + st.session_state.messages + [{"role": "user", "content": user_input}]
    
    try:
        completion = client.chat.completions.create(
            model=deployment,
            messages=messages,
            max_tokens=800,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False,
            extra_body={
                "data_sources": [{
                    "type": "azure_search",
                    "parameters": {
                        "endpoint": search_endpoint,
                        "index_name": search_index,
                        "semantic_configuration": "default",
                        "query_type": "vector_semantic_hybrid",
                        "fields_mapping": {},
                        "in_scope": True,
                        "filter": None,
                        "strictness": 3,
                        "top_n_documents": 5,
                        "authentication": {
                            "type": "api_key",
                            "key": search_key
                        },
                        "embedding_dependency": {
                            "type": "deployment_name",
                            "deployment_name": "text-embedding-3-small"
                        }
                    }
                }]
            }
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.title("🤖 Dave - Your Friendly AI Assistant")
st.write("Hello! How can I assist you today?")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Dave is thinking..."):
            response = get_response(prompt)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
