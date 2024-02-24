import streamlit as st
from streamlit_pills import pills

from app import chat_openai, get_latest_stock_data, metric_agg, recommend_stock


# Streamed response emulator
def response_generator(prompt):
    if prompt == "recommend":
        tmp_resp = metric_agg()
        final_resp = recommend_stock(tmp_resp)
        return tmp_resp, final_resp
    else:
        final_resp = chat_openai(prompt)

        return "", final_resp


st.title("StockGPT📈")
st.info(
    "Use this page to get recommend trade ideas based on  technical metrics."
    "feel free to chat",
    icon="ℹ️",
)

# add pills
# todo


pill_lists = [
    "recommend trade options",
    "fetch stock data",
]
selected = pills(
    "Outline your task!",
    pill_lists,
    icons=["💰", "🤖️"],
    clearable=True,
    index=None,
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            res1, res2 = response_generator(prompt)
            if len(res1) >= 1:
                wrap_res = f"this is what i get for diff metrics {res1} \n. For Conclusion: {res2}"
                st.markdown(wrap_res)
            else:
                wrap_res = res2
                st.markdown(res2)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": wrap_res})
    # st.session_state.messages.append({"role": "assistant", "content": res2})


if selected == pill_lists[0]:
    with st.spinner("Thinking..."):
        tmp_resp = metric_agg()
        final_resp = recommend_stock(tmp_resp)
        wrap_res = f"this is what i get for diff metrics {tmp_resp}. \n\n\n For Conclusion: {final_resp}"
        st.markdown(wrap_res)
    st.session_state.messages.append({"role": "assistant", "content": wrap_res})

if selected == pill_lists[1]:
    with st.spinner("Fecthing..."):
        get_latest_stock_data()
        st.markdown("stock fetched.")
    st.session_state.messages.append({"role": "assistant", "content": "stock fetched."})
