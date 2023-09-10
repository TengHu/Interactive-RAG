import logging
import os

import openai
import streamlit as st
from bot import RAGBot
from streamlit_pills import pills

openai.api_key = os.getenv("OPENAI_API_KEY")
logging.basicConfig(
    filename="app.log",
    filemode="a",
    format="%(asctime)s.%(msecs)04d %(levelname)s {%(module)s} [%(funcName)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@st.cache_resource
def get_agent():
    logger.info("Loading RAG Bot ...")
    agent = RAGBot(logger, st)
    return agent


if __name__ == "__main__":
    agent = get_agent()

    st.subheader("RAG Bot with Search Integration")
    user_input = st.text_input("You: ", placeholder="Ask me anything ...", key="input")

    if st.button("Submit", type="primary"):
        st.markdown("----")
        res_box = st.empty()

        response = agent(user_input)

        if type(response) == str:
            res_box.markdown(f"*{response}*")
            res_box.write(response)

            agent.messages.append({"role": "assistant", "content": response})
        else:
            report = []
            result_to_display = ""
            # Looping over the response
            for resp in response:
                if hasattr(resp.choices[0].delta, "content"):
                    report.append(resp.choices[0].delta.content)
                    result_to_display = "".join(report).strip()
                    res_box.markdown(f"*{result_to_display}*")
                res_box.write(result_to_display)

            agent.messages.append({"role": "assistant", "content": result_to_display})
    st.markdown("----")
