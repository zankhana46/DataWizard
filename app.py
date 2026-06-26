import streamlit as st
import pandas as pd
import io
import os
import time
import store

st.set_page_config(
    page_title="DataWizard",
    page_icon="🧙",
    layout="wide",
)

# ── Helper functions (must be defined before use) ─────────────────────────────

def _looks_like_table(text: str) -> bool:
    """Heuristic: at least 2 lines containing ' | ' → treat as table."""
    lines = [l for l in text.splitlines() if " | " in l]
    return len(lines) >= 2


def _render_table(text: str):
    """Parse pipe-separated text into a DataFrame and display it."""
    lines = text.splitlines()
    table_lines = []
    pre_text = []
    post_text = []
    in_table = False

    for line in lines:
        if " | " in line and not set(line.strip()) <= set("-| "):
            in_table = True
            table_lines.append(line)
        elif in_table and set(line.strip()) <= set("-| "):
            continue  # separator row
        elif in_table:
            post_text.append(line)
        else:
            pre_text.append(line)

    if pre_text:
        st.markdown("\n".join(pre_text))

    if table_lines:
        try:
            header = [h.strip() for h in table_lines[0].split("|")]
            rows = [
                [c.strip() for c in row.split("|")]
                for row in table_lines[1:]
                if row.strip()
            ]
            df = pd.DataFrame(rows, columns=header)
            st.dataframe(df, use_container_width=True)
        except Exception:
            st.markdown("\n".join(table_lines))

    if post_text:
        st.markdown("\n".join(post_text))


def get_executor():
    if st.session_state.get("agent_executor") is None:
        with st.spinner("Initializing DataWizard agent…"):
            from agent import build_agent_executor
            st.session_state["agent_executor"] = build_agent_executor()
    return st.session_state["agent_executor"]


# ── Session state defaults ────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = {}

if "agent_executor" not in st.session_state:
    st.session_state["agent_executor"] = None

# Auto-load sample CSVs from data/ into the thread-safe store
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
for fname in os.listdir(DATA_DIR):
    if fname.endswith(".csv") and fname not in store.all_files():
        try:
            store.put(fname, pd.read_csv(os.path.join(DATA_DIR, fname)))
        except Exception:
            pass

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧙 DataWizard")
    st.markdown("---")

    st.subheader("Upload CSV Files")
    uploaded = st.file_uploader(
        "Drop your CSV files here",
        type=["csv"],
        accept_multiple_files=True,
        help="Files are stored in memory for this session only.",
    )

    if uploaded:
        for f in uploaded:
            if f.name not in store.all_files():
                df = pd.read_csv(io.BytesIO(f.read()))
                store.put(f.name, df)
                st.success(f"Loaded: {f.name} ({df.shape[0]}×{df.shape[1]})")

    if store.all_files():
        st.markdown("**Available datasets:**")
        for name, df in store.all_files().items():
            st.caption(f"• {name} — {df.shape[0]} rows × {df.shape[1]} cols")

    st.markdown("---")
    st.info("**Database:** `sample.db` is always available — ask questions about it anytime.")

    st.markdown("---")
    st.markdown("**Example questions:**")
    st.markdown(
        "- What tables are in the database?\n"
        "- Show total revenue by region\n"
        "- How many customers are on each plan?\n"
        "- Analyze sample_regression.csv\n"
        "- Train a classification model on sample_classification.csv targeting 'defaulted'"
    )

    if st.button("Clear chat history"):
        st.session_state["messages"] = []
        st.session_state["agent_executor"] = None
        st.rerun()

# ── Main chat area ────────────────────────────────────────────────────────────

st.title("🧙 DataWizard — AI Data Analysis & SQL Agent")
st.caption("Ask anything about your CSVs or the built-in sample database.")

def _render_cot(steps: list) -> None:
    """Render a chain-of-thought step list inside an expander."""
    if not steps:
        return
    with st.expander("🧠 Step-by-step reasoning", expanded=False):
        for i, (kind, text) in enumerate(steps, 1):
            if kind == "decide":
                st.markdown(f"**Step {i} — Agent decides:** {text}")
            elif kind == "tool_result":
                st.markdown(f"**Step {i} — Tool result:**")
                st.code(text, language="text")
            st.divider()


for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        for title, img_bytes in msg.get("plots", []):
            st.image(img_bytes, caption=title, use_column_width=True)
        for label, records in msg.get("tables", []):
            st.caption(label)
            st.dataframe(pd.DataFrame(records), use_container_width=True)
        content = msg["content"]
        if content:
            st.markdown(content)
        for label, snippet in msg.get("code", []):
            with st.expander(f"🐍 {label}"):
                st.code(snippet, language="python")
        _render_cot(msg.get("cot", []))

if prompt := st.chat_input("Ask a question about your data or the database…"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Initialise outside try/with so the messages.append below never hits NameError
    saved_plots: list = []
    saved_tables: list = []
    code_snippets: list = []
    cot_steps: list = []

    with st.chat_message("assistant"):
        status_box = st.empty()
        response = ""
        try:
            agent = get_executor()
            config = {"configurable": {"thread_id": "datawizard-session"}, "recursion_limit": 15}
            deadline = time.time() + 180

            for state in agent.stream(
                {"messages": [("human", prompt)]},
                config=config,
                stream_mode="values",
            ):
                if time.time() > deadline:
                    response = "_(timed out after 3 minutes)_"
                    break

                messages = state.get("messages", [])
                if not messages:
                    continue

                last = messages[-1]
                last_type = type(last).__name__

                if last_type == "ToolMessage":
                    tool_name = getattr(last, "name", "tool")
                    status_box.caption(f"⚙ Ran `{tool_name}`…")
                    result_text = last.content or ""
                    if isinstance(result_text, list):
                        result_text = str(result_text)
                    # Truncate very long tool outputs in CoT display
                    display_result = result_text[:800] + ("…" if len(result_text) > 800 else "")
                    cot_steps.append(("tool_result", f"[{tool_name}] {display_result}"))

                elif last_type in ("AIMessage", "AIMessageChunk"):
                    content = last.content
                    if isinstance(content, list):
                        content = " ".join(
                            p.get("text", "") for p in content
                            if isinstance(p, dict) and p.get("type") == "text"
                        )
                    tool_calls = getattr(last, "tool_calls", None)
                    if tool_calls:
                        for tc in tool_calls:
                            name = tc.get("name", "?")
                            args = tc.get("args", {})
                            args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
                            cot_steps.append(("decide", f"Call `{name}({args_str})`"))
                    elif isinstance(content, str) and content.strip():
                        response = content

        except Exception as e:
            response = f"An error occurred: {e}"

        status_box.empty()

        if not response:
            response = "_(No response — the agent completed its steps but produced no summary. Try rephrasing.)_"

        # Give worker threads a moment to finish writing to the queues
        time.sleep(0.3)

        # Drain plots, save bytes for persistence, then render
        plots = store.pop_plots()
        saved_plots = []
        for title, buf in plots:
            img_bytes = buf.read()
            saved_plots.append((title, img_bytes))
            st.image(img_bytes, caption=title, use_column_width=True)

        if response:
            st.markdown(response)

        # Render and persist DataFrames
        raw_dfs = store.pop_dataframes()
        for label, df in raw_dfs:
            st.caption(label)
            st.dataframe(df, use_container_width=True)
            saved_tables.append((label, df.to_dict(orient="records")))

        # Render and persist code snippets
        code_snippets = store.pop_code()
        for label, snippet in code_snippets:
            with st.expander(f"🐍 {label}"):
                st.code(snippet, language="python")

        # Render CoT toggle
        _render_cot(cot_steps)

    st.session_state["messages"].append({
        "role": "assistant",
        "content": response,
        "plots": saved_plots,
        "tables": saved_tables,
        "code": code_snippets,
        "cot": cot_steps,
    })
