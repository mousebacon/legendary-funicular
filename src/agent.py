import anthropic
import math
import requests
import streamlit as st

API_KEY = st.secrets["ANTHROPIC_API_KEY"]

client = anthropic.Anthropic(api_key=API_KEY)

# --- Tool definitions ---

tools = [
    {
        "name": "calculate",
        "description": "Evaluates a math expression and returns the result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A math expression, e.g. '847 * 293' or 'sqrt(144)'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_weather",
        "description": "Returns the current weather forecast for a US city using Weather.gov.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name, e.g. 'Salt Lake City' or 'Chicago'"
                }
            },
            "required": ["city"]
        }
    }
]

# --- Weather helpers ---

def get_coordinates(city):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city, "count": 1, "language": "en", "format": "json"}
    response = requests.get(url, params=params, timeout=10)
    results = response.json().get("results")
    if not results:
        return None, None
    return results[0]["latitude"], results[0]["longitude"]

def get_weather(city):
    lat, lon = get_coordinates(city)
    if lat is None:
        return f"Couldn't find coordinates for '{city}'."
    points_url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
    headers = {"User-Agent": "simple-agent-demo (your@email.com)"}
    points_resp = requests.get(points_url, headers=headers, timeout=10)
    if points_resp.status_code != 200:
        return f"Weather.gov couldn't find a forecast for '{city}' (may be outside the US)."
    forecast_url = points_resp.json()["properties"]["forecast"]
    forecast_resp = requests.get(forecast_url, headers=headers, timeout=10)
    periods = forecast_resp.json()["properties"]["periods"]
    summary = []
    for period in periods[:2]:
        summary.append(f"{period['name']}: {period['detailedForecast']}")
    return "\n".join(summary)

# --- Tool runner ---

def run_tool(name, inputs):
    if name == "calculate":
        try:
            allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
            result = eval(inputs["expression"], {"__builtins__": {}}, allowed)
            return str(result)
        except Exception as e:
            return f"Error: {e}"
    elif name == "get_weather":
        return get_weather(inputs["city"])
    return "Unknown tool."

# --- Agent loop ---
# Returns the assistant's final text response as a string

def run_agent(user_message):
    st.session_state.messages.append({"role": "user", "content": user_message})

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            tools=tools,
            messages=st.session_state.messages
        )

        if response.stop_reason == "end_turn":
            st.session_state.messages.append({"role": "assistant", "content": response.content})
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""

        if response.stop_reason == "tool_use":
            st.session_state.messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = run_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            st.session_state.messages.append({"role": "user", "content": tool_results})

# --- Streamlit UI ---

st.title("Simple Agent")

# Initialize session state on first run
if "messages" not in st.session_state:
    st.session_state.messages = []
if "display" not in st.session_state:
    st.session_state.display = []

# Render chat history
for msg in st.session_state.display:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new input
if prompt := st.chat_input("Ask me anything..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.display.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = run_agent(prompt)
        st.markdown(reply)
    st.session_state.display.append({"role": "assistant", "content": reply})