import anthropic
import math
import os
import requests


client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

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


# weather.gov API calls
def get_coordinates(city):
    """Convert a city name to lat/lon using Open-Meteo's geocoding API."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city, "count": 1, "language": "en", "format": "json"}
    response = requests.get(url, params=params, timeout=10)
    results = response.json().get("results")
    if not results:
        return None, None
    return results[0]["latitude"], results[0]["longitude"]

def get_weather(city):
    """Fetch a forecast from Weather.gov for the given city."""
    lat, lon = get_coordinates(city)
    if lat is None:
        return f"Couldn't find coordinates for '{city}'."

    # Step 1: get the forecast office and grid coordinates for this location
    points_url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
    headers = {"User-Agent": "simple-agent-demo (your@email.com)"}
    points_resp = requests.get(points_url, headers=headers, timeout=10)
    if points_resp.status_code != 200:
        return f"Weather.gov couldn't find a forecast for '{city}' (may be outside the US)."

    forecast_url = points_resp.json()["properties"]["forecast"]

    # Step 2: fetch the actual forecast
    forecast_resp = requests.get(forecast_url, headers=headers, timeout=10)
    periods = forecast_resp.json()["properties"]["periods"]

    # Return the next two periods (e.g. "This Afternoon" and "Tonight")
    summary = []
    for period in periods[:2]:
        summary.append(f"{period['name']}: {period['detailedForecast']}")
    return "\n".join(summary)

# --- Tool implementations ---

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

# --- The agent loop ---
# messages lives here now, outside the function, so it persists across turns

messages = []

def run_agent(user_message):
    messages.append({"role": "user", "content": user_message})

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    print(f"\nAgent: {block.text}\n")
            # Add the assistant's final response to history
            messages.append({"role": "assistant", "content": response.content})
            break

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  [Tool call: {block.name}({block.input})]")
                    result = run_tool(block.name, block.input)
                    print(f"  [Tool result: {result}]")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            messages.append({"role": "user", "content": tool_results})

# --- Interactive loop ---

print("Agent ready. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() in ("quit", "exit"):
        print("Goodbye.")
        break
    run_agent(user_input)