from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class TravelState(TypedDict):
    user_input: str
    city: str
    weather: str
    suggestion: str
    final_answer: str


def parse_request(state: TravelState) -> dict:
    """从用户输入里提取一个最简单的城市信息。"""
    text = state["user_input"]

    if "北京" in text:
        city = "北京"
    elif "上海" in text:
        city = "上海"
    else:
        city = "未知城市"

    return {"city": city}


def mock_weather(state: TravelState) -> dict:
    """第一天先不用真实 API，先用假数据理解图的执行流程。"""
    city = state["city"]

    weather_map = {
        "北京": "晴天",
        "上海": "多云",
        "未知城市": "天气未知",
    }
    weather = weather_map.get(city, "天气未知")
    return {"weather": weather}


def make_suggestion(state: TravelState) -> dict:
    """根据城市和天气生成一个简单建议。"""
    city = state["city"]
    weather = state["weather"]

    if weather == "晴天":
        suggestion = f"{city}适合去公园或历史景点散步。"
    elif weather == "多云":
        suggestion = f"{city}适合去城市地标和室内展馆。"
    else:
        suggestion = f"{city}建议先确认天气，再决定出行安排。"

    return {"suggestion": suggestion}


def build_answer(state: TravelState) -> dict:
    """把前面节点产出的状态汇总成最终答案。"""
    final_answer = (
        f"用户问题: {state['user_input']}\n"
        f"识别城市: {state['city']}\n"
        f"模拟天气: {state['weather']}\n"
        f"旅行建议: {state['suggestion']}"
    )
    return {"final_answer": final_answer}


def build_graph():
    graph_builder = StateGraph(TravelState)

    graph_builder.add_node("parse_request", parse_request)
    graph_builder.add_node("mock_weather", mock_weather)
    graph_builder.add_node("make_suggestion", make_suggestion)
    graph_builder.add_node("build_answer", build_answer)

    graph_builder.add_edge(START, "parse_request")
    graph_builder.add_edge("parse_request", "mock_weather")
    graph_builder.add_edge("mock_weather", "make_suggestion")
    graph_builder.add_edge("make_suggestion", "build_answer")
    graph_builder.add_edge("build_answer", END)

    return graph_builder.compile()


def main() -> None:
    graph = build_graph()

    initial_state = {
        "user_input": "我想周末去深圳玩，请给我一个基础建议。",
        "city": "",
        "weather": "",
        "suggestion": "",
        "final_answer": "",
    }

    result = graph.invoke(initial_state)
    print(result["final_answer"])


if __name__ == "__main__":
    main()
