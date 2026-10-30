from agent_graph import agent

graph = agent.get_graph()

# Mermaid text you can paste anywhere
mermaid = graph.draw_mermaid()
with open("workflow.mmd", "w", encoding="utf-8") as f:
    f.write(mermaid)

# Optional PNG image (this uses the mermaid.ink website, so it needs internet)
with open("workflow.png", "wb") as f:
    f.write(graph.draw_mermaid_png())

print(mermaid)