def json_to_llm_text(obj, indent=0):
    """
    Convert a Python dictionary/list JSON object into a compact text format
    optimized for LLM token usage.
    """

    lines = []
    prefix = " " * indent

    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(json_to_llm_text(value, indent + 1))
            else:
                lines.append(f"{prefix}{key}={value}")

    elif isinstance(obj, list):
        values = []
        for item in obj:
            if isinstance(item, (dict, list)):
                values.append(json_to_llm_text(item, indent))
            else:
                values.append(str(item))

        lines.append(f"{prefix}[{','.join(values)}]")

    else:
        lines.append(f"{prefix}{obj}")

    return "\n".join(lines)

def compact_json(obj):
    if isinstance(obj, dict):
        return " ".join(
            f"{k}:{compact_json(v)}"
            for k, v in obj.items()
        )

    if isinstance(obj, list):
        return "[" + ",".join(compact_json(x) for x in obj) + "]"

    if isinstance(obj, str):
        return obj

    return str(obj)