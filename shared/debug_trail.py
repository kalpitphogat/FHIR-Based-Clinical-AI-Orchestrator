from shared.fhir_hook import extract_fhir_context


# ---------- helpers ----------

def _get_agent_name(ctx):
    try:
        if hasattr(ctx, "agent") and ctx.agent:
            return ctx.agent.name
        if hasattr(ctx, "agent_name"):
            return ctx.agent_name
    except Exception:
        pass
    return "unknown_agent"


def _get_context(args, kwargs):
    return kwargs.get("context") or kwargs.get("tool_context") or (args[0] if args else None)


# ---------- MODEL CALLBACKS ----------

def log_before_model(*args, **kwargs):
    ctx = _get_context(args, kwargs)
    agent = _get_agent_name(ctx)

    print(f"\n================ MODEL CALL [{agent}] ================")

    try:
        msg = getattr(ctx, "message", None)
        if msg:
            print("User:", getattr(msg, "parts", msg))
    except Exception:
        pass

    print("=====================================================\n")


def log_after_model(*args, **kwargs):
    ctx = _get_context(args, kwargs)
    agent = _get_agent_name(ctx)

    response = kwargs.get("response") or (args[0] if args else None)

    # 🔥 Skip useless intermediate steps
    if not response:
        return

    # Skip tool routing noise
    if isinstance(response, dict) and "tool_calls" in response:
        return

    print(f"\n=========== FINAL RESPONSE [{agent}] ===========")
    print(response)
    print("===============================================\n")


# ---------- TOOL CALLBACKS ----------

def log_tool_call(*args, **kwargs):
    ctx = kwargs.get("tool_context")
    agent = _get_agent_name(ctx)

    tool = kwargs.get("tool")
    tool_name = getattr(tool, "name", str(tool))

    print(f"\n>>>>>>>>>>>> TOOL CALLED [{agent}] >>>>>>>>>>>>>>")
    print("Tool:", tool_name)

    args_passed = kwargs.get("args")
    if args_passed:
        print("Args:", args_passed)

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")


def log_tool_result(*args, **kwargs):
    ctx = kwargs.get("tool_context")
    agent = _get_agent_name(ctx)

    tool = kwargs.get("tool")
    tool_name = getattr(tool, "name", str(tool))

    result = kwargs.get("tool_response")

    print(f"\n<<<<<<<<<<<< TOOL RESULT [{agent}] <<<<<<<<<<<<<<")
    print("Tool:", tool_name)

    # Clean output (avoid huge dumps)
    if isinstance(result, dict):
        if "observations" in result:
            print(f"Observations count: {result.get('count')}")
        elif "conditions" in result:
            print(f"Conditions count: {result.get('count')}")
        elif "medications" in result:
            print(f"Medications count: {result.get('count')}")
        else:
            print(result)
    else:
        print(result)

    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")


