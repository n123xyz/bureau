"""Agent runners with rate limit handling, output streaming, and usage tracking."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    RateLimitEvent,
    ResultMessage,
    SystemMessage,
    TaskProgressMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)

from bureau_mod.ollama_client import OllamaLocalClient
from bureau_mod.config import Config
from bureau_mod.context import extract_json
from bureau_mod.rate_limit import (
    enforce_usage_cap,
    is_rate_limit_error,
    set_rate_limit,
    update_utilization,
    wait_for_rate_limit,
)
import bureau_mod.state as state_mod
from bureau_mod.state import (
    ACTIVE_CLIENTS,
    PAUSE_EVENT,
    STATE,
    emit_event,
    emit_stats,
    emit_task_output,
    emit_agent_stream,
)

log = logging.getLogger("bureau")


def _extract_tokens(result: Any) -> tuple[int, int]:
    """Extract cumulative input/output token counts from a ResultMessage.

    The usage dict may contain cache-specific fields; include those in
    the input total so the numbers actually reflect real usage.
    """
    if not result or not result.usage:
        return 0, 0
    u = result.usage
    in_tok = (u.get("input_tokens", 0)
              + u.get("cache_read_input_tokens", 0)
              + u.get("cache_creation_input_tokens", 0))
    out_tok = u.get("output_tokens", 0)
    return in_tok, out_tok


def _emit_structured_summary(task_id: str, output: dict[str, Any]) -> None:
    """Emit the full structured output content to the task log."""
    try:
        text = json.dumps(output, indent=2)
    except (TypeError, ValueError):
        text = str(output)
    emit_task_output(task_id, "structured output:")
    for line in text.splitlines():
        emit_task_output(task_id, "  " + line)


def _summarize_tool_input(name: str, inp: dict[str, Any]) -> str:
    if name in ("Read", "Write", "Edit", "MultiEdit"):
        return inp.get("file_path", inp.get("path", ""))
    if name == "Bash":
        cmd = inp.get("command", "")
        return cmd[:120] + ("..." if len(cmd) > 120 else "")
    if name == "Glob":
        return inp.get("pattern", "")
    if name == "Grep":
        return f'{inp.get("pattern", "")} in {inp.get("path", ".")}'
    if name in ("WebFetch", "WebSearch"):
        return inp.get("url", inp.get("query", ""))
    if name in ("Agent", "Task"):
        return inp.get("description", "")[:80]
    if name == "TodoWrite":
        todos = inp.get("todos", [])
        return f"{len(todos)} items"
    if name == "Skill":
        return inp.get("skill_name", str(inp)[:80])
    if name == "StructuredOutput":
        # Full content emitted separately via _emit_structured_summary
        return ""
    for v in inp.values():
        if isinstance(v, str) and v:
            return v[:80]
    return ""


async def drain_response(
    client: ClaudeSDKClient,
    timeout: float,
    stall_timeout: float,
    label: str,
    task_id: str | None = None,
    quiet: bool = False,
) -> tuple[ResultMessage | None, str]:
    """Drain all messages from one query with stall detection and output streaming."""
    result: ResultMessage | None = None
    text_parts: list[str] = []
    tool_count = 0
    last_activity = time.monotonic()

    async def watchdog() -> None:
        while True:
            await asyncio.sleep(10)
            idle = time.monotonic() - last_activity
            if idle > stall_timeout:
                log.warning(f"    [{label}] stalled {idle:.0f}s — interrupting")
                try:
                    await client.interrupt()
                except Exception:
                    pass
                return
            if idle > 30:
                log.debug(f"    [{label}] waiting... ({idle:.0f}s idle)")

    watchdog_task = asyncio.create_task(watchdog())

    try:
        async with asyncio.timeout(timeout):
            async for msg in client.receive_messages():
                last_activity = time.monotonic()

                # Check for shutdown between messages
                if STATE.stopping:
                    log.info(f"    [{label}] stopping — interrupting agent")
                    try:
                        await client.interrupt()
                    except Exception:
                        pass
                    break

                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            text_parts.append(block.text)
                            if task_id:
                                emit_agent_stream(task_id, block.text)
                            if not quiet:
                                print(".", end="", flush=True)
                        elif isinstance(block, ThinkingBlock):
                            log.info(f"    [{label}] (thinking...)")
                            if task_id:
                                emit_task_output(task_id, "(thinking...)")
                                if hasattr(block, 'thinking') and block.thinking:
                                    emit_agent_stream(task_id, block.thinking)
                        elif isinstance(block, ToolUseBlock):
                            tool_count += 1
                            summary = _summarize_tool_input(block.name, block.input)
                            line = (f"#{tool_count} {block.name}"
                                    f"{': ' + summary if summary else ''}")
                            log.info(f"    [{label}] {line}")
                            if task_id:
                                emit_task_output(task_id, line)
                                try:
                                    t_json = json.dumps(block.input, indent=2)
                                    emit_agent_stream(task_id, f"\n[Tool: {block.name}]\n{t_json}\n")
                                except Exception:
                                    emit_agent_stream(task_id, f"\n[Tool: {block.name}]\n{block.input}\n")
                        elif isinstance(block, ToolResultBlock):
                            if block.is_error:
                                err_line = f"tool error: {str(block.content)[:200]}"
                                log.warning(f"    [{label}] {err_line}")
                                if task_id:
                                    emit_task_output(task_id, err_line)

                elif isinstance(msg, SystemMessage):
                    log.debug(f"    [{label}] system: {msg.subtype}")

                elif isinstance(msg, TaskProgressMessage):
                    if task_id and msg.last_tool_name:
                        emit_task_output(
                            task_id,
                            f"(progress: {msg.last_tool_name})"
                        )

                elif isinstance(msg, RateLimitEvent):
                    info = msg.rate_limit_info
                    update_utilization(
                        info.utilization,
                        info.rate_limit_type,
                        info.status,
                    )
                    if info.status == "rejected":
                        resets_at = info.resets_at
                        retry_after = None
                        if resets_at:
                            retry_after = max(0, resets_at - time.time())
                        set_rate_limit(retry_after)

                elif isinstance(msg, ResultMessage):
                    result = msg
                    break  # ResultMessage terminates the stream

    except TimeoutError:
        log.warning(f"    [{label}] timeout after {timeout:.0f}s — interrupting")
        try:
            await client.interrupt()
            async with asyncio.timeout(15):
                async for msg in client.receive_messages():
                    if isinstance(msg, ResultMessage):
                        result = msg
                        break
        except Exception:
            pass
    finally:
        watchdog_task.cancel()
        try:
            await watchdog_task
        except asyncio.CancelledError:
            pass

    if not quiet:
        print()
    return result, "".join(text_parts)


async def run_agent(
    *,
    prompt: str,
    system_prompt: str | None = None,
    cwd: str,
    cfg: Config,
    phase_name: str = "",
    task_type: str = "executor",
    label: str = "agent",
    task_id: str | None = None,
    max_retries: int = 3,
) -> str:
    """Spawn an agent with rate limit handling and output streaming."""
    await PAUSE_EVENT.wait()
    await wait_for_rate_limit()
    await enforce_usage_cap(cfg.usage_cap)

    if STATE.stopping:
        raise asyncio.CancelledError("Stopping")

    STATE.total_agents += 1
    agent_num = STATE.total_agents

    # Resolve model/effort/thinking for this phase+task_type
    model, effort, thinking = cfg.resolve(phase_name, task_type)

    env = {}
    actual_model = model
    if model and model.startswith("ollama/"):
        actual_model = model[len("ollama/"):]
        env["ANTHROPIC_BASE_URL"] = cfg.ollama_base_url
        env["ANTHROPIC_AUTH_TOKEN"] = "ollama"

    # Record on the task node
    if task_id:
        task = STATE.get_task(task_id)
        if task:
            task.model = model
            task.effort = effort
            task.thinking = thinking
            task.task_type = task_type
            if not task.prompt:
                task.prompt = prompt
            emit_event("task", task.to_dict())

    is_ollama = model and (model.startswith("ollama/") or not model.startswith("claude-"))
    if is_ollama:
        actual_model = model[len("ollama/"):] if model.startswith("ollama/") else model
        env["ANTHROPIC_BASE_URL"] = cfg.ollama_base_url
        env["ANTHROPIC_AUTH_TOKEN"] = "ollama"
        for attempt in range(max_retries):
            sem = state_mod.AGENT_SEMAPHORE
            assert sem is not None, "AGENT_SEMAPHORE not initialized"
            async with sem:
                await PAUSE_EVENT.wait()
                if STATE.stopping:
                    raise asyncio.CancelledError("Stopping")
                
                log.info(f"  [{label}] agent #{agent_num} starting (ollama:{actual_model}) attempt {attempt+1}")
                if task_id:
                    emit_task_output(task_id, f"ollama agent #{agent_num} starting ({actual_model})")
                
                opts = ClaudeAgentOptions(
                    model=actual_model,
                    effort=effort,
                    env=env,
                    permission_mode=cfg.permission_mode,
                    cwd=cwd,
                    setting_sources=cfg.setting_sources,
                    thinking=cfg.thinking_config(thinking),
                )
                if system_prompt:
                    opts.system_prompt = system_prompt
                
                client = OllamaLocalClient(opts, allow_network=cfg.allow_network)
                if task_id:
                    ACTIVE_CLIENTS[task_id] = client

                t0 = time.monotonic()
                try:
                    await client.connect()
                    # Inject MCP tool documentation into the prompt
                    if hasattr(client, 'mcp_doc') and client.mcp_doc:
                        prompt += client.mcp_doc
                        if task_id:
                            task = STATE.get_task(task_id)
                            if task:
                                task.prompt = prompt
                                emit_event("task", task.to_dict())
                    await client.query(prompt)
                    result, text = await drain_response(
                        client, cfg.timeout, cfg.stall_timeout, label,
                        task_id=task_id,
                    )

                    elapsed = time.monotonic() - t0
                    log.info(f"  [{label}] done {elapsed:.1f}s")
                    if task_id:
                        emit_task_output(task_id, f"done {elapsed:.1f}s")
                    return text
                except Exception as e:
                    log.error(f"  [{label}] ollama failed: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(5 * (attempt + 1))
                        continue
                    return ""
                finally:
                    if task_id:
                        ACTIVE_CLIENTS.pop(task_id, None)
                    await client.disconnect()
        return ""

    for attempt in range(max_retries):
        sem = state_mod.AGENT_SEMAPHORE
        assert sem is not None, "AGENT_SEMAPHORE not initialized"
        async with sem:
            await PAUSE_EVENT.wait()
            if STATE.stopping:
                raise asyncio.CancelledError("Stopping")

            opts = ClaudeAgentOptions(
                model=actual_model,
                effort=effort,
                env=env,
                permission_mode=cfg.permission_mode,
                cwd=cwd,
                setting_sources=cfg.setting_sources,
                thinking=cfg.thinking_config(thinking),
            )
            if system_prompt:
                opts.system_prompt = system_prompt

            log.info(f"  [{label}] agent #{agent_num} starting "
                     f"({model}/{effort}/{thinking}) attempt {attempt+1}")
            log.debug(f"  [{label}] PROMPT:\n{prompt}")
            if task_id:
                emit_task_output(task_id,
                                 f"agent #{agent_num} starting "
                                 f"({model}/{effort}/{thinking})")
            t0 = time.monotonic()

            client = ClaudeSDKClient(opts)
            # Register for interrupt support
            if task_id:
                ACTIVE_CLIENTS[task_id] = client

            try:
                await client.connect()
                await client.query(prompt)
                result, text = await drain_response(
                    client, cfg.timeout, cfg.stall_timeout, label,
                    task_id=task_id,
                )

                cost = (result.total_cost_usd or 0.0) if result else 0.0
                STATE.total_cost += cost
                STATE.cost_in_window += cost
                elapsed = time.monotonic() - t0
                turns = getattr(result, "num_turns", "?") if result else "?"
                in_tok, out_tok = _extract_tokens(result)
                log.debug(f"  [{label}] raw usage: {result.usage if result else None}")
                STATE.total_input_tokens += in_tok
                STATE.total_output_tokens += out_tok
                log.info(f"  [{label}] done {elapsed:.1f}s  turns={turns}  "
                         f"tok={in_tok}in/{out_tok}out  "
                         f"api_cost=${cost:.4f}")
                log.debug(f"  [{label}] RESPONSE ({len(text)} chars):\n{text}")
                emit_stats()

                if task_id:
                    emit_task_output(
                        task_id,
                        f"done {elapsed:.1f}s tok={in_tok}in/{out_tok}out "
                        f"~${cost:.4f}"
                    )

                if result and result.is_error:
                    error_msg = str(result.result)
                    log.error(f"  [{label}] error: {error_msg}")
                    is_rl, retry_after = is_rate_limit_error(Exception(error_msg))
                    if is_rl:
                        set_rate_limit(retry_after)
                        await wait_for_rate_limit()
                        continue

                return text

            except Exception as e:
                is_rl, retry_after = is_rate_limit_error(e)
                if is_rl:
                    log.warning(f"  [{label}] rate limited: {e}")
                    set_rate_limit(retry_after)
                    await wait_for_rate_limit()
                    continue
                else:
                    log.error(f"  [{label}] failed: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(5 * (attempt + 1))
                        continue
                    return ""
            finally:
                if task_id:
                    ACTIVE_CLIENTS.pop(task_id, None)
                try:
                    await client.disconnect()
                except Exception:
                    pass

    return ""


async def run_structured_agent(
    *,
    prompt: str,
    schema: dict[str, Any],
    system_prompt: str | None = None,
    cwd: str,
    cfg: Config,
    phase_name: str = "",
    task_type: str = "executor",
    label: str = "agent",
    task_id: str | None = None,
    max_retries: int = 3,
) -> Any:
    """Run an agent expecting structured JSON output."""
    await PAUSE_EVENT.wait()
    await wait_for_rate_limit()
    await enforce_usage_cap(cfg.usage_cap)

    if STATE.stopping:
        raise asyncio.CancelledError("Stopping")

    STATE.total_agents += 1
    agent_num = STATE.total_agents

    model, effort, thinking = cfg.resolve(phase_name, task_type)

    env = {}
    actual_model = model
    if model and model.startswith("ollama/"):
        actual_model = model[len("ollama/"):]
        env["ANTHROPIC_BASE_URL"] = cfg.ollama_base_url
        env["ANTHROPIC_AUTH_TOKEN"] = "ollama"

    if task_id:
        task = STATE.get_task(task_id)
        if task:
            task.model = model
            task.effort = effort
            task.thinking = thinking
            task.task_type = task_type
            if not task.prompt:
                task.prompt = prompt
            emit_event("task", task.to_dict())

    is_ollama = model and (model.startswith("ollama/") or not model.startswith("claude-"))
    if is_ollama:
        actual_model = model[len("ollama/"):] if model.startswith("ollama/") else model
        env["ANTHROPIC_BASE_URL"] = cfg.ollama_base_url
        env["ANTHROPIC_AUTH_TOKEN"] = "ollama"
        for attempt in range(max_retries):
            sem = state_mod.AGENT_SEMAPHORE
            assert sem is not None, "AGENT_SEMAPHORE not initialized"
            async with sem:
                await PAUSE_EVENT.wait()
                if STATE.stopping:
                    raise asyncio.CancelledError("Stopping")
                
                log.info(f"  [{label}] structured agent #{agent_num} starting (ollama:{actual_model}) attempt {attempt+1}")
                if task_id:
                    emit_task_output(task_id, f"ollama structured agent #{agent_num} starting ({actual_model})")
                
                opts = ClaudeAgentOptions(
                    model=actual_model,
                    effort=effort,
                    env=env,
                    permission_mode=cfg.permission_mode,
                    cwd=cwd,
                    setting_sources=cfg.setting_sources,
                    thinking=cfg.thinking_config(thinking),
                    output_format={"type": "json_schema", "schema": schema},
                )
                if system_prompt:
                    opts.system_prompt = system_prompt
                
                client = OllamaLocalClient(opts, allow_network=cfg.allow_network)
                if task_id:
                    ACTIVE_CLIENTS[task_id] = client

                t0 = time.monotonic()
                try:
                    await client.connect()
                    await client.query(prompt)
                    result, text = await drain_response(
                        client, cfg.timeout, cfg.stall_timeout, label,
                        task_id=task_id,
                    )
                    
                    elapsed = time.monotonic() - t0
                    log.info(f"  [{label}] done {elapsed:.1f}s")
                    
                    if result and result.structured_output is not None:
                        output = result.structured_output
                        if isinstance(output, str):
                            try:
                                output = json.loads(output)
                            except Exception:
                                pass
                        if task_id and isinstance(output, dict):
                            _emit_structured_summary(task_id, output)
                        return output
                    else:
                        try:
                            output = json.loads(text)
                        except Exception:
                            output = extract_json(text)
                        if task_id and isinstance(output, dict):
                            _emit_structured_summary(task_id, output)
                        return output
                except Exception as e:
                    log.error(f"  [{label}] ollama failed: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(5 * (attempt + 1))
                        continue
                    return None
                finally:
                    if task_id:
                        ACTIVE_CLIENTS.pop(task_id, None)
                    await client.disconnect()
        return None

    for attempt in range(max_retries):
        sem = state_mod.AGENT_SEMAPHORE
        assert sem is not None, "AGENT_SEMAPHORE not initialized"
        async with sem:
            await PAUSE_EVENT.wait()
            if STATE.stopping:
                raise asyncio.CancelledError("Stopping")

            opts = ClaudeAgentOptions(
                model=actual_model,
                effort=effort,
                env=env,
                permission_mode=cfg.permission_mode,
                cwd=cwd,
                setting_sources=cfg.setting_sources,
                thinking=cfg.thinking_config(thinking),
                output_format={"type": "json_schema", "schema": schema},
            )
            if system_prompt:
                opts.system_prompt = system_prompt

            log.info(f"  [{label}] agent #{agent_num} starting "
                     f"(structured, {model}/{effort}/{thinking})")
            log.debug(f"  [{label}] PROMPT:\n{prompt}")
            if task_id:
                emit_task_output(task_id,
                                 f"structured agent #{agent_num} "
                                 f"({model}/{effort}/{thinking})")
            t0 = time.monotonic()

            client = ClaudeSDKClient(opts)
            if task_id:
                ACTIVE_CLIENTS[task_id] = client

            try:
                await client.connect()
                await client.query(prompt)
                result, text = await drain_response(
                    client, cfg.timeout, cfg.stall_timeout, label,
                    task_id=task_id,
                )

                cost = (result.total_cost_usd or 0.0) if result else 0.0
                STATE.total_cost += cost
                STATE.cost_in_window += cost
                elapsed = time.monotonic() - t0
                turns = getattr(result, "num_turns", "?") if result else "?"
                in_tok, out_tok = _extract_tokens(result)
                log.debug(f"  [{label}] raw usage: {result.usage if result else None}")
                STATE.total_input_tokens += in_tok
                STATE.total_output_tokens += out_tok
                log.info(f"  [{label}] done {elapsed:.1f}s  turns={turns}  "
                         f"tok={in_tok}in/{out_tok}out  "
                         f"api_cost=${cost:.4f}")
                emit_stats()

                if result and result.is_error:
                    error_msg = str(result.result)
                    log.error(f"  [{label}] error: {error_msg}")
                    is_rl, retry_after = is_rate_limit_error(Exception(error_msg))
                    if is_rl:
                        set_rate_limit(retry_after)
                        await wait_for_rate_limit()
                        continue

                if result and result.structured_output is not None:
                    output = result.structured_output
                    if isinstance(output, str):
                        try:
                            output = json.loads(output)
                        except json.JSONDecodeError:
                            pass
                    log.debug(f"  [{label}] STRUCTURED OUTPUT: "
                              f"{json.dumps(output, indent=2)}")
                    # Emit structured output content for UI
                    if task_id and isinstance(output, dict):
                        _emit_structured_summary(task_id, output)
                    return output

                log.warning(f"  [{label}] no structured_output, fallback")
                log.debug(f"  [{label}] RESPONSE ({len(text)} chars):\n{text}")
                return extract_json(text)

            except Exception as e:
                is_rl, retry_after = is_rate_limit_error(e)
                if is_rl:
                    log.warning(f"  [{label}] rate limited: {e}")
                    set_rate_limit(retry_after)
                    await wait_for_rate_limit()
                    continue
                else:
                    log.error(f"  [{label}] failed: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(5 * (attempt + 1))
                        continue
                    return None
            finally:
                if task_id:
                    ACTIVE_CLIENTS.pop(task_id, None)
                try:
                    await client.disconnect()
                except Exception:
                    pass

    return None
