import asyncio
import glob
import json
import os
import subprocess
import time
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)

class LinuxSandbox:
    def __init__(
        self, 
        workspace_dir: str, 
        full_disk_read: bool = False,
        readable_roots: list[str] = None,
        writable_roots: list[str] = None,
        read_only_subpaths: list[str] = None,
        unreadable_paths: list[str] = None
    ):
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.full_disk_read = full_disk_read
        self.readable_roots = readable_roots or ["/bin", "/sbin", "/usr", "/etc", "/lib", "/lib64"]
        self.writable_roots = writable_roots or [self.workspace_dir]
        self.read_only_subpaths = read_only_subpaths or [os.path.join(self.workspace_dir, ".git")]
        self.unreadable_paths = unreadable_paths or []

    def execute(self, command: str, allow_network: bool = False):
        try:
            bwrap_args = [
                "bwrap",
                "--new-session",
                "--die-with-parent",
                "--unshare-user",
                "--unshare-pid",
                "--unshare-ipc",
                "--unshare-uts",
            ]
            if not allow_network:
                bwrap_args.append("--unshare-net")
                
            bwrap_args.extend(["--proc", "/proc"])
            
            if self.full_disk_read:
                bwrap_args.extend(["--ro-bind", "/", "/", "--dev", "/dev"])
            else:
                bwrap_args.extend(["--tmpfs", "/", "--dev", "/dev"])
                for root in set(self.readable_roots):
                    if os.path.exists(root):
                        bwrap_args.extend(["--ro-bind", root, root])

            # Apply complete unreadable masking first
            for unreadable in set(self.unreadable_paths):
                # Ensure the parent exists in BWrap before trying to mount tmpfs over it
                bwrap_args.extend(["--perms", "000", "--tmpfs", unreadable])

            # Layer explicit writable roots
            for writable in set(self.writable_roots):
                if os.path.exists(writable):
                    bwrap_args.extend(["--bind", writable, writable])

            # Re-apply read-only protections for subpaths under those writable roots
            for ro_sub in set(self.read_only_subpaths):
                if os.path.exists(ro_sub):
                    bwrap_args.extend(["--ro-bind", ro_sub, ro_sub])

            bwrap_args.extend(["--chdir", self.workspace_dir])
            bwrap_args.extend(["--", "sh", "-c", command])

            result = subprocess.run(
                bwrap_args,
                cwd=self.workspace_dir,
                capture_output=True,
                text=True,
                timeout=120
            )
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            return output
        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 120s"
        except FileNotFoundError:
            # Fallback if bwrap isn't directly available (e.g., testing locally without it)
            result = subprocess.run(command, shell=True, cwd=self.workspace_dir, capture_output=True, text=True, timeout=120)
            return result.stdout + result.stderr

def bash(command: str) -> str:
    """Execute a bash command in the workspace directory.
    
    Args:
        command: The bash command string to execute. (e.g. 'ls -la')
    """
    # This is a placeholder since the actual execution intercepts it within the client's loop using the sandbox
    pass

def read(file_path: str) -> str:
    """Read the contents of a file on the file system.
    
    Args:
        file_path: The absolute or relative path to the file.
    """
    try:
        return open(file_path, "r", encoding="utf-8").read()
    except Exception as e:
        return f"Error: {e}"

def write(file_path: str, content: str) -> str:
    """Write exact content to a file, creating any parent directories if needed. Overwrites the file completely.
    
    Args:
        file_path: The absolute or relative path to the file.
        content: The exact string content to write to the file.
    """
    try:
        os.makedirs(os.path.dirname(os.path.realpath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} bytes to {file_path}"
    except Exception as e:
        return f"Error: {e}"

def multi_edit(file_path: str, old_string: str, new_string: str) -> str:
    """Edit an existing file by replacing an exact string match with a new string.
    
    Args:
        file_path: The absolute or relative path to the file.
        old_string: The exact existing string in the file to replace. Be very careful to include indentation matching the file exactly!
        new_string: The new string to replace it with.
    """
    try:
        content = open(file_path, "r", encoding="utf-8").read()
        if old_string not in content:
            return "Error: old_string not found in file."
        content = content.replace(old_string, new_string)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return "Successfully edited file."
    except Exception as e:
        return f"Error: {e}"

def glob_files(pattern: str) -> str:
    """Find files matching a glob pattern relative to the current directory.
    
    Args:
        pattern: The glob pattern, e.g. '**/*.py' or 'src/*.js'
    """
    try:
        files = glob.glob(pattern, recursive=True)
        return "\n".join(files[:100])
    except Exception as e:
        return f"Error: {e}"


class OllamaLocalClient:
    def __init__(self, opts: ClaudeAgentOptions, allow_network: bool = False):
        self.opts = opts
        self.model = opts.model
        self.cwd = str(opts.cwd or os.getcwd())
        self.allow_network = allow_network
        
        # Store initial messages
        self.messages = []
        if getattr(opts, "system_prompt", None):
            self.messages.append({"role": "system", "content": opts.system_prompt})
            
        import ollama
        base_url = opts.env.get("ANTHROPIC_BASE_URL") if getattr(opts, "env", None) else None
        self.client = ollama.AsyncClient(host=base_url)
        
        # Tools provided natively to Ollama
        self.tools = [bash, read, write, multi_edit, glob_files]
        self.sandbox = LinuxSandbox(workspace_dir=self.cwd)
        self._output_format = getattr(opts, "output_format", None)
        self._queue = asyncio.Queue()
        self._running = False
        self._task = None
        
    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def interrupt(self):
        self._running = False

    async def query(self, prompt: str):
        self.messages.append({"role": "user", "content": prompt})
        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def receive_messages(self):
        while True:
            msg = await self._queue.get()
            if msg is None:
                break
            yield msg

    async def _emit(self, msg):
        await self._queue.put(msg)

    async def _run_loop(self):
        try:
            # Change directory to the workspace
            original_cwd = os.getcwd()
            os.chdir(self.cwd)
            
            while self._running:
                kwargs = {
                    "model": self.model,
                    "messages": self.messages,
                    "stream": True,
                }
                
                is_structured = self._output_format and self._output_format.get("type") == "json_schema"
                if is_structured:
                    kwargs["format"] = self._output_format["schema"]
                else:
                    kwargs["tools"] = self.tools

                stream = await self.client.chat(**kwargs)
                
                thinking = ""
                content = ""
                tool_calls = []
                # Buffer for UI events
                blocks = []
                
                async for chunk in stream:
                    if not self._running:
                        break
                    cm = chunk.message
                    if getattr(cm, 'thinking', None):
                        thinking += cm.thinking
                        if len(blocks) == 0 or not isinstance(blocks[-1], ThinkingBlock):
                            blocks.append(ThinkingBlock(thinking=cm.thinking, signature=""))
                        else:
                            blocks[-1].thinking += cm.thinking
                            
                    if getattr(cm, 'content', None):
                        content += cm.content
                        if len(blocks) == 0 or not isinstance(blocks[-1], TextBlock):
                            blocks.append(TextBlock(text=cm.content))
                        else:
                            blocks[-1].text += cm.content
                            
                    if getattr(cm, 'tool_calls', None):
                        for tc in cm.tool_calls:
                            # Map native Ollama tool call to ToolUseBlock
                            tool_id = "tc_" + str(len(tool_calls))
                            tb = ToolUseBlock(
                                id=tool_id,
                                name=tc.function.name,
                                input=tc.function.arguments
                            )
                            blocks.append(tb)
                            tool_calls.append(tc)

                # Emit the assistant response formatted properly
                if blocks:
                    await self._emit(AssistantMessage(content=blocks, model=self.model))
                
                # Append internally for message history
                assistant_msg = {"role": "assistant"}
                if content:
                    assistant_msg["content"] = content
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                if not content and not tool_calls:
                    # prevent API error for empty model turn 
                    assistant_msg["content"] = "ok" 
                
                self.messages.append(assistant_msg)
                
                if not tool_calls:
                    # End of agent loop because no further tools
                    usage = {"input_tokens": 0, "output_tokens": 0}
                    res = ResultMessage(
                        subtype="structured" if is_structured else "text",
                        duration_ms=0,
                        duration_api_ms=0,
                        num_turns=1,
                        session_id="ollama",
                        result=content,
                        usage=usage,
                        is_error=False,
                        total_cost_usd=0.0
                    )
                    
                    if is_structured:
                        import json
                        try:
                            res.structured_output = json.loads(content)
                        except Exception:
                            res.structured_output = None
                    
                    await self._emit(res)
                    break
                
                # Execute tools locally
                for index, tc in enumerate(tool_calls):
                    name = tc.function.name
                    args = tc.function.arguments
                    result_str = "Error: unknown tool"
                    
                    # Ensure safe local execution
                    try:
                        if name == "bash":
                            # Route bash command through the sandbox
                            result_str = self.sandbox.execute(
                                command=args.get("command", ""),
                                allow_network=self.allow_network
                            )
                        elif name == "read":
                            result_str = read(args.get("file_path", ""))
                        elif name == "write":
                            result_str = write(args.get("file_path", ""), args.get("content", ""))
                        elif name == "multi_edit":
                            result_str = multi_edit(args.get("file_path", ""), args.get("old_string", ""), args.get("new_string", ""))
                        elif name == "glob_files":
                            result_str = glob_files(args.get("pattern", ""))
                    except Exception as ex:
                        result_str = f"Error executing tool: {ex}"
                        
                    # Standard ollama role for tools is "tool"
                    self.messages.append({
                        "role": "tool",
                        "content": str(result_str)[:10000] # trunc
                    })
                    
                    # Yield tool results visually for web dashboard
                    await self._emit(AssistantMessage(content=[ToolResultBlock(
                        tool_use_id="tc_" + str(index), 
                        is_error=False,
                        content=str(result_str)[:1000]
                    )], model=self.model))
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
            await self._emit(ResultMessage(
                subtype="error", 
                duration_ms=0, 
                duration_api_ms=0, 
                num_turns=1, 
                session_id="ollama", 
                result=str(e), 
                usage={"input_tokens":0,"output_tokens":0}, 
                is_error=True
            ))
            
        finally:
            os.chdir(original_cwd)
            await self._queue.put(None)
