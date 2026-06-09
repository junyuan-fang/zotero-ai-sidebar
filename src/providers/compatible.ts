import OpenAI from "openai";
import type {
  AgentTool,
  Message,
  Provider,
  ProviderStreamOptions,
  StreamChunk,
  ToolExecutionResult,
} from "./types";
import type { ModelPreset } from "../settings/types";
import { DEFAULT_CONTEXT_POLICY } from "../context/policy";

const COMPATIBLE_REQUEST_TIMEOUT_MS = 120_000;
const COMPATIBLE_MAX_RETRIES = 2;

// Third-party OpenAI-compatible adapter (Chat Completions API).
//
// WHY this exists separately from openai.ts: the `openai` kind talks the
// Responses API (`POST /responses`). Relays/gateways like one-api, new-api or
// a self-hosted proxy (e.g. https://oneapi.qunhequnhe.com) only implement the
// classic `POST /v1/chat/completions`. This adapter targets that endpoint via
// the OpenAI SDK's `chat.completions.create`, so any provider reachable behind
// such a gateway works with just an API key + Base URL + model id.
//
// Design mirrors openai.ts' load-bearing decisions where they still apply:
//   - INVARIANT: `parallel_tool_calls: false`. Tools run strictly sequentially
//     so each tool's output is in the message list before the next call.
//   - `maxToolIterations` is a SAFETY FUSE, not routing logic.
// Hosted tools (web_search / MCP) are NOT offered here: those are
// Responses-API features and gateways don't expose them. Reasoning effort is
// also omitted — there is no portable Chat Completions parameter for it. We do
// surface `reasoning_content` deltas (DeepSeek / many relays emit them) as
// thinking, so the sidebar's collapsible block still renders when available.

interface ChatUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  prompt_tokens_details?: { cached_tokens?: number };
}

interface ChatToolCallDelta {
  index?: number;
  id?: string;
  function?: { name?: string; arguments?: string };
}

interface ChatStreamChoiceDelta {
  content?: string | null;
  // Non-standard but widely emitted by reasoning models behind relays.
  reasoning_content?: string | null;
  reasoning?: string | null;
  tool_calls?: ChatToolCallDelta[];
}

interface ChatStreamChunk {
  choices?: Array<{ delta?: ChatStreamChoiceDelta; finish_reason?: string | null }>;
  usage?: ChatUsage | null;
}

interface AccumulatedToolCall {
  id: string;
  name: string;
  arguments: string;
}

type ChatMessage = {
  role: "system" | "user" | "assistant" | "tool";
  content: string | Array<Record<string, unknown>> | null;
  tool_calls?: Array<{
    id: string;
    type: "function";
    function: { name: string; arguments: string };
  }>;
  tool_call_id?: string;
};

export class CompatibleProvider implements Provider {
  async *stream(
    messages: Message[],
    systemPrompt: string,
    preset: ModelPreset,
    signal: AbortSignal,
    options: ProviderStreamOptions = {},
  ): AsyncIterable<StreamChunk> {
    const client = new OpenAI({
      apiKey: preset.apiKey,
      ...(preset.baseUrl ? { baseURL: normalizeBaseUrl(preset.baseUrl) } : {}),
      timeout: COMPATIBLE_REQUEST_TIMEOUT_MS,
      maxRetries: COMPATIBLE_MAX_RETRIES,
      dangerouslyAllowBrowser: true,
    });

    const tools = options.tools ?? [];
    const toolMap = new Map(tools.map((tool) => [tool.name, tool]));
    const chatTools = tools.length ? tools.map(chatToolSpec) : undefined;

    // `chat` accumulates across iterations: system prompt, the converted
    // conversation, then each assistant tool-call turn and the tool outputs we
    // synthesize from local execution. The model sees the same shape every turn.
    const chat: ChatMessage[] = [
      { role: "system", content: systemPrompt },
      ...toChatMessages(messages),
    ];
    const maxIterations =
      options.maxToolIterations ?? DEFAULT_CONTEXT_POLICY.maxToolIterations;

    for (let iteration = 0; iteration <= maxIterations; iteration++) {
      let stream: AsyncIterable<unknown>;
      try {
        stream = (await client.chat.completions.create(
          {
            model: preset.model,
            messages: chat as never,
            max_tokens: preset.maxTokens,
            stream: true,
            stream_options: { include_usage: true },
            ...(chatTools
              ? {
                  tools: chatTools,
                  tool_choice: "auto",
                  parallel_tool_calls: false,
                }
              : {}),
          } as never,
          { signal },
        )) as unknown as AsyncIterable<unknown>;
      } catch (err) {
        yield { type: "error", message: errMsg(err) };
        return;
      }

      let assistantText = "";
      const toolCalls = new Map<number, AccumulatedToolCall>();
      let usage: ChatUsage | undefined;

      try {
        for await (const raw of stream) {
          if (signal.aborted) throw new Error("Request was aborted.");
          const chunk = raw as ChatStreamChunk;
          if (chunk.usage) usage = chunk.usage;
          const delta = chunk.choices?.[0]?.delta;
          if (!delta) continue;

          if (delta.content) {
            assistantText += delta.content;
            yield { type: "text_delta", text: delta.content };
          }
          const thinking = delta.reasoning_content ?? delta.reasoning;
          if (thinking) yield { type: "thinking_delta", text: thinking };

          for (const part of delta.tool_calls ?? []) {
            const index = part.index ?? 0;
            const existing =
              toolCalls.get(index) ?? { id: "", name: "", arguments: "" };
            if (part.id) existing.id = part.id;
            if (part.function?.name) existing.name = part.function.name;
            if (part.function?.arguments)
              existing.arguments += part.function.arguments;
            toolCalls.set(index, existing);
          }
        }
      } catch (err) {
        yield { type: "error", message: errMsg(err) };
        return;
      }

      const calls = [...toolCalls.values()].filter((c) => c.name);

      // Natural exit: model produced text-only output. No tool calls ⇒ done.
      if (calls.length === 0) {
        if (usage) yield usageChunk(usage);
        return;
      }

      // Replay the assistant tool-call turn into `chat` BEFORE running tools.
      // Chat Completions requires every `tool` message to be preceded by an
      // assistant message carrying the matching `tool_calls[].id`.
      chat.push({
        role: "assistant",
        content: assistantText || null,
        tool_calls: calls.map((call) => ({
          id: call.id,
          type: "function",
          function: { name: call.name, arguments: call.arguments },
        })),
      });

      for (const call of calls) {
        yield {
          type: "tool_call",
          name: call.name,
          status: "started",
          summary: `调用 Zotero 工具: ${call.name}`,
        };
        const result = await executeToolCall(
          call,
          toolMap,
          signal,
          options.permissionMode ?? "default",
        );
        yield {
          type: "tool_call",
          name: call.name,
          status: result.status,
          summary: result.result.summary,
          context: result.result.context,
        };
        chat.push({
          role: "tool",
          tool_call_id: call.id,
          content: result.result.output,
        });
      }
    }

    // Safety-fuse blew. INVARIANT: never silently truncate; surface as error
    // so the user sees the loop bound was the limiter, not the model.
    yield {
      type: "error",
      message:
        "Tool loop stopped because the model exceeded the local tool iteration limit.",
    };
  }
}

async function executeToolCall(
  call: AccumulatedToolCall,
  toolMap: Map<string, AgentTool>,
  signal: AbortSignal,
  permissionMode: "default" | "yolo",
): Promise<{ status: "completed" | "error"; result: ToolExecutionResult }> {
  if (signal.aborted) {
    return {
      status: "error",
      result: { output: "Tool call aborted.", summary: "工具调用已停止" },
    };
  }

  const tool = toolMap.get(call.name);
  if (!tool) {
    return {
      status: "error",
      result: {
        output: `Unknown local tool: ${call.name}`,
        summary: `未知工具 ${call.name}`,
      },
    };
  }

  // INVARIANT: write tools gate through requiresApproval. In default mode they
  // refuse; only YOLO mode bypasses. Mirrors openai.ts.
  // REF: CLAUDE.md non-negotiable "No hidden Zotero writes".
  if (tool.requiresApproval && permissionMode !== "yolo") {
    return {
      status: "error",
      result: {
        output: `Local tool ${call.name} requires approval. Enable YOLO mode to run it without approval.`,
        summary: `需要审批: ${call.name}`,
      },
    };
  }

  let args: unknown;
  try {
    args = call.arguments ? JSON.parse(call.arguments) : {};
  } catch {
    return {
      status: "error",
      result: {
        output: `Invalid JSON arguments for local tool: ${call.name}`,
        summary: `工具参数 JSON 无效: ${call.name}`,
      },
    };
  }

  try {
    return { status: "completed", result: await tool.execute(args) };
  } catch (err) {
    return {
      status: "error",
      result: {
        output: errMsg(err),
        summary: `工具执行失败: ${call.name}`,
      },
    };
  }
}

function chatToolSpec(tool: AgentTool): Record<string, unknown> {
  return {
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
    },
  };
}

export function toChatMessages(messages: Message[]): ChatMessage[] {
  return messages.map((message) => {
    if (!message.images?.length) {
      return { role: message.role, content: message.content };
    }

    const content: Array<Record<string, unknown>> = [];
    if (message.content) {
      content.push({ type: "text", text: message.content });
    }
    message.images.forEach((image, index) => {
      const label = image.marker ?? `[Image #${index + 1}]`;
      content.push({ type: "text", text: `<image name=${label}>` });
      content.push({
        type: "image_url",
        image_url: { url: image.dataUrl, detail: "high" },
      });
      content.push({ type: "text", text: "</image>" });
    });
    return { role: message.role, content };
  });
}

// Gateways are usually configured WITH the `/v1` suffix in the pasted URL, but
// some users paste just the host. The OpenAI SDK appends `/chat/completions`
// to whatever baseURL it gets, so we leave the path untouched and only trim a
// trailing slash to avoid `//chat/completions`.
function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.trim().replace(/\/+$/, "");
}

function usageChunk(usage: ChatUsage): StreamChunk {
  return {
    type: "usage",
    input: usage.prompt_tokens ?? 0,
    output: usage.completion_tokens ?? 0,
    cacheRead: usage.prompt_tokens_details?.cached_tokens ?? 0,
  };
}

function errMsg(err: unknown): string {
  const message = err instanceof Error ? err.message : String(err);
  const status =
    typeof err === "object" && err != null && "status" in err
      ? Number((err as { status?: unknown }).status)
      : statusFromMessage(message);
  if (status === 429) {
    return "模型服务限流了（HTTP 429）。插件已自动重试但仍失败；稍等一会儿再点「重试」，或换一个模型/Base URL。";
  }
  if (status && status >= 500 && status < 600) {
    return `模型网关临时失败（HTTP ${status}）。插件已自动重试但仍失败；请稍后点「重试」或检查 Base URL 是否支持 /v1/chat/completions。`;
  }
  return message;
}

function statusFromMessage(message: string): number | null {
  const match =
    /\b(?:HTTP\s*)?([1-5][0-9]{2})\b/.exec(message) ??
    /\b([1-5][0-9]{2})\s+status code\b/i.exec(message);
  return match ? Number(match[1]) : null;
}
