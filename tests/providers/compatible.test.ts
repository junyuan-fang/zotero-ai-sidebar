import { beforeEach, describe, it, expect, vi } from 'vitest';
import { CompatibleProvider, toChatMessages, normalizeBaseUrl } from '../../src/providers/compatible';
import type { ModelPreset } from '../../src/settings/types';
import type { StreamChunk } from '../../src/providers/types';

const requestLog = vi.hoisted(() => ({
  requests: [] as Array<{ messages?: unknown[]; tools?: unknown[]; baseURL?: string }>,
  baseURL: '' as string,
}));

vi.mock('openai', () => {
  const textStream = async function* () {
    yield { choices: [{ delta: { content: 'Hi' } }] };
    yield { choices: [{ delta: { content: ' there' } }] };
    yield {
      choices: [{ delta: {}, finish_reason: 'stop' }],
      usage: { prompt_tokens: 7, completion_tokens: 2, prompt_tokens_details: { cached_tokens: 0 } },
    };
  };
  class FakeOpenAI {
    toolCallCount = 0;
    constructor(opts: { baseURL?: string }) {
      requestLog.baseURL = opts.baseURL ?? '';
    }
    chat = {
      completions: {
        create: async (params: { stream?: boolean; tools?: unknown[]; messages?: unknown[] }) => {
          requestLog.requests.push({ messages: params.messages, tools: params.tools, baseURL: requestLog.baseURL });
          if (params.tools?.length) {
            this.toolCallCount++;
            return this.toolCallCount === 1
              ? (async function* () {
                  yield { choices: [{ delta: { reasoning_content: 'need a tool' } }] };
                  yield {
                    choices: [
                      {
                        delta: {
                          tool_calls: [
                            { index: 0, id: 'call_1', function: { name: 'zotero_get_full_pdf', arguments: '{}' } },
                          ],
                        },
                      },
                    ],
                  };
                  yield { choices: [{ delta: {}, finish_reason: 'tool_calls' }] };
                })()
              : (async function* () {
                  yield { choices: [{ delta: { content: 'Summary from tool output' } }] };
                  yield {
                    choices: [{ delta: {}, finish_reason: 'stop' }],
                    usage: { prompt_tokens: 10, completion_tokens: 4 },
                  };
                })();
          }
          return textStream();
        },
      },
    };
  }
  return { default: FakeOpenAI };
});

const preset: ModelPreset = {
  id: 'c',
  label: '第三方',
  provider: 'compatible',
  apiKey: 'sk-xxxxxx',
  baseUrl: 'https://oneapi.qunhequnhe.com/v1/',
  model: 'deepseek-chat',
  maxTokens: 1000,
};

describe('CompatibleProvider', () => {
  beforeEach(() => {
    requestLog.requests = [];
    requestLog.baseURL = '';
  });

  it('emits text deltas then usage', async () => {
    const p = new CompatibleProvider();
    const got: StreamChunk[] = [];
    for await (const c of p.stream(
      [{ role: 'user', content: 'hi' }],
      'be helpful',
      preset,
      new AbortController().signal,
    )) {
      got.push(c);
    }
    expect(got).toEqual([
      { type: 'text_delta', text: 'Hi' },
      { type: 'text_delta', text: ' there' },
      { type: 'usage', input: 7, output: 2, cacheRead: 0 },
    ]);
  });

  it('normalizes base URLs', () => {
    // Bare host → default to the conventional /v1 prefix (fixes the 405).
    expect(normalizeBaseUrl('https://oneapi.qunhequnhe.com')).toBe('https://oneapi.qunhequnhe.com/v1');
    expect(normalizeBaseUrl('https://oneapi.qunhequnhe.com/')).toBe('https://oneapi.qunhequnhe.com/v1');
    // Explicit path is respected, trailing slash trimmed.
    expect(normalizeBaseUrl('https://oneapi.qunhequnhe.com/v1')).toBe('https://oneapi.qunhequnhe.com/v1');
    expect(normalizeBaseUrl('https://oneapi.qunhequnhe.com/v1/')).toBe('https://oneapi.qunhequnhe.com/v1');
    expect(normalizeBaseUrl('https://host/api/v3')).toBe('https://host/api/v3');
    expect(normalizeBaseUrl('')).toBe('');
  });

  it('trims the trailing slash from the configured base URL', async () => {
    const p = new CompatibleProvider();
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    for await (const _ of p.stream([{ role: 'user', content: 'hi' }], 's', preset, new AbortController().signal)) {
    }
    expect(requestLog.baseURL).toBe('https://oneapi.qunhequnhe.com/v1');
  });

  it('prepends the system prompt as a system message', async () => {
    const p = new CompatibleProvider();
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    for await (const _ of p.stream([{ role: 'user', content: 'hi' }], 'be helpful', preset, new AbortController().signal)) {
    }
    expect(requestLog.requests[0].messages).toEqual([
      { role: 'system', content: 'be helpful' },
      { role: 'user', content: 'hi' },
    ]);
  });

  it('executes local tools and feeds outputs back to the model', async () => {
    const p = new CompatibleProvider();
    const got: StreamChunk[] = [];

    for await (const c of p.stream(
      [{ role: 'user', content: '总结当前论文' }],
      'be helpful',
      preset,
      new AbortController().signal,
      {
        tools: [
          {
            name: 'zotero_get_full_pdf',
            description: 'Read the current PDF.',
            parameters: { type: 'object', properties: {} },
            execute: async () => ({
              output: '[Paper full text]\ncontent',
              summary: '读取 PDF 全文',
              context: { planMode: 'full_pdf', fullTextChars: 7 },
            }),
          },
        ],
        maxToolIterations: 2,
      },
    )) {
      got.push(c);
    }

    expect(got).toEqual([
      { type: 'thinking_delta', text: 'need a tool' },
      {
        type: 'tool_call',
        name: 'zotero_get_full_pdf',
        status: 'started',
        summary: '调用 Zotero 工具: zotero_get_full_pdf',
      },
      {
        type: 'tool_call',
        name: 'zotero_get_full_pdf',
        status: 'completed',
        summary: '读取 PDF 全文',
        context: { planMode: 'full_pdf', fullTextChars: 7 },
      },
      { type: 'text_delta', text: 'Summary from tool output' },
      { type: 'usage', input: 10, output: 4, cacheRead: 0 },
    ]);
    expect(requestLog.requests).toHaveLength(2);
    expect(requestLog.requests[1].messages).toEqual([
      { role: 'system', content: 'be helpful' },
      { role: 'user', content: '总结当前论文' },
      {
        role: 'assistant',
        content: null,
        tool_calls: [
          { id: 'call_1', type: 'function', function: { name: 'zotero_get_full_pdf', arguments: '{}' } },
        ],
      },
      { role: 'tool', tool_call_id: 'call_1', content: '[Paper full text]\ncontent' },
    ]);
  });

  it('blocks approval-required tools unless YOLO is enabled', async () => {
    const p = new CompatibleProvider();
    const got: StreamChunk[] = [];

    for await (const c of p.stream(
      [{ role: 'user', content: 'write note' }],
      'be helpful',
      preset,
      new AbortController().signal,
      {
        tools: [
          {
            name: 'zotero_get_full_pdf',
            description: 'Pretend write tool.',
            parameters: { type: 'object', properties: {} },
            requiresApproval: true,
            execute: async () => ({ output: 'should not run' }),
          },
        ],
        maxToolIterations: 1,
        permissionMode: 'default',
      },
    )) {
      got.push(c);
    }

    expect(got).toContainEqual({
      type: 'tool_call',
      name: 'zotero_get_full_pdf',
      status: 'error',
      summary: '需要审批: zotero_get_full_pdf',
      context: undefined,
    });
  });

  it('converts screenshot attachments into Chat Completions image parts', () => {
    expect(
      toChatMessages([
        {
          role: 'user',
          content: '分析这张图',
          images: [
            {
              id: 'img-1',
              marker: '[Image #1]',
              name: 'shot.png',
              mediaType: 'image/png',
              dataUrl: 'data:image/png;base64,abc',
              size: 3,
            },
          ],
        },
      ]),
    ).toEqual([
      {
        role: 'user',
        content: [
          { type: 'text', text: '分析这张图' },
          { type: 'text', text: '<image name=[Image #1]>' },
          { type: 'image_url', image_url: { url: 'data:image/png;base64,abc', detail: 'high' } },
          { type: 'text', text: '</image>' },
        ],
      },
    ]);
  });
});
