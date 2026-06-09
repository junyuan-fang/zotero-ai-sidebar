// 'compatible' = any third-party OpenAI-compatible gateway (one-api / new-api /
// self-hosted relay) reached via the classic Chat Completions endpoint. WHY a
// separate kind from 'openai': the 'openai' adapter talks the Responses API
// (`POST /responses`), which relays generally do NOT implement — they expose
// `POST /v1/chat/completions`. See providers/compatible.ts.
export type ProviderKind = 'anthropic' | 'openai' | 'compatible';
export type ReasoningEffort = 'none' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh';
export type ReasoningSummary = 'auto' | 'concise' | 'detailed' | 'none';
export type AgentPermissionMode = 'default' | 'yolo';

export interface ModelPreset {
  id: string;
  label: string;
  provider: ProviderKind;
  apiKey: string;
  baseUrl: string;
  // Currently-active model — what providers actually send to the API. Stays
  // a single string so provider adapters don't need to change.
  model: string;
  // Available models for this preset. The composer-footer switcher lets the
  // user pick one and writes the choice back to `model`. Persisted in prefs
  // so the selection is sticky across sessions.
  // GOTCHA: optional for back-compat with legacy presets that only had
  // `model`. `normalizePreset` in storage.ts back-fills this on load.
  models?: string[];
  maxTokens: number;
  extras?: {
    reasoningEffort?: ReasoningEffort;
    reasoningSummary?: ReasoningSummary;
    agentPermissionMode?: AgentPermissionMode;
    omitMaxOutputTokens?: boolean;
    [key: string]: unknown;
  };
}

export const DEFAULT_BASE_URLS: Record<ProviderKind, string> = {
  anthropic: '',
  openai: '',
  // Third-party gateways have no canonical default; the user must paste one
  // (e.g. https://oneapi.qunhequnhe.com). Left blank so the field reads empty.
  compatible: '',
};

export const DEFAULT_MODELS: Record<ProviderKind, string> = {
  anthropic: '',
  openai: '',
  compatible: '',
};

export const MODEL_SUGGESTIONS: Record<ProviderKind, string[]> = {
  anthropic: [],
  openai: ['gpt-5.5', 'gpt-5.4', 'gpt-5.4-mini', 'gpt-5.3-codex', 'gpt-5.2'],
  compatible: [],
};

export const DEFAULT_REASONING_EFFORT: ReasoningEffort = 'xhigh';
export const DEFAULT_REASONING_SUMMARY: ReasoningSummary = 'concise';

export const REASONING_EFFORT_OPTIONS: Array<[ReasoningEffort, string]> = [
  ['low', 'Low - 快速，较少推理'],
  ['medium', 'Medium - 默认平衡'],
  ['high', 'High - 更强推理'],
  ['xhigh', 'Extra high - 最强推理'],
];

export const REASONING_SUMMARY_OPTIONS: Array<[ReasoningSummary, string]> = [
  ['concise', 'Concise - 简短显示思考摘要'],
  ['detailed', 'Detailed - 更详细的思考摘要'],
  ['auto', 'Auto - 由模型决定'],
  ['none', 'None - 不显示思考'],
];

export function defaultPresetLabel(provider: ProviderKind): string {
  switch (provider) {
    case 'anthropic':
      return 'Claude';
    case 'openai':
      return 'GPT';
    case 'compatible':
      return '第三方';
  }
}

export function newPreset(provider: ProviderKind): ModelPreset {
  const defaultModel = DEFAULT_MODELS[provider];
  return {
    id: crypto.randomUUID(),
    label: defaultPresetLabel(provider),
    provider,
    apiKey: '',
    baseUrl: DEFAULT_BASE_URLS[provider],
    model: defaultModel,
    models: defaultModel ? [defaultModel] : [],
    maxTokens: 8192,
    extras: provider === 'openai'
      ? {
          reasoningEffort: DEFAULT_REASONING_EFFORT,
          reasoningSummary: DEFAULT_REASONING_SUMMARY,
        }
      : undefined,
  };
}

export type TranslateThinking = 'low' | 'medium' | 'high' | 'xhigh';
export type TranslateContextLevel = 'none' | 'paragraph' | 'page';
export type TranslateOverlayPosition = 'above' | 'below';
export type TranslateTriggerMode = 'single' | 'double';
export type TranslateOverlaySize = 'compact' | 'adaptive';

export interface TranslateSettings {
  enabled: boolean;
  presetId: string;
  model: string;
  thinking: TranslateThinking;
  ctxLevel: TranslateContextLevel;
  overlayPosition: TranslateOverlayPosition;
  overlaySize: TranslateOverlaySize;
  triggerMode: TranslateTriggerMode;
  prevSentenceKey: string;
  nextSentenceKey: string;
}

export const DEFAULT_TRANSLATE_SETTINGS: TranslateSettings = {
  enabled: false,
  presetId: '',
  model: '',
  thinking: 'low',
  ctxLevel: 'none',
  overlayPosition: 'above',
  overlaySize: 'compact',
  triggerMode: 'single',
  prevSentenceKey: 'Shift+Enter',
  nextSentenceKey: 'Enter',
};
