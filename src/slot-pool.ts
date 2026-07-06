import {
  LlamaChat,
  type LlamaContext,
  type LlamaContextSequence,
  type ChatWrapper,
  type ChatHistoryItem,
} from "node-llama-cpp";

export interface Slot {
  id: number;
  sequence: LlamaContextSequence;
  llamaChat: LlamaChat;
  busy: boolean;
  /** Chat history as of the last completed generation (for prefix matching) */
  chatHistory: ChatHistoryItem[];
  /** `response.lastEvaluation.contextWindow` from the last generation on this sequence */
  lastContextWindow: ChatHistoryItem[] | null;
  contextShiftMetadata: any;
  lastUsedAt: number;
}

/**
 * Pool of generation slots, one per context sequence.
 *
 * `acquire()` prefers the idle slot whose cached history shares the longest prefix with
 * the incoming request (KV-cache reuse for multi-turn agent loops), falling back to the
 * least-recently-used idle slot. When all slots are busy, callers wait FIFO.
 */
export class SlotPool {
  private readonly slots: Slot[] = [];
  private readonly waiters: Array<{
    resolve: (slot: Slot) => void;
    reject: (err: Error) => void;
  }> = [];
  private disposed = false;

  constructor(context: LlamaContext, chatWrapper: ChatWrapper) {
    for (let i = 0; i < context.totalSequences; i++) {
      const sequence = context.getSequence();
      this.slots.push({
        id: i,
        sequence,
        llamaChat: new LlamaChat({ contextSequence: sequence, chatWrapper }),
        busy: false,
        chatHistory: [],
        lastContextWindow: null,
        contextShiftMetadata: null,
        lastUsedAt: 0,
      });
    }
  }

  get size() {
    return this.slots.length;
  }

  async acquire(history: ChatHistoryItem[], signal?: AbortSignal): Promise<Slot> {
    if (this.disposed) throw new Error("Model has been disposed");
    if (signal?.aborted) throw signal.reason ?? new Error("Aborted");

    const idle = this.slots.filter((slot) => !slot.busy);
    if (idle.length > 0) {
      let best = idle[0]!;
      let bestScore = -1;
      for (const slot of idle) {
        const score = sharedPrefixLength(slot.chatHistory, history);
        if (score > bestScore || (score === bestScore && slot.lastUsedAt < best.lastUsedAt)) {
          best = slot;
          bestScore = score;
        }
      }
      best.busy = true;
      best.lastUsedAt = Date.now();
      return best;
    }

    return new Promise<Slot>((resolve, reject) => {
      const waiter = { resolve, reject };
      this.waiters.push(waiter);
      signal?.addEventListener(
        "abort",
        () => {
          const index = this.waiters.indexOf(waiter);
          if (index >= 0) {
            this.waiters.splice(index, 1);
            reject(signal.reason ?? new Error("Aborted while waiting for a generation slot"));
          }
        },
        { once: true }
      );
    });
  }

  release(slot: Slot) {
    const waiter = this.waiters.shift();
    if (waiter != null) {
      slot.lastUsedAt = Date.now();
      waiter.resolve(slot); // stays busy, handed over directly
      return;
    }

    slot.busy = false;
  }

  async dispose() {
    this.disposed = true;
    for (const waiter of this.waiters.splice(0))
      waiter.reject(new Error("Model has been disposed"));
    for (const slot of this.slots) await slot.llamaChat.dispose();
  }
}

export function sharedPrefixLength(a: ChatHistoryItem[], b: ChatHistoryItem[]): number {
  const max = Math.min(a.length, b.length);
  let count = 0;
  for (let i = 0; i < max; i++) {
    if (JSON.stringify(a[i]) !== JSON.stringify(b[i])) break;
    count++;
  }
  return count;
}
