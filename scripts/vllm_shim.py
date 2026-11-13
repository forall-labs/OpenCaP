"""Minimal OpenAI-compatible chat server over vLLM's in-process engine.

Why this exists: on Colab (L4, vLLM 0.28.0) `vllm serve` initialises the engine
correctly -- weights load, KV cache sizes, CUDA graphs capture -- and then never
binds its port. Reproduced twice, stalling at exactly the same point with the
process alive and idle. The engine is fine; the HTTP layer is not.

So we keep the engine and replace the HTTP layer with ~120 lines of stdlib.
CaP-X only needs POST /v1/chat/completions, so that is all we serve.

Requests are micro-batched: `vllm.LLM.chat()` is a blocking batch call and is
not thread-safe, so a single worker thread drains a queue and submits whatever
has arrived. Serialising instead would cap throughput at one request at a time,
which matters because CaP-X runs several sim workers concurrently.

    python3 scripts/vllm_shim.py --model Qwen/Qwen3-1.7B --port 8110
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Batch composition depends on arrival timing, and vLLM numerics are
# batch-size dependent -- so micro-batching makes greedy decoding
# non-reproducible run to run. Overridable so determinism can be tested and,
# if needed, bought at the cost of throughput.
BATCH_MAX = int(__import__("os").environ.get("OPENCAP_BATCH_MAX", "8"))
BATCH_WAIT_S = 0.05

#: Hybrid-reasoning models (Qwen3, Qwen3.5, ...) emit their chain of thought
#: inline. vLLM's real server separates it via --reasoning-parser; we must do
#: the same, because CaP-X writes the response straight into code.py and a
#: <think> block there is a SyntaxError, not a wrong answer.
_THINK = re.compile(r"<think>.*?</think>\s*", re.S)
_OPEN_THINK = re.compile(r"<think>.*\Z", re.S)  # truncated: never closed


def split_reasoning(text: str) -> tuple[str, str]:
    """-> (content, reasoning). Handles the truncated case explicitly.

    A model that spends its whole token budget thinking returns an *unclosed*
    <think>, leaving no content at all. Silently passing that through looks
    exactly like a model that cannot code -- which is how this was nearly
    mistaken for a capability result.
    """
    reasoning = "".join(m.group(0) for m in _THINK.finditer(text))
    content = _THINK.sub("", text)
    trunc = _OPEN_THINK.search(content)
    if trunc:
        reasoning += trunc.group(0)
        content = content[: trunc.start()]
    return content.strip(), reasoning.strip()

_jobs: "queue.Queue[dict]" = queue.Queue()
_llm = None
_model_name = ""
#: Set by scripts/run_sampled_arm.sh. None at temperature 0, where it is
#: inert anyway; an integer per draw for the sampled arm.
_SAMPLE_SEED = (int(os.environ["OPENCAP_SAMPLE_INDEX"])
                if os.environ.get("OPENCAP_SAMPLE_INDEX") else None)

_enable_thinking = False
_ready = threading.Event()

#: Sidecar of per-request generation stats. CaP-X does not record token
#: logprobs, and they cannot be recovered after the fact -- recomputing them
#: means re-running the whole grid. So the shim writes them as it serves and
#: the join happens later, keyed by request id.
_sidecar_path: str | None = None
_sidecar_lock = threading.Lock()


def _log_stats(rec: dict) -> None:
    if not _sidecar_path:
        return
    with _sidecar_lock:
        with open(_sidecar_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


def _worker() -> None:
    from vllm import SamplingParams

    while True:
        first = _jobs.get()
        batch = [first]
        deadline = time.time() + BATCH_WAIT_S
        while len(batch) < BATCH_MAX and time.time() < deadline:
            try:
                batch.append(_jobs.get(timeout=max(0, deadline - time.time())))
            except queue.Empty:
                break

        convos = [j["messages"] for j in batch]
        chat_kw = {}
        if not _enable_thinking:
            # Uniform across the whole ladder: no per-model tuning, and with
            # MAX_NEW_TOKENS=2048 a thinking model spends the entire budget
            # reasoning and emits no code at all.
            chat_kw["chat_template_kwargs"] = {"enable_thinking": False}
        params = [
            SamplingParams(
                temperature=j.get("temperature", 0.0),
                max_tokens=j.get("max_tokens", 2048),
                seed=j.get("seed"),
            )
            for j in batch
        ]
        try:
            outs = _llm.chat(convos, params, use_tqdm=False, **chat_kw)
            for job, out in zip(batch, outs):
                content, reasoning = split_reasoning(out.outputs[0].text)
                n_out = len(out.outputs[0].token_ids or [])
                cum = out.outputs[0].cumulative_logprob
                # Mean per-token logprob is the comparable quantity: the
                # cumulative value scales with length, so long generations
                # would look "less confident" purely for being long.
                mean_lp = (cum / n_out) if (cum is not None and n_out) else None
                # ARTIFACT #15. The response below used to hardcode
                # finish_reason="stop" for every request. vLLM knows the truth
                # -- "length" when the generation hit max_tokens -- and the
                # shim threw it away, so a program cut mid-statement reached
                # the corpus indistinguishable from one the model finished.
                # 33.4% of requests from the <=2B band were cut and 0% from
                # >=12B: the same axis as every by-band result in this project.
                #
                # A field that always says "stop" is a check that cannot fail.
                # Derived as a FALLBACK rather than assumed, so an engine that
                # stops naming it does not silently read as "stop".
                fr = getattr(out.outputs[0], "finish_reason", None)
                if not fr:
                    fr = "length" if n_out >= j.get("max_tokens", 2048) else "stop"
                job["result"] = {
                    "text": content,
                    "reasoning": reasoning,
                    "prompt_tokens": len(out.prompt_token_ids or []),
                    "completion_tokens": n_out,
                    "finish_reason": fr,
                    "cumulative_logprob": cum,
                    "logprob_mean": mean_lp,
                }
                job["done"].set()
        except Exception as exc:                      # noqa: BLE001
            for job in batch:
                job["error"] = repr(exc)
                job["done"].set()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # keep the pod log readable
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # readiness probes hang up on 503; not worth a traceback

    def do_GET(self):
        if self.path.rstrip("/").endswith("/v1/models"):
            if not _ready.is_set():
                self._send(503, {"error": "loading"})
                return
            self._send(200, {"object": "list",
                             "data": [{"id": _model_name, "object": "model"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if "chat/completions" not in self.path:
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as exc:                      # noqa: BLE001
            self._send(400, {"error": repr(exc)})
            return

        job = {
            "messages": req.get("messages", []),
            "temperature": float(req.get("temperature", 0.0)),
            "max_tokens": int(req.get("max_tokens", 2048)),
            # CaP-X does not pass a seed. At temperature 0 that is harmless --
            # argmax has no RNG -- but the sampled arm needs each draw k to be
            # a distinct, REPRODUCIBLE stream, or k re-runs of draw 3 are three
            # different experiments. OPENCAP_SAMPLE_INDEX supplies it.
            "seed": req.get("seed") if req.get("seed") is not None
                    else _SAMPLE_SEED,
            "done": threading.Event(),
        }
        _jobs.put(job)
        job["done"].wait()

        if "error" in job:
            self._send(500, {"error": job["error"]})
            return
        r = job["result"]
        rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        _log_stats({
            "request_id": rid,
            "model": _model_name,
            "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"],
            "finish_reason": r.get("finish_reason", ""),
            "logprob_mean": r["logprob_mean"],
            "cumulative_logprob": r["cumulative_logprob"],
            "had_reasoning": bool(r["reasoning"]),
            "empty_content": not r["text"],
            "t": time.time(),
        })
        self._send(200, {
            "id": rid,
            "object": "chat.completion",
            "model": _model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": r["text"],
                            "reasoning_content": r["reasoning"] or None},
                # The engine's word, not ours. See the worker loop above.
                "finish_reason": r.get("finish_reason", "stop"),
            }],
            "usage": {
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "total_tokens": r["prompt_tokens"] + r["completion_tokens"],
            },
            # Non-standard, deliberately: Sec. 04's calibration axis needs a
            # confidence signal and this is the cheapest place to carry it.
            "cumulative_logprob": r["cumulative_logprob"],
        })


def main() -> int:
    global _llm, _model_name, _enable_thinking
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8110)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--enable-thinking", action="store_true",
                    help="Reasoning-mode ablation only; OFF is the control.")
    ap.add_argument("--stats-file", default=None,
                    help="JSONL sidecar of per-request generation stats (H4).")
    ap.add_argument("--deterministic", action="store_true",
                    help="Trade throughput for reproducibility. Disables the "
                         "two things that change the COMPUTATION PATH between "
                         "runs even when requests are serialised: automatic "
                         "prefix caching (a cache hit takes a different code "
                         "path from a cold prefill, and cache state differs "
                         "run to run) and CUDA graphs (capture/replay is not "
                         "bit-identical to eager). Measured: BATCH_MAX=1 alone "
                         "narrowed run-to-run spread from 5 trials to 1, but "
                         "did not reach zero.")
    args = ap.parse_args()

    global _sidecar_path
    _model_name = args.model
    _enable_thinking = args.enable_thinking
    _sidecar_path = args.stats_file
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[shim] listening on :{args.port} (503 until loaded)", flush=True)

    from vllm import LLM
    kw = dict(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed)
    if args.quantization:
        kw["quantization"] = args.quantization
    if args.deterministic:
        kw["enable_prefix_caching"] = False
        kw["enforce_eager"] = True
        # cuBLAS needs a fixed workspace for reproducible reductions
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        print("[shim] deterministic: prefix caching off, eager mode, "
              "fixed cuBLAS workspace", flush=True)
    _llm = LLM(**kw)
    threading.Thread(target=_worker, daemon=True).start()
    _ready.set()
    print("[shim] READY", flush=True)

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
