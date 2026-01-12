import traceback
import openai
import time
import os
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import logging, sys
from REspEval.respeval.utils_TSP_flow_aggregate_plot import _handle_windows_path_length_limit

import sqlite3
from contextlib import contextmanager


logging.basicConfig(
    level=logging.INFO,                      # show INFO and above
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,                       # send to stdout (default is stderr)
    force=True                               # override existing handlers (Py>=3.8)
)

# ---------- concurrency-safe SQLite cache ----------

class SqliteCache:
    """
    Minimal JSON-value cache using SQLite with WAL.
    Safe for multi-process reads/writes on the same machine.
    """
    def __init__(self, path: str, timeout=30):
        self.path = os.fspath(Path(path))
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self._init_db()

    def _init_db(self):
        with self._connect() as con:
            # Enable WAL for better concurrency (many readers + 1 writer)
            con.execute("PRAGMA journal_mode=WAL;")
            # Durability/speed trade-off
            con.execute("PRAGMA synchronous=NORMAL;")
            con.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    k TEXT PRIMARY KEY,
                    v TEXT NOT NULL
                )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_cache_k ON cache(k)")

    @contextmanager
    def _connect(self):
        con = sqlite3.connect(self.path, timeout=self.timeout)
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def get(self, key: str):
        try:
            with self._connect() as con:
                row = con.execute("SELECT v FROM cache WHERE k = ?", (key,)).fetchone()
                return json.loads(row[0]) if row else None
        except Exception as e:
            logging.warning(f"Cache read failed ({e}); continuing without cache.")
            return None

    def set(self, key: str, record: dict):
        try:
            payload = json.dumps(record, ensure_ascii=False)
            with self._connect() as con:
                con.execute(
                    "INSERT INTO cache(k, v) VALUES(?, ?) "
                    "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                    (key, payload),
                )
        except Exception as e:
            logging.warning(f"Cache write failed ({e}); continuing without cache.")

import json as _json
class OpenAIModel(object):
    """
    Simple Azure OpenAI wrapper with on-disk caching (SQLite, multi-process safe).
    - To enable caching, pass cache_file="path/to/cache.db".
    - Cache key: SHA256 over (model_name, system_prompt, user_input, max_output_length, reasoning_effort, cache_version).
    - Stored value: {'output': <str>, 'meta': {...}} as JSON in SQLite.
    """

    def __init__(self, model_name, key_path="api.key", cache_file=""):
        self.model_name = model_name
        key_path = Path(_handle_windows_path_length_limit(Path(key_path)))
        self.key_path = key_path

        # bump when inputs/format change; also included in filename to avoid collisions
        self.cache_version = "2"

        # read api key and base from the key file
        with open(key_path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
            api_version = lines[0].strip().split('=')[1].strip()
            api_base    = lines[1].strip().split('=')[1].strip()
            api_key     = lines[2].strip().split('=')[1].strip()

        self.client = openai.AzureOpenAI(
            api_version=api_version,
            azure_endpoint=api_base,
            api_key=api_key
        )

        # cache
        self.cache_file = cache_file
        self._cache = None
        if self.cache_file:
            cf = Path(self.cache_file)
            # include version in filename to prevent old readers loading new format
            versioned = cf.with_name(f"{cf.stem}.v{self.cache_version}{cf.suffix or '.db'}")
            self.cache_file = os.fspath(versioned)
            self._cache = SqliteCache(self.cache_file)
            logging.info(f"Using SQLite cache at {self.cache_file}")

    # ----------------------------
    # Cache helpers
    # ----------------------------
    def _make_cache_key(self, system_prompt, user_input, max_output_length, reasoning_effort):
        payload = {
            "cache_version": self.cache_version,
            "model_name": self.model_name,
            "system_prompt": system_prompt,
            "user_input": user_input,
            "max_output_length": max_output_length,
            "reasoning_effort": reasoning_effort or "",
        }
        blob = _json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def load_cache(self, key):
        if not self._cache:
            return None
        return self._cache.get(key)

    def save_cache(self, key, record):
        if not self._cache:
            return
        self._cache.set(key, record)

    

    # ----------------------------
    # Generation
    # ----------------------------
    def generate(self, system_prompt, user_input, max_output_length=1500, reasoning_effort="minimal"):
        # 1) Try cache
        cache_key = self._make_cache_key(system_prompt, user_input, max_output_length, reasoning_effort)
        cached = self.load_cache(cache_key)
        if cached and isinstance(cached, dict) and "output" in cached:
            logging.info("!!!!!!!!! Cache hit for identical prompt. Returning cached output.")
            return cached["output"]

        # 2) Call model (only if not cached)
        if self.model_name in ['gpt-5', 'gpt-5 mini', 'gpt-5 nano']:
            azure_model_id_dict = {
                'gpt-5': 'gpt-5-2025-08-07',
                'gpt-5 mini': 'gpt-5-mini-2025-08-07',
                'gpt-5 nano': 'gpt-5-nano-2025-08-07'
            }
            azure_model = azure_model_id_dict[self.model_name]

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
            response = self.call_gpt5_azure(
                messages,
                azure_model,
                max_len=max_output_length,
                reasoning_effort=reasoning_effort
            )
            output = response.choices[0].message.content
        else:
            raise NotImplementedError(f"Model '{self.model_name}' is not supported in this wrapper.")

        # 3) Save to cache
        self.save_cache(
            cache_key,
            {
                "output": output,
                "meta": {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "model_name": self.model_name,
                    "max_output_length": max_output_length,
                    "reasoning_effort": reasoning_effort or "",
                    "system_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
                    "user_sha256": hashlib.sha256(user_input.encode("utf-8")).hexdigest(),
                }
            }
        )
        return output

    # ----------------------------
    # Azure OpenAI call
    # ----------------------------
    def call_gpt5_azure(self, message, azure_model, max_len=1500, reasoning_effort="minimal", max_retries=5):
        """
        Call the OpenAI API with the provided message and parameters.
        If there is no output due to token limit (finish_reason == 'length' and empty content),
        retry with max_completion_tokens increased by 1500 (up to max_retries times).
        """
        response = None
        received = False
        num_rate_errors = 0
        attempts_len = 0

        from openai import OpenAIError, APIConnectionError, RateLimitError, BadRequestError

        while not received:
            try:
                kwargs = dict(model=azure_model, messages=message, max_completion_tokens=max_len)
                if reasoning_effort:
                    kwargs["reasoning_effort"] = reasoning_effort

                response = self.client.chat.completions.create(**kwargs)

                # Check for "no output because of token limit"
                choice = response.choices[0]
                content = getattr(choice.message, "content", None) or ""
                if choice.finish_reason == "length" and content.strip() == "":
                    attempts_len += 1
                    if attempts_len > max_retries:
                        logging.warning(
                            f"No output due to token limit after {attempts_len} tries (max_len={max_len}). Returning last response."
                        )
                        received = True
                    else:
                        max_len += 1500
                        logging.warning(
                            f"No output due to token limit. Retrying with max_completion_tokens={max_len} "
                            f"(attempt {attempts_len}/{max_retries})..."
                        )
                        continue  # retry loop
                else:
                    received = True  # normal success

            except BadRequestError:
                logging.critical(f"InvalidRequestError\nPrompt passed in:\n\n{message}\n\n")
                traceback.print_exc()
                raise  # stop execution or handle upstream

            except (RateLimitError, APIConnectionError, OpenAIError) as e:
                logging.error(f"OpenAI API error: {e} ({num_rate_errors}). Waiting {2 ** num_rate_errors} sec")
                traceback.print_exc()
                time.sleep(2 ** num_rate_errors)
                num_rate_errors += 1
                continue

            except Exception as e:
                logging.error(f"Unexpected error: {e} ({num_rate_errors}). Waiting {2 ** num_rate_errors} sec")
                traceback.print_exc()
                time.sleep(2 ** num_rate_errors)
                num_rate_errors += 1
                continue

        return response