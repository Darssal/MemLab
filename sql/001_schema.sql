CREATE TABLE IF NOT EXISTS memory_items (
  key TEXT NOT NULL,
  agent_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  text_body TEXT NOT NULL,
  bytes INTEGER NOT NULL,
  importance DOUBLE PRECISION NOT NULL,
  created_at_ms BIGINT NOT NULL,
  last_access_ms BIGINT NOT NULL,
  access_count BIGINT NOT NULL DEFAULT 0,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  embedding JSONB,
  PRIMARY KEY (key, agent_id)
);

CREATE INDEX IF NOT EXISTS idx_memory_items_agent_time
  ON memory_items (agent_id, last_access_ms DESC);

CREATE TABLE IF NOT EXISTS checkpoints (
  agent_id TEXT NOT NULL,
  thread_id TEXT NOT NULL,
  ts_ms BIGINT NOT NULL,
  state_json JSONB NOT NULL,
  PRIMARY KEY (agent_id, thread_id, ts_ms)
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_latest
  ON checkpoints (agent_id, thread_id, ts_ms DESC);
