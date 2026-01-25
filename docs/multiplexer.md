# LLM Multiplexer

The LLM Multiplexer allows you to use multiple LLM models with automatic failover and load balancing. This feature improves reliability by automatically switching to backup models when primary models are rate-limited or unavailable.

## Quick Start

Add the following to your `~/.config/vibe/config.toml`:

```toml
[multiplexer]
enabled = true
mode = "failover"

[[multiplexer.pool]]
model = "devstral-2"

[[multiplexer.pool]]
model = "devstral-small"
is_fallback = true
```

## Features

- **Automatic Failover**: Seamlessly switch to backup models when primary models fail
- **Load Balancing**: Distribute requests across models using weighted selection
- **Rate Limit Handling**: Automatically disable rate-limited models temporarily
- **Statistics**: Track per-model success rates and usage
- **Backward Compatible**: Works with existing single-model configurations

## Configuration Options

### `[multiplexer]`

#### `enabled`

Enable or disable the multiplexer.

- **Type**: boolean
- **Default**: `false`
- **Example**: `enabled = true`

When disabled, vibe uses the standard single-model backend with `active_model`.

#### `mode`

The multiplexer operation mode.

- **Type**: string
- **Default**: `"single"`
- **Options**:
  - `"single"` - Use only `active_model` (backward compatible)
  - `"failover"` - Try models in order, switch on errors
  - `"load_balance"` - Weighted random selection across pool

```toml
[multiplexer]
enabled = true
mode = "failover"
```

#### `rate_limit_disable_duration_ms`

How long to disable a model after receiving a rate limit error (HTTP 429).

- **Type**: integer (milliseconds)
- **Default**: `60000` (60 seconds)
- **Minimum**: `1000`

```toml
[multiplexer]
rate_limit_disable_duration_ms = 120000  # 2 minutes
```

#### `persistent_error_disable_duration_ms`

How long to disable a model after a connection or server error.

- **Type**: integer (milliseconds)
- **Default**: `15000` (15 seconds)
- **Minimum**: `1000`

```toml
[multiplexer]
persistent_error_disable_duration_ms = 30000  # 30 seconds
```

### `[[multiplexer.pool]]`

Each pool entry defines a model in the multiplexer pool.

#### `model` (required)

Model alias referencing a model in `[[models]]`.

- **Type**: string
- **Required**: Yes

```toml
[[multiplexer.pool]]
model = "devstral-2"
```

#### `weight`

Selection weight for load balancing. Higher weights mean more likely selection.

- **Type**: integer
- **Default**: `1`
- **Minimum**: `1`

```toml
[[multiplexer.pool]]
model = "devstral-2"
weight = 70  # 70% of requests

[[multiplexer.pool]]
model = "devstral-small"
weight = 30  # 30% of requests
```

#### `max_concurrent`

Maximum parallel requests to this model.

- **Type**: integer or null
- **Default**: `null` (unlimited)
- **Minimum**: `1`

```toml
[[multiplexer.pool]]
model = "devstral-2"
max_concurrent = 5  # Max 5 concurrent requests
```

#### `is_fallback`

Mark this model as a fallback. Fallback models are only used when all primary models fail.

- **Type**: boolean
- **Default**: `false`

```toml
[[multiplexer.pool]]
model = "devstral-2"
is_fallback = false  # Primary model

[[multiplexer.pool]]
model = "devstral-small"
is_fallback = true  # Fallback model
```

## Modes Explained

### Single Mode (`mode = "single"`)

Uses only the `active_model` from your configuration. This is the default behavior and maintains backward compatibility.

```toml
[multiplexer]
enabled = true
mode = "single"
```

### Failover Mode (`mode = "failover"`)

Tries models in pool order. When a model returns an error (rate limit, server error), the multiplexer automatically switches to the next available model.

```toml
[multiplexer]
enabled = true
mode = "failover"

[[multiplexer.pool]]
model = "devstral-2"  # Primary

[[multiplexer.pool]]
model = "devstral-small"
is_fallback = true  # Used when primary fails
```

**Behavior:**
1. All requests go to the primary model
2. On rate limit (429), primary is disabled for `rate_limit_disable_duration_ms`
3. Requests automatically route to next available model
4. After timeout, primary is re-enabled

### Load Balance Mode (`mode = "load_balance"`)

Distributes requests across models using weighted random selection.

```toml
[multiplexer]
enabled = true
mode = "load_balance"

[[multiplexer.pool]]
model = "devstral-2"
weight = 70

[[multiplexer.pool]]
model = "devstral-small"
weight = 30
```

**Behavior:**
- 70% of requests go to `devstral-2`
- 30% of requests go to `devstral-small`
- If a model is disabled (rate limited), its weight is redistributed

## Slash Commands

### `/mux`

Display multiplexer status and statistics.

```
/mux
```

Output includes:
- Current mode
- Pool status (available/total models)
- Per-model statistics:
  - Success count
  - Rate limit count
  - Failure count
  - Success rate
  - Disabled status

Example output:
```
## Multiplexer Status

**Mode:** failover
**Pool:** 2/2 models available

### Per-Model Statistics

**mistral-vibe-cli-latest**: 🟢 active
  - Success: 45 | Rate limited: 2 | Failed: 0 | Success rate: 96%

**devstral-small-latest**: 🟢 active
  - Success: 5 | Rate limited: 0 | Failed: 0 | Success rate: 100%
```

## Examples

See [Multiplexer Examples](./multiplexer-examples.md) for complete configuration examples.

## Troubleshooting

### "unknown model alias" Error

The model alias in `[[multiplexer.pool]]` doesn't match any `[[models]]` entry.

```toml
# Wrong - alias doesn't exist
[[multiplexer.pool]]
model = "nonexistent-alias"

# Correct - use the alias from [[models]]
[[models]]
name = "mistral-vibe-cli-latest"
alias = "devstral-2"  # This is the alias

[[multiplexer.pool]]
model = "devstral-2"  # Use the alias here
```

### "at least one non-fallback model" Error

All models in the pool are marked as fallbacks.

```toml
# Wrong - all fallbacks
[[multiplexer.pool]]
model = "model1"
is_fallback = true

[[multiplexer.pool]]
model = "model2"
is_fallback = true

# Correct - at least one primary
[[multiplexer.pool]]
model = "model1"
is_fallback = false  # Primary

[[multiplexer.pool]]
model = "model2"
is_fallback = true  # Fallback
```

### Missing API Key Error

Pool models require valid API keys for their providers.

```toml
# Ensure the API key environment variable is set
[[providers]]
name = "mistral"
api_key_env_var = "MISTRAL_API_KEY"  # Must be set in environment
```

## See Also

- [Configuration Examples](./multiplexer-examples.md)
- [Migration Guide](../MIGRATION.md)
