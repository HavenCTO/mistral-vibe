# Migration Guide: LLM Multiplexer

This guide helps existing vibe users migrate to the new LLM Multiplexer feature.

## Overview

The LLM Multiplexer is an optional feature that provides automatic failover and load balancing across multiple LLM models. **No migration is required** — your existing configuration continues to work unchanged.

## For Users on Single-Model Configuration

### No Action Required

If you're using a simple single-model configuration, **everything continues to work as before**:

```toml
# Your existing config - no changes needed
active_model = "devstral-2"

[[providers]]
name = "mistral"
api_base = "https://api.mistral.ai/v1"
api_key_env_var = "MISTRAL_API_KEY"
backend = "mistral"

[[models]]
name = "mistral-vibe-cli-latest"
provider = "mistral"
alias = "devstral-2"
```

### Opt-in to Multiplexer

To enable multiplexer benefits, add a `[multiplexer]` section:

```toml
# Keep your existing config as-is
active_model = "devstral-2"

[[providers]]
name = "mistral"
api_base = "https://api.mistral.ai/v1"
api_key_env_var = "MISTRAL_API_KEY"
backend = "mistral"

[[models]]
name = "mistral-vibe-cli-latest"
provider = "mistral"
alias = "devstral-2"

[[models]]
name = "devstral-small-latest"
provider = "mistral"
alias = "devstral-small"

# NEW: Add multiplexer
[multiplexer]
enabled = true
mode = "failover"

[[multiplexer.pool]]
model = "devstral-2"

[[multiplexer.pool]]
model = "devstral-small"
is_fallback = true
```

## For Users with Custom Providers

Custom providers work seamlessly with the multiplexer:

```toml
# Your existing custom provider
[[providers]]
name = "my-custom"
api_base = "https://my-api.example.com/v1"
api_key_env_var = "MY_API_KEY"
backend = "generic"

[[models]]
name = "my-model"
provider = "my-custom"
alias = "custom"

# Add to multiplexer pool
[multiplexer]
enabled = true
mode = "load_balance"

[[multiplexer.pool]]
model = "devstral-2"
weight = 70

[[multiplexer.pool]]
model = "custom"
weight = 30
```

## For Users with Multiple Models

If you already have multiple models configured, you can now use them together:

**Before (manual model switching):**
```toml
active_model = "devstral-2"  # Manually change this to switch models

[[models]]
name = "mistral-vibe-cli-latest"
provider = "mistral"
alias = "devstral-2"

[[models]]
name = "devstral-small-latest"
provider = "mistral"
alias = "devstral-small"
```

**After (automatic failover):**
```toml
active_model = "devstral-2"

[[models]]
name = "mistral-vibe-cli-latest"
provider = "mistral"
alias = "devstral-2"

[[models]]
name = "devstral-small-latest"
provider = "mistral"
alias = "devstral-small"

[multiplexer]
enabled = true
mode = "failover"

[[multiplexer.pool]]
model = "devstral-2"

[[multiplexer.pool]]
model = "devstral-small"
is_fallback = true
```

## New Configuration Options

### Multiplexer Section

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `false` | Enable multiplexer mode |
| `mode` | string | `"single"` | Operation mode: `single`, `failover`, `load_balance` |
| `rate_limit_disable_duration_ms` | integer | `60000` | Model disable time after rate limit |
| `persistent_error_disable_duration_ms` | integer | `15000` | Model disable time after error |

### Pool Entries

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model` | string | required | Model alias from `[[models]]` |
| `weight` | integer | `1` | Selection weight for load balancing |
| `max_concurrent` | integer | unlimited | Max parallel requests |
| `is_fallback` | boolean | `false` | Use only when primaries fail |

## New Slash Command

The `/mux` command shows multiplexer status:

```
/mux
```

This displays:
- Current mode (single/failover/load_balance)
- Pool status (available/total models)
- Per-model statistics

## Transition Timeline

### Phase 1: Opt-in (Current)

- Multiplexer available but **disabled by default**
- Users explicitly enable with `enabled = true`
- Full backward compatibility maintained

### Phase 2: Soft Default (Future)

- New installations will have `enabled = true` in default config
- Existing configurations remain unchanged
- Documentation will emphasize multiplexer benefits

### Phase 3: Hard Default (Future)

- Multiplexer enabled by default for all users
- `enabled = false` required to use single-model mode
- Deprecation warnings for single-model without explicit opt-out

### Phase 4: Legacy Removal (Future)

- Single-model mode removed
- All requests go through multiplexer
- Single-model = multiplexer with one model in pool

## Frequently Asked Questions

### Will my existing config break?

**No.** The multiplexer is opt-in only. Your existing configuration continues to work exactly as before.

### Do I need to change my providers or models?

**No.** The multiplexer works with your existing providers and models. You just reference them by alias in the pool configuration.

### What happens if I don't enable the multiplexer?

**Nothing changes.** Vibe will use the `active_model` with the standard single-model backend, exactly as it did before.

### Can I use local models with the multiplexer?

**Yes.** Local models (llama.cpp, Ollama, etc.) work with the multiplexer. You can even set up local-primary with cloud-fallback:

```toml
[[multiplexer.pool]]
model = "local"

[[multiplexer.pool]]
model = "cloud"
is_fallback = true
```

### How do I check if the multiplexer is working?

Use the `/mux` command in vibe to see the current status and statistics.

### What if a pool model's API key is missing?

Vibe will show an error at startup if any pool model's provider is missing its required API key.

## Troubleshooting

### Configuration Errors

**"unknown model alias"**
- Ensure the `model` value in `[[multiplexer.pool]]` matches an `alias` in `[[models]]`

**"at least one non-fallback model"**
- At least one pool entry must have `is_fallback = false` (or omit it, as `false` is default)

**"Missing API key"**
- Set the environment variable for each provider used in the pool

### Runtime Issues

**Model keeps getting disabled**
- Check `/mux` status for rate limit counts
- Increase `rate_limit_disable_duration_ms` if needed
- Add more models to the pool

**No failover happening**
- Verify `mode = "failover"` is set
- Check that fallback models are configured correctly

## See Also

- [Multiplexer Documentation](docs/multiplexer.md)
- [Configuration Examples](docs/multiplexer-examples.md)
