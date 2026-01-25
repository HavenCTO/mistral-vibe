# Multiplexer Configuration Examples

This document provides complete configuration examples for common multiplexer use cases.

## Basic Failover Configuration

The simplest multiplexer setup with automatic failover to a backup model.

```toml
# ~/.config/vibe/config.toml

active_model = "devstral-2"

# Providers
[[providers]]
name = "mistral"
api_base = "https://api.mistral.ai/v1"
api_key_env_var = "MISTRAL_API_KEY"
backend = "mistral"

# Models
[[models]]
name = "mistral-vibe-cli-latest"
provider = "mistral"
alias = "devstral-2"
input_price = 0.4
output_price = 2.0

[[models]]
name = "devstral-small-latest"
provider = "mistral"
alias = "devstral-small"
input_price = 0.1
output_price = 0.3

# Multiplexer
[multiplexer]
enabled = true
mode = "failover"

[[multiplexer.pool]]
model = "devstral-2"

[[multiplexer.pool]]
model = "devstral-small"
is_fallback = true
```

## Weighted Load Balancing

Distribute traffic across multiple models with custom weights.

```toml
# ~/.config/vibe/config.toml

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

[multiplexer]
enabled = true
mode = "load_balance"

[[multiplexer.pool]]
model = "devstral-2"
weight = 70  # 70% of requests

[[multiplexer.pool]]
model = "devstral-small"
weight = 30  # 30% of requests
```

## Multi-Provider Setup

Use models from different providers in the same pool.

```toml
# ~/.config/vibe/config.toml

active_model = "devstral-2"

# Mistral Provider
[[providers]]
name = "mistral"
api_base = "https://api.mistral.ai/v1"
api_key_env_var = "MISTRAL_API_KEY"
backend = "mistral"

# Custom Provider
[[providers]]
name = "custom-llm"
api_base = "https://api.custom-llm.com/v1"
api_key_env_var = "CUSTOM_API_KEY"
backend = "generic"

# Mistral Model
[[models]]
name = "mistral-vibe-cli-latest"
provider = "mistral"
alias = "devstral-2"

# Custom Model
[[models]]
name = "custom-model-v1"
provider = "custom-llm"
alias = "custom"

[multiplexer]
enabled = true
mode = "failover"

[[multiplexer.pool]]
model = "devstral-2"

[[multiplexer.pool]]
model = "custom"
is_fallback = true
```

## Rate Limit Resilience

Configuration optimized for handling rate limits with longer cooldowns.

```toml
# ~/.config/vibe/config.toml

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

[multiplexer]
enabled = true
mode = "failover"
# Longer cooldown after rate limits
rate_limit_disable_duration_ms = 120000  # 2 minutes
# Shorter cooldown for connection errors
persistent_error_disable_duration_ms = 10000  # 10 seconds

[[multiplexer.pool]]
model = "devstral-2"

[[multiplexer.pool]]
model = "devstral-small"
is_fallback = true
```

## Concurrency Limiting

Limit parallel requests to avoid overwhelming providers.

```toml
# ~/.config/vibe/config.toml

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

[multiplexer]
enabled = true
mode = "load_balance"

[[multiplexer.pool]]
model = "devstral-2"
weight = 60
max_concurrent = 3  # Max 3 parallel requests

[[multiplexer.pool]]
model = "devstral-small"
weight = 40
max_concurrent = 5  # Max 5 parallel requests
```

## Local + Cloud Hybrid

Use a local model as primary with cloud fallback.

```toml
# ~/.config/vibe/config.toml

active_model = "local"

# Local llama.cpp server
[[providers]]
name = "llamacpp"
api_base = "http://127.0.0.1:8080/v1"
api_key_env_var = ""
backend = "generic"

# Mistral cloud
[[providers]]
name = "mistral"
api_base = "https://api.mistral.ai/v1"
api_key_env_var = "MISTRAL_API_KEY"
backend = "mistral"

[[models]]
name = "devstral"
provider = "llamacpp"
alias = "local"
input_price = 0.0
output_price = 0.0

[[models]]
name = "mistral-vibe-cli-latest"
provider = "mistral"
alias = "cloud"
input_price = 0.4
output_price = 2.0

[multiplexer]
enabled = true
mode = "failover"

# Local model is primary (free)
[[multiplexer.pool]]
model = "local"

# Cloud fallback when local is unavailable
[[multiplexer.pool]]
model = "cloud"
is_fallback = true
```

## Three-Tier Failover

Complex setup with primary, secondary, and emergency fallback.

```toml
# ~/.config/vibe/config.toml

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

[[models]]
name = "mistral-tiny"
provider = "mistral"
alias = "tiny"

[multiplexer]
enabled = true
mode = "failover"

# Primary - best quality
[[multiplexer.pool]]
model = "devstral-2"
weight = 1

# Secondary - good balance
[[multiplexer.pool]]
model = "devstral-small"
weight = 1

# Emergency fallback - always available
[[multiplexer.pool]]
model = "tiny"
is_fallback = true
```

## Minimal Configuration (Backward Compatible)

Keep using single-model mode while having multiplexer available.

```toml
# ~/.config/vibe/config.toml

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

# Multiplexer disabled by default - existing behavior unchanged
[multiplexer]
enabled = false
```

## Environment Variable Reference

Required environment variables for these examples:

```bash
# Mistral API
export MISTRAL_API_KEY="your-mistral-api-key"

# Custom provider (if using)
export CUSTOM_API_KEY="your-custom-api-key"
```

## See Also

- [Multiplexer Documentation](./multiplexer.md)
- [Migration Guide](../MIGRATION.md)
