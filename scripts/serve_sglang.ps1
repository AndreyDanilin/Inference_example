param(
    [string]$Config = "configs/sglang.qwen3-0.6b.yaml",
    [string]$Model = "Qwen/Qwen3-0.6B",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

uv run --extra cuda inference-lab serve --config $Config --model $Model @RemainingArgs
