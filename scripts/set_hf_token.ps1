# Store the Hugging Face token that is on the clipboard inside WSL (~/.config/niko/secrets.sh).
# The token never goes into the repo. The clipboard is cleared afterwards.
$text = Get-Clipboard -Raw
if ($null -eq $text) { $text = "" }
$found = [regex]::Matches($text, 'hf_[A-Za-z0-9]{20,}')
if ($found.Count -eq 0) {
    Write-Host ""
    Write-Host "No Hugging Face token (hf_...) on the clipboard." -ForegroundColor Yellow
    Write-Host "Copy the token first (Copy button on huggingface.co), then run this again."
    exit 1
}
$token = $found[$found.Count - 1].Value
$script = "/mnt/d/Pack/Track Nhad/scripts/wsl/set_hf_token.sh"
$token | wsl.exe -d Ubuntu-24.04 --exec bash $script
if ($LASTEXITCODE -eq 0) {
    Set-Clipboard -Value " "
    Write-Host ""
    Write-Host "Done. You can close this window." -ForegroundColor Green
}
