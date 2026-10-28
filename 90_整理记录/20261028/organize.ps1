param([Parameter(Mandatory=$true)][string]$PlanPath)
$ErrorActionPreference = 'Stop'
$plan = Get-Content -LiteralPath $PlanPath -Raw -Encoding utf8 | ConvertFrom-Json
$taskRoot = [IO.Path]::GetFullPath($plan.root).TrimEnd('\')
$stageRoot = Split-Path -Parent ([IO.Path]::GetFullPath($PlanPath))
$recordRoot = Join-Path $taskRoot '90_整理记录\20261028'
$rootPrefix = $taskRoot + '\'
function SafePath([string]$relative, [bool]$mustExist = $false) {
    $full = [IO.Path]::GetFullPath((Join-Path $taskRoot $relative))
    if (-not $full.StartsWith($rootPrefix,[StringComparison]::OrdinalIgnoreCase)) { throw "Out of intended root: $relative" }
    if ($mustExist) {
        $resolved = (Resolve-Path -LiteralPath $full).ProviderPath
        if (-not $resolved.StartsWith($rootPrefix,[StringComparison]::OrdinalIgnoreCase)) { throw "Resolved outside root: $resolved" }
        $item = Get-Item -LiteralPath $resolved -Force
        if ($item.LinkType -in @('Junction','SymbolicLink')) { throw "Do not move/delete links: $resolved" }
    }
    return $full
}
function Hash([string]$p) { return (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant() }
function Record($obj) {
    $obj | ConvertTo-Json -Compress -Depth 6 | Add-Content -LiteralPath (Join-Path $recordRoot '移动收据.jsonl') -Encoding utf8
}
$beforeVisible = @(Get-ChildItem -LiteralPath $taskRoot -Force | Where-Object { -not $_.Name.StartsWith('.') }).Count
foreach ($m in $plan.moves) {
    $null = SafePath $m.from $true
    $dest = SafePath $m.to
    if (Test-Path -LiteralPath $dest) { throw "Destination exists, do not merge: $dest" }
}
foreach ($f in $plan.verify_moved_files) {
    $p = SafePath $f.original $true
    if ((Hash $p) -ne $f.sha256) { throw "File changed after inventory: $p" }
}
foreach ($c in $plan.cache_delete) {
    $p = SafePath $c.original $true
    if ((Hash $p) -ne $c.sha256) { throw "Cache changed after inventory: $p" }
}
foreach ($d in $plan.duplicate_delete) {
    if ((Hash (SafePath $d.original $true)) -ne $d.sha256 -or (Hash (SafePath $d.keep $true)) -ne $d.sha256) { throw 'Duplicate hash mismatch' }
}
if (Test-Path -LiteralPath (Join-Path $recordRoot 'execution_result.json')) { throw 'Already organized; do not rerun blindly' }
New-Item -ItemType Directory -Path $recordRoot -Force | Out-Null
$entryBackup = Join-Path $recordRoot '整理前入口文件'
New-Item -ItemType Directory -Path $entryBackup -Force | Out-Null
foreach ($name in @('README_从这里开始.md','00_Codex工作同步索引.md')) {
    Copy-Item -LiteralPath (SafePath $name $true) -Destination (Join-Path $entryBackup $name)
}
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zipPath = Join-Path $recordRoot 'cache_backup.zip'
$zip = [IO.Compression.ZipFile]::Open($zipPath,[IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($c in $plan.cache_delete) {
        [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,(SafePath $c.original $true),$c.original.Replace('\','/'),[IO.Compression.CompressionLevel]::Optimal) | Out-Null
    }
} finally { $zip.Dispose() }
$verifyZip = [IO.Compression.ZipFile]::OpenRead($zipPath)
try {
    if ($verifyZip.Entries.Count -ne $plan.cache_delete.Count) { throw 'Cache backup incomplete' }
    foreach ($c in $plan.cache_delete) {
        $entry = $verifyZip.GetEntry($c.original.Replace('\','/'))
        $stream = $entry.Open()
        $alg = [Security.Cryptography.SHA256]::Create()
        try { $value = ([BitConverter]::ToString($alg.ComputeHash($stream))).Replace('-','').ToLowerInvariant() }
        finally { $alg.Dispose(); $stream.Dispose() }
        if ($value -ne $c.sha256) { throw "Cache backup checksum failed: $($c.original)" }
    }
} finally { $verifyZip.Dispose() }
foreach ($m in $plan.moves) {
    $src = SafePath $m.from $true
    $dest = SafePath $m.to
    New-Item -ItemType Directory -Path (Split-Path -Parent $dest) -Force | Out-Null
    Move-Item -LiteralPath $src -Destination $dest
    Record @{action='move';from=$m.from;to=$m.to}
}
foreach ($f in $plan.verify_moved_files) {
    if ((Hash (SafePath $f.final $true)) -ne $f.sha256) { throw "Moved file differs: $($f.final)" }
}
$cacheParents = [Collections.Generic.HashSet[string]]::new()
foreach ($c in $plan.cache_delete) {
    $path = SafePath $c.path $true
    if ((Hash $path) -ne $c.sha256) { throw "Refusing changed cache: $path" }
    Remove-Item -LiteralPath $path -Force
    Record @{action='delete_cache';original=$c.original;path=$c.path;sha256=$c.sha256;backup='cache_backup.zip'}
    $parent = Split-Path -Parent $path
    while ($parent.StartsWith($rootPrefix,[StringComparison]::OrdinalIgnoreCase) -and ($parent -match '(\\__pycache__|\\.ruff_cache|\\.pytest_cache|\\.mypy_cache)(\\|$)')) {
        $null = $cacheParents.Add($parent)
        $parent = Split-Path -Parent $parent
    }
}
foreach ($p in ($cacheParents | Sort-Object Length -Descending)) {
    if (Test-Path -LiteralPath $p) {
        $relative = $p.Substring($rootPrefix.Length)
        $safe = SafePath $relative $true
        if (@(Get-ChildItem -LiteralPath $safe -Force).Count -eq 0) { Remove-Item -LiteralPath $safe -Force }
    }
}
foreach ($d in $plan.duplicate_delete) {
    $path = SafePath $d.path $true
    $keep = SafePath $d.keep $true
    if ((Hash $path) -ne $d.sha256 -or (Hash $keep) -ne $d.sha256) { throw 'Duplicate changed before deletion' }
    Remove-Item -LiteralPath $path -Force
    @("# 重复副本已合并", "", "本文件与保留副本SHA256完全一致，已于2026-10-28清理重复内容。", "",
      "保留位置：$keep", "SHA256：$($d.sha256)", "可用本次整理的rollback.ps1从保留副本恢复。") |
      Set-Content -LiteralPath ($path + '.重复副本说明.md') -Encoding utf8
    Record @{action='delete_exact_duplicate';original=$d.original;path=$d.path;keep=$d.keep;sha256=$d.sha256}
}
$experimentSource = [IO.Path]::GetFullPath($plan.copy_experiment.from)
$workspacePrefix = [IO.Path]::GetFullPath($plan.workspace).TrimEnd('\') + '\'
if (-not $experimentSource.StartsWith($workspacePrefix,[StringComparison]::OrdinalIgnoreCase)) { throw 'Unexpected experiment source' }
$experimentDest = SafePath $plan.copy_experiment.to
if (Test-Path -LiteralPath $experimentDest) { throw 'Experiment destination exists' }
New-Item -ItemType Directory -Path $experimentDest | Out-Null
$copied = 0
foreach ($f in Get-ChildItem -LiteralPath $experimentSource -Recurse -File) {
    $relative = $f.FullName.Substring($experimentSource.Length+1)
    $dest = [IO.Path]::GetFullPath((Join-Path $experimentDest $relative))
    if (-not $dest.StartsWith($experimentDest+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Copy escaped experiment folder' }
    New-Item -ItemType Directory -Path (Split-Path -Parent $dest) -Force | Out-Null
    Copy-Item -LiteralPath $f.FullName -Destination $dest
    if ((Hash $f.FullName) -ne (Hash $dest)) { throw "Copied experiment mismatch: $relative" }
    $copied++
}
foreach ($name in @('README_从这里开始.md','00_Codex工作同步索引.md')) {
    Copy-Item -LiteralPath (Join-Path $stageRoot $name) -Destination (Join-Path $taskRoot $name) -Force
}
foreach ($f in Get-ChildItem -LiteralPath $stageRoot -File) {
    if ($f.Name -in @('README_从这里开始.md','00_Codex工作同步索引.md')) { continue }
    Copy-Item -LiteralPath $f.FullName -Destination (Join-Path $recordRoot $f.Name) -Force
}
$mapping = @()
foreach ($f in $plan.verify_moved_files) { $mapping += [pscustomobject]@{old_path=$f.original;new_path=$f.final;action='moved';sha256=$f.sha256} }
foreach ($c in $plan.cache_delete) { $mapping += [pscustomobject]@{old_path=$c.original;new_path='90_整理记录/20261028/cache_backup.zip';action='cache_deleted_backed_up';sha256=$c.sha256} }
foreach ($d in $plan.duplicate_delete) { $mapping += [pscustomobject]@{old_path=$d.original;new_path=$d.keep;action='duplicate_merged';sha256=$d.sha256} }
$mapping | Export-Csv -LiteralPath (Join-Path $recordRoot '新旧路径映射.csv') -NoTypeInformation -Encoding utf8BOM
$afterVisible = @(Get-ChildItem -LiteralPath $taskRoot -Force | Where-Object { -not $_.Name.StartsWith('.') }).Count
$result = [pscustomobject]@{
    status='COMPLETED';date='2026-10-28';root=$taskRoot;root_visible_items_before=$beforeVisible;root_visible_items_after=$afterVisible
    moved_root_items=$plan.moves.Count;sha_verified_moved_files=$plan.verify_moved_files.Count
    deleted_cache_files=$plan.cache_delete.Count;deleted_cache_bytes=($plan.cache_delete | Measure-Object bytes -Sum).Sum
    merged_duplicate_files=$plan.duplicate_delete.Count;merged_duplicate_bytes=($plan.duplicate_delete | Measure-Object bytes -Sum).Sum
    copied_experiment_files=$copied;cache_backup_bytes=(Get-Item -LiteralPath $zipPath).Length
    protected_assets_preserved=$true
}
$result | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $recordRoot 'execution_result.json') -Encoding utf8
$result | ConvertTo-Json -Depth 5

