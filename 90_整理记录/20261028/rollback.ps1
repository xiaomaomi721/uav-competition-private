param([string]$Root = 'C:\Users\Lenovo\OneDrive\Documents\桌面\无人机')
$ErrorActionPreference='Stop'
$Root=[IO.Path]::GetFullPath($Root).TrimEnd('\')
$prefix=$Root+'\'
$record=Join-Path $Root '90_整理记录\20261028'
$plan=Get-Content -LiteralPath (Join-Path $record 'organization_plan.json') -Raw -Encoding utf8 | ConvertFrom-Json
function Safe([string]$rel) {
 $p=[IO.Path]::GetFullPath((Join-Path $Root $rel))
 if(-not $p.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)){throw "Unsafe path: $rel"}
 if(Test-Path -LiteralPath $p){
  $r=(Resolve-Path -LiteralPath $p).ProviderPath
  if(-not $r.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase)){throw "Resolved outside root: $r"}
  if((Get-Item -LiteralPath $p -Force).LinkType -in @('Junction','SymbolicLink')){throw "Refuse link: $p"}
 }
 return $p
}
function Hash([string]$p){(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()}
foreach($m in $plan.moves){
 if(Test-Path -LiteralPath (Safe $m.from)){throw "Original location now exists; inspect before restoring: $($m.from)"}
 if(-not (Test-Path -LiteralPath (Safe $m.to))){throw "Organized path missing: $($m.to)"}
}
foreach($f in $plan.verify_moved_files){
 if((Hash (Safe $f.final)) -ne $f.sha256){throw "Content changed after organization; preserve edits and resolve first: $($f.final)"}
}
$moves=@($plan.moves)
[array]::Reverse($moves)
foreach($m in $moves){
 $from=Safe $m.to;$to=Safe $m.from
 New-Item -ItemType Directory -Path (Split-Path -Parent $to) -Force | Out-Null
 Move-Item -LiteralPath $from -Destination $to
}
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip=[IO.Compression.ZipFile]::OpenRead((Join-Path $record 'cache_backup.zip'))
try{
 foreach($c in $plan.cache_delete){
  $to=Safe $c.original
  if(Test-Path -LiteralPath $to){throw "Cache exists, refusing overwrite: $to"}
  New-Item -ItemType Directory -Path (Split-Path -Parent $to) -Force | Out-Null
  $e=$zip.GetEntry($c.original.Replace('\','/'))
  [IO.Compression.ZipFileExtensions]::ExtractToFile($e,$to)
  if((Hash $to) -ne $c.sha256){throw "Restored cache hash failed: $to"}
 }
}finally{$zip.Dispose()}
foreach($d in $plan.duplicate_delete){
 $keep=Safe $d.keep;$to=Safe $d.original
 if((Hash $keep) -ne $d.sha256){throw 'Canonical ZIP changed'}
 if(Test-Path -LiteralPath $to){throw "Duplicate target exists: $to"}
 Copy-Item -LiteralPath $keep -Destination $to
}
foreach($name in @('README_从这里开始.md','00_Codex工作同步索引.md')){
 $current=Safe $name
 if(Test-Path -LiteralPath $current){
  $saved=Join-Path $record ($name+'.恢复前保留')
  if(Test-Path -LiteralPath $saved){throw "Previous rollback save exists: $saved"}
  Copy-Item -LiteralPath $current -Destination $saved
 }
 Copy-Item -LiteralPath (Join-Path (Join-Path $record '整理前入口文件') $name) -Destination $current -Force
}
'已恢复39项原目录路径、缓存和重复ZIP；新增BLDA材料与整理收据继续保留。'

