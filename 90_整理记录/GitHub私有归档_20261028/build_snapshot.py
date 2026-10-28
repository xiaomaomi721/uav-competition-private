import json, shutil, hashlib
from pathlib import Path
from prepare_snapshot import BASE, SOURCE, DEST, digest

plan = json.loads((BASE / 'snapshot_plan.json').read_text(encoding='utf-8'))
assert not plan['credential_scan'], 'Review scan findings before export.'
assert not DEST.exists(), 'Do not overwrite an existing export.'
DEST.mkdir(parents=True)
for item in plan['files']:
    if item['action'] != 'upload': continue
    src, dst = SOURCE / item['path'], DEST / item['path']
    assert digest(src) == item['sha256'], f'Source changed: {item["path"]}'
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    assert digest(dst) == item['sha256']

records = DEST / '99_GitHub同步'
records.mkdir()
(records / '文件清单_20261028.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
local = [x for x in plan['files'] if x['action']=='local_only']
lines = ['# 保留在本地的二进制文件', '', '源目录：`' + str(SOURCE) + '`。下表路径均相对于此目录。', '',
         '模型权重、比赛提交 ZIP、归档/缓存 ZIP 未随 Git 仓库重复上传，原始文件没有删除或改动。完整上传/本地清单见同目录 JSON。', '',
         '| 相对路径 | 字节数 | SHA256 |', '|---|---:|---|']
lines += [f'| `{x["path"]}` | {x["bytes"]} | `{x["sha256"]}` |' for x in local]
(records / '本地大文件清单.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
(DEST / '.gitignore').write_text('''# Machine settings, credentials and regenerable caches
.claude/
.codex/
.agents/
.aws/
.ssh/
.env
.env.*
*.pem
*.key
__pycache__/
*.pyc
.venv/
.ruff_cache/
.pytest_cache/
# Local model weights and binary archives: see 99_GitHub同步
*.pt
*.pth
*.ckpt
*.safetensors
*.onnx
*.zip
*.7z
*.rar
*.tar
*.gz
''', encoding='utf-8')
(DEST / 'README.md').write_text('''# 无人机语义分割比赛资料（私有）

本仓库是桌面“无人机”文件夹于 **2026-10-28** 整理后的资料快照，供项目协作和分析使用。

## 从哪里开始

- [项目导航：从这里开始](README_从这里开始.md)
- [当前比赛、线上成绩与方案](01_当前比赛/)
- [代码与实验](02_代码与实验/)
- [历史文档与复盘](03_历史文档/)
- [审计与模板](04_审计与模板/)
- [提交包相关记录](提交包/)
- [目录整理记录与旧路径映射](90_整理记录/)

## 本次上传范围

原目录中的 853 个文档、代码、配置、报告及可视化文件按原内容上传，另生成本说明、忽略规则和清单。模型权重与 ZIP 等 292 个二进制文件留在本地，原始文件未改动；路径、大小和 SHA256 见[本地大文件清单](99_GitHub同步/本地大文件清单.md)。机器配置目录 `.claude` 未上传。

本仓库用于资料归档，未承诺直接克隆即可运行所有历史实验。历史文档中的绝对本机/服务器路径是原始记录，部分文件名中的“当前”“待提交”只代表记录当时状态。事实数据、用户报分、作者分析和未执行计划应分开判断。

BLDA-Soft 最新记录为完成教师缓存、CPU 检查及 50 步工程试跑；正式 C0/C1 训练未启动，不是已验证提分的方案。详见[执行状态](02_代码与实验/本轮实验/BLDA_Soft_执行_20261027/00_执行状态_20261028.md)。

完整快照清单：[文件清单](99_GitHub同步/文件清单_20261028.json)。仓库私有可见性在创建和上传后均需核验；本次上传不会自动同步之后的本地改动。
''', encoding='utf-8')
print(json.dumps({'snapshot':str(DEST),'original_files_copied':plan['summary']['included_files'],'copy_hashes_verified':True},ensure_ascii=False))
