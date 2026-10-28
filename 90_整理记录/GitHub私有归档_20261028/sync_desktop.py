"""Sync this operation's records and navigation, preserving previous entries."""
import json, shutil, hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parent
ROOT=Path(r'C:\Users\Lenovo\OneDrive\Documents\桌面\无人机')
DEST=ROOT/'90_整理记录'/'GitHub私有归档_20261028'
REPO=BASE.parent/'_github_private'/'repo'
check=json.loads((BASE/'file_verification.json').read_text(encoding='utf-8'))
assert check['status']=='PASS'
DEST.mkdir(parents=True,exist_ok=True)
report=f'''# GitHub私有仓库创建与上传记录

日期：2026-10-28（北京时间）

- 仓库：https://github.com/xiaomaomi721/uav-competition-private
- 账号：xiaomaomi721（已核对 GitHub 连接账号与本机 Git 凭据账号一致）。
- 可见性：**Private / 私有**，创建时指定 private=true，上传后通过 GitHub API 再次核验。
- 默认分支：codex/project-archive-20261028。
- 源目录：`{ROOT}`。
- 本地 Git 快照：`{REPO}`。
- 首轮提交：`{check['commit']}`。
- 首轮远端文件：857 个；所有路径与 Git blob SHA 均与本机提交一致。

## 上传范围

原目录的 853 个文档、代码、配置、实验报告、图片、PDF、Word、PPT 等文件（112917756 字节）已按原始内容上传；首轮另生成 README.md、.gitignore 和两份清单。后续提交附加本次操作记录和索引入口。

292 个模型权重与 ZIP 等二进制文件（2082517319 字节）按项目“大型数据/模型/缓存保留原位置”的要求留在本地，没有删除，路径、大小、SHA256 已收录。机器设置目录 .claude 未上传。文本及 Office XML 扫描未发现凭据候选；这不等于对所有图片/PDF进行完整安全审计。

完整清单在仓库 `99_GitHub同步/`。首次上传范围对应 `文件清单_20261028.json`，桌面索引后续增补属于本次操作记录，不改变历史研究结论。

## 使用与更新

这是一次手动快照，后续本地修改不会自动上传。以后更新应重新核对源目录变化、敏感文件与排除项，再提交推送。不要把模型、提交 ZIP 或机器凭据直接 git add -f。

本次未修改旧仓库 2856643972xsb/uav_semantic_segmentation，也未修改服务器文件、启动训练或消耗比赛提交机会。GitHub 插件不提供创建仓库和批量本地目录上传，使用本机 Git Credential Manager 的既有身份进行 GitHub API 创建和 Git 推送；令牌仅在进程内存使用，未输出或写入记录。
'''
(BASE/'私有仓库同步记录.md').write_text(report,encoding='utf-8')
names=['私有仓库同步记录.md','prepare_snapshot.py','build_snapshot.py','private_repo_api.py','verify_files.py','sync_desktop.py']
for name in names: shutil.copy2(BASE/name,DEST/name)
for source,dest in [('repository_created.json','仓库创建回执.json'),('repository_verified.json','首轮仓库核验.json'),('file_verification.json','首轮文件核验.json')]:
    shutil.copy2(BASE/source,DEST/dest)
for name,heading,section in [
    ('00_Codex工作同步索引.md','## GitHub私有归档', '\n## GitHub私有归档\n\n2026-10-28 已建立 [uav-competition-private 私有仓库](https://github.com/xiaomaomi721/uav-competition-private)，上传当前整理后的文档、代码与结果记录。权重和 ZIP 保留本地并在仓库登记哈希；详见[同步记录](90_整理记录/GitHub私有归档_20261028/私有仓库同步记录.md)。后续修改不会自动上传。\n'),
    ('README_从这里开始.md','## GitHub私有副本', '\n## GitHub私有副本\n\n[打开 GitHub 私有仓库](https://github.com/xiaomaomi721/uav-competition-private)。当前文档、代码和结果记录已上传，模型与 ZIP 保留在本地，见[本次同步记录](90_整理记录/GitHub私有归档_20261028/私有仓库同步记录.md)。后续修改需要另行同步。\n')]:
    p=ROOT/name
    text=p.read_text(encoding='utf-8-sig')
    if heading not in text:
        (BASE/(name+'.before')).write_text(text,encoding='utf-8')
        p.write_text(text.rstrip()+'\n'+section,encoding='utf-8')
    shutil.copy2(p,REPO/name)
for p in DEST.iterdir():
    if p.is_file():
        target=REPO/'90_整理记录'/'GitHub私有归档_20261028'/p.name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target)
index=BASE.parent/'00_Codex工作同步索引.md'
text=index.read_text(encoding='utf-8-sig')
if '## GitHub私有归档' not in text:
    index.write_text(text.rstrip()+'\n\n## GitHub私有归档\n\n2026-10-28 已创建并上传 [uav-competition-private](https://github.com/xiaomaomi721/uav-competition-private)，可见性已核验为 Private。[同步记录](C:/Users/Lenovo/OneDrive/Documents/桌面/无人机/90_整理记录/GitHub私有归档_20261028/私有仓库同步记录.md)。\n',encoding='utf-8')
print(json.dumps({'desktop_record_directory':str(DEST),'synced_record_files':len(list(DEST.iterdir())),'navigation_updated':True},ensure_ascii=False))
