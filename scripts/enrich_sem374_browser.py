#!/usr/bin/env python3
"""Add sparse-depth checks and target overlays to the local comparison artifact."""
import json,html,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_sem373_joint import write,digest
b=ROOT/'outputs/quality/SEM-374';s=json.loads((b/'summary.json').read_text());p=b/'browser-r1/index.html';page=p.read_text()
headers=['系统','点角色','观测数','平均绝对相对逆深度误差','平均alpha'];rows=[]
for a,label in [('baseline','原始RGB130'),('control','控制'),('joint','联合')]:
 for k,v in s['depth'][a].items():rows.append([label,k,str(v['count']),f"{v['absolute_relative_inverse_depth']:.8f}",f"{v['mean_alpha']:.6f}"])
block='<h2>多视角稀疏结构检查</h2><p>fit点参与新增深度损失，check点不参与该损失；二者都来自原初始化COLMAP，均不是独立三维GT。误差为 |预测期望逆Z−参考逆Z|/参考逆Z，按观测平均；更低不等于真实几何一定更好。类别1=deck、3=tower、4=foundation。</p><table><tr>'+''.join('<th>'+x+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
block+='<h2>实际结构监督位置</h2><p>绿色＋为fit，青色×为check。左照片、右人工标签。原生坐标、多视角一致点；索面和背景未生成深度目标。点击查看完整分辨率。</p>'
for n in ['159','160','140','146','081','057','064','054','174']:block+=f'<details><summary>{n}.png 结构点</summary><a href="../track-panels-final/{n}.png"><img style="max-width:100%" src="../track-panels-final/{n}.png" loading="lazy"></a></details>'
block+='<p><a href="../summary.json">完整数值汇总</a> · <a href="../final-audit.json">最终核验</a> · <a href="../tracks-independent-audit.json">结构点独立核验</a></p>'
if (b/'visual_review.md').exists() and html.escape((b/'visual_review.md').read_text()) not in page:block+='<h2>最终视觉复核</h2><div style="white-space:pre-wrap">'+html.escape((b/'visual_review.md').read_text())+'</div>'
page=page.replace('</html>',block+'</html>');p.write_text(page);write(b/'browser-r1/final-presentation.json',dict(index_sha256=digest(p),summary_sha256=digest(b/'summary.json')))
