#!/usr/bin/env python3
"""Post-hoc qualitative enlargements; does not define or change scoring ROIs."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--review', type=Path, default=ROOT/'outputs/quality/rgb144/review-r1')
    a = p.parse_args()
    data = json.loads((a.review/'data.json').read_text())
    boxes = {**{f'frame{i:04d}': [400, 0, 1320, 450] for i in [71, 72, 73]},
             **{f'frame{i:04d}': [600, 300, 1320, 989] for i in [96, 97, 98]}}
    output = a.review/'trajectory_details'
    output.mkdir(exist_ok=False)
    sections = []
    for stem, box in boxes.items():
        w, h = box[2]-box[0], box[3]-box[1]
        panel = Image.new('RGB', (3*w, h+26), 'white')
        draw = ImageDraw.Draw(panel)
        for i, tile in enumerate(data['tiles'][1:]):
            draw.text((i*w+6, 6), tile['label'], fill='black')
            with Image.open(a.review/tile['path']/(stem+'.png')) as image:
                panel.paste(image.crop(box), (i*w, 26))
        panel.save(output/(stem+'.png'))
        sections.append(f'<h2>{stem} · {box}</h2><a href="{stem}.png"><img src="{stem}.png" style="max-width:100%"></a>')
    (output/'boxes.json').write_text(json.dumps({'scope': 'post-hoc qualitative display only; no GT; no new scoring ROI', 'boxes': boxes}, indent=2)+'\n')
    (output/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>轨迹重点放大</title><body style="font:16px system-ui;margin:24px"><h1>轨迹重点放大 · RGB140 / RGB145 / RGB146</h1><p>相同帧和像素框；事后定性展示，不参与指标计算。无真值，不能据此宣称准确率或闪烁改善。</p><a href="../index.html">返回完整对照</a>'+''.join(sections)+'</body></html>')


if __name__ == '__main__':
    main()
