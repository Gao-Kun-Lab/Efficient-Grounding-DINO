#!/usr/bin/env python3
from pathlib import Path

root = Path('WORK_DIR/detection_data')  # 众多数据集的存储路径
for d in sorted(root.rglob('')):
    if d.is_dir():
        print(d.relative_to(root))
        # 当前目录下前 3 个普通文件
        for f in sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name))[:3]:
            if f.is_file():
                print('    ', f.name)