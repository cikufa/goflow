"""Extract actual video frames into a labeled contact sheet, without simulator changes."""
import argparse
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument('video',type=Path)
parser.add_argument('--output',required=True,type=Path)
parser.add_argument('--title',default='Recorded simulator frames')
parser.add_argument('--raw-dir',type=Path)
args=parser.parse_args()
reader=imageio.get_reader(str(args.video))
count=reader.count_frames()
indices=sorted(set(round(i*(count-1)/5) for i in range(6)))
sheet=Image.new('RGB',(1280,3*390+40),'white')
draw=ImageDraw.Draw(sheet)
draw.text((12,12),args.title,fill='black')
if args.raw_dir:
    args.raw_dir.mkdir(parents=True,exist_ok=True)
    for index,frame in enumerate(reader):
        imageio.imwrite(args.raw_dir/f'frame_{index:05d}.png',frame)
for panel,index in enumerate(indices):
    frame=Image.fromarray(reader.get_data(index)).convert('RGB')
    x,y=(panel%2)*640,(panel//2)*390+40
    draw.text((x+10,y+8),f'Frame {index} / {count-1}',fill='black')
    sheet.paste(frame.resize((640,360)),(x,y+30))
args.output.parent.mkdir(parents=True,exist_ok=True)
sheet.save(args.output)
reader.close()
print(args.output)
