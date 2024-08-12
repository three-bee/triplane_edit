import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--indir", type=str, required=True)

def ds_imgs(folder):
    n_p = os.path.join(folder,'resized')
    os.makedirs(n_p, exist_ok=False)

    for p in os.listdir(folder):
        if p.split('.')[-1] == 'png':
            img = cv2.resize(cv2.imread(os.path.join(folder, p)), (256,256))
            cv2.imwrite(os.path.join(n_p, p.split('.')[-2]+'.jpg'), img)
            print(p)

if __name__ == '__main__':
    args = parser.parse_args()
    ds_imgs(args.indir)