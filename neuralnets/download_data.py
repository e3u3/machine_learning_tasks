import os
import glob
import numpy as np
import scipy.io as sio
import tarfile

db_folder = 'flowers'

if not os.path.isfile(os.path.join(db_folder, 'downloaded')):
  os.mkdir('flowers')

  print('Downloading data...')  
  tar_fn = os.path.join(db_folder, "oxfordflower17.tar")
  tar_url = "http://www.robots.ox.ac.uk/~vgg/data/bicos/data/oxfordflower17.tar"
  os.system('wget -O {} {}'.format(tar_fn, tar_url))
  
  print('Extracting data... (this may take a few minutes too)')
  tarfile.open(tar_fn).extractall(path=db_folder)

  print('Prepare data')
  image_fns = sorted(glob.glob(os.path.join(db_folder, 'oxfordflower17', 'jpg', '*.jpg')))
  lbl_fn = os.path.join(db_folder, 'oxfordflower17', 'imagelabels.mat')
  labels = sio.loadmat(lbl_fn)['labels'][0] - 1
  set_fn = os.path.join(db_folder, 'oxfordflower17', 'set.mat')
  val = []
  for lbl in range(17):
    val.extend([i for i in range(len(image_fns)) if labels[i] == lbl][:15])
  train = [i for i in range(len(image_fns)) if i not in val]

  class_names = ['daffodil', 'snowdrop',  'lily_valley', 'bluebell', 'crocus', 'iris', 'tigerlily', 'tulip', 'fritillary', 'sunflower', 'daisy', 'colts_foot', 'dandelion', 'cowslip', 'buttercup', 'windflower', 'pansy']

  for lbl in range(17):
    train_dir = os.path.join(db_folder, 'train', class_names[lbl])
    if not os.path.isdir(train_dir):
      os.makedirs(train_dir)
    fns = [image_fns[i] for i in train if labels[i]==lbl]
    print(class_names[lbl], 'train', len(fns))
    [os.system('cp {} {}'.format(fn, train_dir)) for fn in fns]

    val_dir = os.path.join(db_folder, 'val', class_names[lbl])
    if not os.path.isdir(val_dir):
      os.makedirs(val_dir) 
    fns = [image_fns[i] for i in val if labels[i]==lbl]
    print(class_names[lbl], 'val', len(fns))
    [os.system('cp {} {}'.format(fn, val_dir)) for fn in fns]

  open(os.path.join(db_folder, 'downloaded'), 'w').write('')

