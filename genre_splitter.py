import numpy as np
import shutil

data = np.genfromtxt('./fma_small_genres.csv', delimiter=',', dtype=str)

target_genres = ["  'Hip-Hop'", "  'Folk'", "  'Rock'"]
songs_list = {
  "  'Hip-Hop'": [],
  "  'Folk'": [],
  "  'Rock'": []
}

for row in data:
  genre = row[1]
  if genre in target_genres:
    songs_list[genre].append(row[0])
  

for item in songs_list:
  song_ids = songs_list[item]
  if item == "  'Hip-Hop'":
    output_dir = "data/fma_hip-hop"
  elif item == "  'Folk'":
    output_dir = "data/fma_folk"
  else:
    output_dir = "data/fma_rock"
  
  for id in song_ids:
    parent_dir = id[:3]
    shutil.copy(f'./data/fma_small/{parent_dir}/{id}', f'{output_dir}/{id}')