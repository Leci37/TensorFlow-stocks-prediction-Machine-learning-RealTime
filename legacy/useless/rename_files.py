import os
path_folder = r"C:\Users\Luis\Desktop\LecTrade\LecTrade\Models\TF_multi"
files_on_folder = os.listdir(path_folder)

paths_files = [filename for filename in files_on_folder if "Ñ" in  filename ]

for p in paths_files:
    print(p)
    os.rename(path_folder+"\\" +p, p.replace('Ñ',''))