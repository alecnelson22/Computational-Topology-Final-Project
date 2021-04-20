from pathlib import Path



folder_list = [
    # ('./data/BI_CORR', 16),
    # ('./data/CROSSING_90', 16),
    # ('./data/CROSSING_120', 16)
    ('./data/UNI_CORR_500', 25)
    ]

for (folder, fps) in folder_list:
    time_per_frame = 1/fps
    for path in Path(folder).rglob('*.txt'):

        txt_file = open(path)
        csv_filename = str(path).replace('.txt', '.csv')
        csv_file = open(csv_filename, 'w')

        lines = txt_file.readlines()
        data_rows = [x for x in lines if not x.startswith('#') and x.strip() != '']
        data_rows = [x.split() for x in data_rows]

        for row in data_rows:
            frame = int(row[1])
            time = frame * time_per_frame
            row.insert(0, str(time))

        row_strings = [','.join(row) + '\n' for row in data_rows]

        csv_file.write('time,id,Frame_25,x,y,z\n')
        # csv_file.write('time,number,frame_16,x,y,z,rot,id,flag\n')
        csv_file.writelines(row_strings)
        txt_file.close()
        csv_file.close()
        # print(path)